"""video2smpl stage: dispatch to prompthmr (default) or camerahmr backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from pipeline.dataset_schema import (
    HMR_BACKEND_CAMERAHMR,
    HMR_BACKEND_PROMPTHMR,
    sample_paths,
    smpl_filename_for_backend,
)
from pipeline.manifest import (
    build_video2smpl_row,
    captions_filled,
    get_hmr_text_prompt,
    load_manifest_list,
    manifest_path,
    resolve_video_abs,
    resolve_video_rel,
    rows_caption_complete,
    rows_pending_smpl,
    save_manifest,
    smpl_filled_for_backend,
    try_load_legacy_manifest,
)
from pipeline.stages.video2smpl.backends.camerahmr import run_camerahmr_sample
from pipeline.stages.video2smpl.backends.prompthmr import run_prompthmr_sample
from pipeline.stages.video2smpl.common import (
    extract_first_frame,
    load_id_mapping,
    max_sample_id_from_dirs,
    normalize_hmr_backend,
    parse_sample_id_numeric,
    resolve_manifest_link,
)
def run(args: argparse.Namespace) -> None:
    if args.id_width < 1:
        raise ValueError("--id_width must be >= 1")

    backend = normalize_hmr_backend(getattr(args, "hmr_backend", HMR_BACKEND_PROMPTHMR))
    manifest_source = str(args.source).strip()
    if not manifest_source:
        raise ValueError('--source is required (manifest "source" label).')

    if backend == HMR_BACKEND_CAMERAHMR:
        wr = str(getattr(args, "weight_root", "") or "").strip()
        if wr:
            os.environ["VIDEO2SMPL_WEIGHT_ROOT"] = str(Path(wr).expanduser().resolve())
        else:
            os.environ["VIDEO2SMPL_WEIGHT_ROOT"] = ""
    else:
        from pipeline.stages.video2smpl.prompthmr_weights import check_weights

        vendor = getattr(args, "prompthmr_vendor", None)
        ckpt = getattr(args, "prompthmr_ckpt_root", None)
        require_slam = not bool(getattr(args, "static_camera", True))
        ok, missing = check_weights(vendor, ckpt, require_slam=require_slam)
        if not ok:
            raise FileNotFoundError(
                "PromptHMR weights incomplete. Missing:\n"
                + "\n".join(missing)
                + "\nRun: bash scripts/copy_prompthmr_vendor.sh"
            )

    work_root = Path(args.root_dir).resolve()
    manifest_link = resolve_manifest_link(args)
    out_trainable = work_root / "processed_trainable_data"
    out_manifest = manifest_path(work_root, args.manifest_name)

    if not out_manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {out_manifest}. Run the select stage first."
        )

    mapping_path = work_root / args.mapping_name
    id_mapping: List[Dict[str, str]] = load_id_mapping(mapping_path)
    path_to_sample: Dict[str, str] = {}
    for item in id_mapping:
        rel = item.get("original_path_relative")
        sid = item.get("sample_id")
        if rel and sid:
            path_to_sample[rel] = sid

    for item in id_mapping:
        n = parse_sample_id_numeric(item.get("sample_id", ""))
        if n is not None:
            pass
    max_sample_id_from_dirs([out_trainable], args.id_width)

    manifest_list = load_manifest_list(out_manifest)
    if not manifest_list:
        legacy = try_load_legacy_manifest(work_root, args.manifest_name)
        if legacy:
            manifest_list = legacy
        else:
            raise ValueError(f"Manifest is empty: {out_manifest}")

    prev_manifest = {str(r["sample_id"]): r for r in manifest_list if r.get("sample_id")}

    caption_complete = rows_caption_complete(manifest_list)
    if not caption_complete:
        raise ValueError(
            f"No caption-complete samples in {out_manifest}. Run captions first."
        )

    pending_before = rows_pending_smpl(manifest_list)
    print(
        f"video2smpl: backend={backend}, caption-complete={len(caption_complete)}, "
        f"pending_smpl={len(pending_before)}",
        flush=True,
    )

    vendor_root = Path(args.vendor_root).resolve()
    prompthmr_vendor = getattr(args, "prompthmr_vendor", None)

    processed = 0
    skipped_done = 0
    skipped_no_caption = 0
    errors = 0

    work_items = sorted(
        caption_complete,
        key=lambda r: int(r["sample_id"]) if str(r.get("sample_id", "")).isdigit() else 0,
    )

    for row in work_items:
        sample_id = str(row["sample_id"])
        video_rel = resolve_video_rel(row)
        if not video_rel:
            print(f"WARN: skip {sample_id}: missing video_path")
            errors += 1
            continue

        if smpl_filled_for_backend(row, backend) and not args.overwrite:
            skipped_done += 1
            continue

        text_prompt = ""
        if backend == HMR_BACKEND_PROMPTHMR:
            if not captions_filled(row):
                skipped_no_caption += 1
                print(f"WARN: skip {sample_id}: caption/action_caption missing (required for prompthmr)")
                continue
            text_prompt = get_hmr_text_prompt(row)
            if not text_prompt:
                skipped_no_caption += 1
                print(f"WARN: skip {sample_id}: empty caption text for prompthmr")
                continue

        try:
            video_path = resolve_video_abs(work_root, row)
        except ValueError as e:
            print(f"WARN: skip {sample_id}: {e}")
            errors += 1
            continue

        if not video_path.is_file():
            print(f"WARN: skip {sample_id}: video not found: {video_path}")
            errors += 1
            continue

        rel = video_rel
        if rel not in path_to_sample:
            id_mapping.append(
                {
                    "sample_id": sample_id,
                    "seq_index": "",
                    "original_filename": row.get("original_video") or video_path.name,
                    "original_stem": video_path.stem,
                    "original_path_relative": rel,
                    "output_sample_dir": sample_id,
                }
            )
            path_to_sample[rel] = sample_id

        sample_train = out_trainable / sample_id
        sample_train.mkdir(parents=True, exist_ok=True)
        smpl_name = smpl_filename_for_backend(backend)
        smpl_npz_path = sample_train / smpl_name

        try:
            if backend == HMR_BACKEND_CAMERAHMR:
                run_camerahmr_sample(
                    video_path=video_path,
                    output_npz=smpl_npz_path,
                    args=args,
                    vendor_root=vendor_root,
                )
            else:
                run_prompthmr_sample(
                    video_path=video_path,
                    output_npz=smpl_npz_path,
                    text_prompt=text_prompt,
                    args=args,
                    vendor_root=prompthmr_vendor,
                )
        except Exception as e:
            print(f"WARN: skip {sample_id}: {backend} failed: {e}")
            errors += 1
            continue

        paths = sample_paths(sample_id, video_path.name, hmr_backend=backend)
        extract_first_frame(video_path, sample_train / "first_frame.jpg")

        old = prev_manifest.get(sample_id, row)
        link_val = old.get("link")
        if link_val is None or str(link_val).strip() == "":
            link_val = manifest_link
        prev_manifest[sample_id] = build_video2smpl_row(
            sample_id=sample_id,
            original_video=str(row.get("original_video") or video_path.name),
            video_rel=paths["video_path"],
            first_frame_rel=paths["first_frame"],
            smpl_rel=paths["smpl_path"],
            smpl_backend=backend,
            source=manifest_source,
            link=str(link_val),
            old_row=old,
        )
        processed += 1

    id_mapping.sort(key=lambda it: int(it["sample_id"]) if it.get("sample_id", "").isdigit() else 0)
    for i, item in enumerate(id_mapping, start=1):
        item["seq_index"] = str(i)

    manifest_out = sorted(
        prev_manifest.values(),
        key=lambda r: int(r["sample_id"]) if str(r.get("sample_id", "")).isdigit() else 0,
    )
    save_manifest(out_manifest, manifest_out)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "root_dir": str(work_root),
                "id_width": args.id_width,
                "count": len(id_mapping),
                "items": id_mapping,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pending_after = rows_pending_smpl(load_manifest_list(out_manifest))
    if pending_after:
        print(
            f"WARN: {len(pending_after)} caption-complete sample(s) still lack smpl_path.",
            flush=True,
        )

    print(
        f"Done. backend={backend}, processed={processed}, "
        f"skipped_done={skipped_done}, skipped_no_caption={skipped_no_caption}, errors={errors}"
    )
    print(f"Manifest: {out_manifest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="video2smpl stage (dual HMR backend)")
    parser.add_argument("--root_dir", type=str, default="examples/training")
    parser.add_argument(
        "--hmr-backend",
        type=str,
        default=HMR_BACKEND_PROMPTHMR,
        choices=[HMR_BACKEND_PROMPTHMR, HMR_BACKEND_CAMERAHMR],
        help="HMR backend (default: prompthmr).",
    )
    parser.add_argument(
        "--weight_root",
        type=str,
        default="/data1/wjh/Video2SMPL",
        help="CameraHMR weights root (camerahmr backend only).",
    )
    parser.add_argument("--vendor_root", type=str, default="third_party")
    parser.add_argument(
        "--prompthmr-vendor",
        type=str,
        default=None,
        help="PromptHMR vendor_bundle dir (default: pipeline/stages/video2smpl/vendor_bundle).",
    )
    parser.add_argument(
        "--prompthmr-ckpt-root",
        type=str,
        default="/data1/wjh/ckpt/PromptHMR",
        help="Checkpoint root for weight validation.",
    )
    parser.add_argument(
        "--static-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PromptHMR static camera (default on).",
    )
    parser.add_argument("--manifest_name", type=str, default="dataset_manifest.json")
    parser.add_argument("--mapping_name", type=str, default="sample_id_to_source.json")
    parser.add_argument("--id_width", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--person_idx", type=int, default=0)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument(
        "--set-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CameraHMR DART floor (camerahmr only).",
    )
    parser.add_argument("--use_shape", action="store_true")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--link", type=str, default=None)
    parser.add_argument("--default_link", type=str, default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
