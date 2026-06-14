"""
Select stage: step1/step2/step3 filters on ``video/``, then ingest passing clips.

Rejected videos are discarded (not moved, not written to manifest).
Select completes after step3 (``stages_completed`` includes ``select``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pipeline.dataset_schema import DIR_PROCESSED, sample_dir_rel, sample_video_rel
from pipeline.llm_defaults import DEFAULT_LLM_BASE_URL, DEFAULT_SELECT_VLM_MODEL
from pipeline.hub import resolve_select_input_dir
from pipeline.manifest import (
    DEFAULT_MANIFEST_NAME,
    apply_select_update,
    load_manifest_by_id,
    manifest_path,
    merge_preserved_fields,
    new_row_template,
    save_manifest,
    try_load_legacy_manifest,
)
from pipeline.stages.select.filters.common import (
    DEFAULT_SELECT_YOLO_PATH,
    SelectFilterConfig,
    resolve_yolo_model,
)
from pipeline.parallel_defaults import DEFAULT_STAGE_WORKERS
from pipeline.sample_renumber import renumber_sample_ids
from pipeline.stage_timing import stage_progress_set_total, stage_progress_update
from pipeline.stages.select.filters.pipeline import run_select_filters
from pipeline.stages.select.filters.step3_vlm import create_select_vlm_client

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass(frozen=True)
class _SelectFilterOutcome:
    idx: int
    video_path: Path
    src_rel: str
    status: Literal["skipped", "rejected", "passed"]
    already_mapped: bool


def _video_src_rel(work_root: Path, video_path: Path) -> str:
    try:
        return str(video_path.relative_to(work_root))
    except ValueError:
        return str(video_path.resolve())


def _filter_one_video(
    *,
    idx: int,
    video_path: Path,
    src_rel: str,
    already_mapped: bool,
    overwrite: bool,
    filter_cfg: SelectFilterConfig,
    skip_filters: bool,
    skip_vlm: bool,
    vlm_client: Any,
) -> _SelectFilterOutcome:
    if already_mapped and not overwrite:
        return _SelectFilterOutcome(
            idx=idx,
            video_path=video_path,
            src_rel=src_rel,
            status="skipped",
            already_mapped=True,
        )

    filter_result = run_select_filters(
        video_path,
        cfg=filter_cfg,
        skip_step1_step2=skip_filters,
        skip_step3=skip_vlm,
        vlm_client=vlm_client,
    )
    status: Literal["rejected", "passed"] = (
        "rejected" if filter_result.status == "rejected" else "passed"
    )
    return _SelectFilterOutcome(
        idx=idx,
        video_path=video_path,
        src_rel=src_rel,
        status=status,
        already_mapped=already_mapped,
    )


def _row_is_ingested(row: Dict[str, Any]) -> bool:
    vp = str(row.get("video_path", "")).replace("\\", "/").strip()
    if vp.startswith(f"{DIR_PROCESSED}/"):
        return True
    return str(row.get("select_status", "")).strip() == "passed"


def _parse_sample_id_numeric(sample_id: str) -> Optional[int]:
    if sample_id.isdigit():
        return int(sample_id)
    return None


def _load_id_mapping(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("items") or [])


def _max_id_from_mapping_and_manifest(
    id_mapping: List[Dict[str, str]],
    prev_by_id: Dict[str, Dict[str, Any]],
    id_width: int,
) -> int:
    m = 0
    for item in id_mapping:
        n = _parse_sample_id_numeric(str(item.get("sample_id", "")))
        if n is not None:
            m = max(m, n)
    for sid in prev_by_id:
        n = _parse_sample_id_numeric(sid)
        if n is not None:
            m = max(m, n)
    return m


def _collect_videos(input_dir: Path) -> List[Path]:
    found: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            found.append(p)
    return found


def _build_filter_config(args: argparse.Namespace) -> SelectFilterConfig:
    return SelectFilterConfig(
        min_duration_s=float(getattr(args, "select_min_duration", 1.0)),
        max_duration_s=float(getattr(args, "select_max_duration", 120.0)),
        min_side_px=int(getattr(args, "select_min_side", 240)),
        step1_sample_frames=int(getattr(args, "select_step1_frames", 12)),
        step2_sample_frames=int(getattr(args, "select_step2_frames", 16)),
        yolo_model=str(getattr(args, "select_yolo_model", DEFAULT_SELECT_YOLO_PATH)),
        vlm_model=str(getattr(args, "select_vlm_model", DEFAULT_SELECT_VLM_MODEL)),
        vlm_frames=int(getattr(args, "select_vlm_frames", 6)),
        vlm_max_side=int(getattr(args, "select_vlm_max_side", 512)),
        vlm_vision_detail=str(getattr(args, "select_vlm_vision_detail", "low")),
        vlm_timeout=float(getattr(args, "select_vlm_timeout", 120.0)),
        vlm_max_retries=int(getattr(args, "select_vlm_max_retries", 2)),
        vlm_base_url=str(getattr(args, "select_vlm_base_url", DEFAULT_LLM_BASE_URL)),
        vlm_http_referer=str(getattr(args, "select_vlm_http_referer", "") or ""),
        vlm_x_title=str(getattr(args, "select_vlm_x_title", "video2smpl-select-vlm")),
    )


def _ingest_video(
    *,
    work_root: Path,
    sample_id: str,
    src_video: Path,
    use_symlink: bool,
    overwrite: bool,
) -> Path:
    """Place video under processed_trainable_data/<sample_id>/; return final path."""
    dest_dir = work_root / sample_dir_rel(sample_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_video.name
    if dest.resolve() == src_video.resolve():
        return dest
    if dest.exists():
        if not overwrite:
            return dest
        dest.unlink()
    if use_symlink:
        os.symlink(str(src_video.resolve()), str(dest))
    else:
        shutil.move(str(src_video), str(dest))
    return dest


def run(args: argparse.Namespace) -> None:
    if args.id_width < 1:
        raise ValueError("--id_width must be >= 1")

    source = str(args.source).strip()
    if not source:
        raise ValueError('--source is required for the "select" stage.')

    work_root = Path(args.root_dir).resolve()
    input_dir = resolve_select_input_dir(work_root, getattr(args, "select_input_dir", None))
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Select input directory not found: {input_dir}. "
            f"Place raw videos under {work_root / 'video'} or pass --select-input-dir."
        )

    use_symlink = bool(getattr(args, "select_symlink", False))
    skip_filters = bool(getattr(args, "select_skip_filters", False))
    skip_vlm = bool(getattr(args, "select_skip_vlm", False))
    filter_cfg = _build_filter_config(args)

    yolo_resolved = resolve_yolo_model(filter_cfg.yolo_model)
    print(f"Select step2 YOLO weights: {yolo_resolved}")

    vlm_client = None
    if not skip_vlm:
        vlm_client = create_select_vlm_client(filter_cfg)
        print(
            f"Select step3 VLM: model={filter_cfg.vlm_model} "
            f"frames={filter_cfg.vlm_frames} detail={filter_cfg.vlm_vision_detail}"
        )
    else:
        print("Select step3 VLM: skipped (--select-skip-vlm); select stage will NOT be marked complete.")

    workers = max(1, int(getattr(args, "select_workers", DEFAULT_STAGE_WORKERS)))
    print(f"Select workers: {workers}")

    mapping_path = work_root / args.mapping_name
    out_manifest = manifest_path(work_root, args.manifest_name)
    link_default = str(getattr(args, "link", None) or getattr(args, "default_link", "") or "")

    id_mapping: List[Dict[str, str]] = _load_id_mapping(mapping_path)
    path_to_sample: Dict[str, str] = {}
    for item in id_mapping:
        rel = item.get("original_path_relative")
        sid = item.get("sample_id")
        if rel and sid:
            path_to_sample[rel] = sid

    if out_manifest.exists():
        prev_by_id = load_manifest_by_id(out_manifest)
    else:
        prev_by_id = {
            str(r["sample_id"]): r
            for r in try_load_legacy_manifest(work_root, args.manifest_name)
            if r.get("sample_id")
        }

    next_id = _max_id_from_mapping_and_manifest(id_mapping, prev_by_id, args.id_width) + 1
    manifest_rows: Dict[str, Dict[str, Any]] = dict(prev_by_id)
    added = 0
    skipped = 0
    rejected = 0

    videos = _collect_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir} (recursive {VIDEO_SUFFIXES})")

    work_items: List[tuple[int, Path, str, bool]] = []
    for idx, video_path in enumerate(videos, start=1):
        src_rel = _video_src_rel(work_root, video_path)
        already_mapped = src_rel in path_to_sample
        work_items.append((idx, video_path, src_rel, already_mapped))

    stage_progress_set_total(len(work_items))
    outcomes: List[_SelectFilterOutcome] = []
    progress_lock = threading.Lock()
    done_filters = 0

    def _run_filter(item: tuple[int, Path, str, bool]) -> _SelectFilterOutcome:
        idx, video_path, src_rel, already_mapped = item
        return _filter_one_video(
            idx=idx,
            video_path=video_path,
            src_rel=src_rel,
            already_mapped=already_mapped,
            overwrite=bool(args.overwrite),
            filter_cfg=filter_cfg,
            skip_filters=skip_filters,
            skip_vlm=skip_vlm,
            vlm_client=vlm_client,
        )

    if workers == 1:
        for item in work_items:
            idx, video_path, _, _ = item
            stage_progress_update(
                done=idx - 1,
                total=len(work_items),
                item=video_path.name,
                note="selecting",
            )
            outcome = _run_filter(item)
            outcomes.append(outcome)
            stage_progress_update(done=idx, item=video_path.name, note=outcome.status)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_run_filter, item): item for item in work_items}
            for fut in as_completed(future_map):
                item = future_map[fut]
                idx, video_path, _, _ = item
                outcome = fut.result()
                with progress_lock:
                    outcomes.append(outcome)
                    done_filters += 1
                    stage_progress_update(
                        done=done_filters,
                        total=len(work_items),
                        item=video_path.name,
                        note=outcome.status,
                    )

    outcomes.sort(key=lambda o: o.idx)
    select_complete = not skip_vlm

    for outcome in outcomes:
        if outcome.status == "skipped":
            skipped += 1
            continue
        if outcome.status == "rejected":
            rejected += 1
            continue

        video_path = outcome.video_path
        src_rel = outcome.src_rel
        already_mapped = outcome.already_mapped

        if already_mapped and args.overwrite:
            sid = path_to_sample[src_rel]
            id_mapping = [it for it in id_mapping if it.get("original_path_relative") != src_rel]
        else:
            sid = f"{next_id:0{args.id_width}d}"
            next_id += 1
            added += 1

        dest_video = _ingest_video(
            work_root=work_root,
            sample_id=sid,
            src_video=video_path,
            use_symlink=use_symlink,
            overwrite=args.overwrite,
        )
        try:
            video_rel = str(dest_video.relative_to(work_root))
        except ValueError:
            video_rel = sample_video_rel(sid, dest_video.name)

        id_mapping.append(
            {
                "sample_id": sid,
                "seq_index": "",
                "original_filename": video_path.name,
                "original_stem": video_path.stem,
                "original_path_relative": src_rel,
                "output_sample_dir": sample_dir_rel(sid),
            }
        )
        path_to_sample[src_rel] = sid

        old = manifest_rows.get(sid, {})
        link_val = old.get("link")
        if link_val is None or str(link_val).strip() == "":
            link_val = link_default

        base = new_row_template(
            sample_id=sid,
            original_video=dest_video.name,
            source=source,
            link=str(link_val),
        )
        updated = apply_select_update(
            base,
            video_path=video_rel,
            select_status="passed",
            select_notes="",
            original_video_path=src_rel,
            mark_complete=select_complete,
        )
        manifest_rows[sid] = merge_preserved_fields(updated, old)

    ordered = [
        manifest_rows[sid]
        for sid in sorted(manifest_rows.keys(), key=lambda s: int(s) if s.isdigit() else 0)
        if sid.isdigit() and _row_is_ingested(manifest_rows[sid])
    ]
    ingested_ids = {str(r.get("sample_id", "")).strip() for r in ordered if r.get("sample_id")}
    id_mapping = [it for it in id_mapping if str(it.get("sample_id", "")).strip() in ingested_ids]

    ordered, id_mapping, id_remap = renumber_sample_ids(
        work_root,
        ordered,
        id_mapping,
        id_width=args.id_width,
    )
    changed = sum(1 for old, new in id_remap.items() if old != new)
    if changed:
        last_id = f"{len(ordered):0{args.id_width}d}" if ordered else "0"
        print(f"  Renumbered sample_id: {changed} dir(s) -> contiguous 000001..{last_id}")

    save_manifest(out_manifest, ordered)

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

    mode = "symlink" if use_symlink else "move"
    print(f"Select done. Scanned: {len(videos)} ({mode})")
    print(f"  Ingested: {added}, rejected: {rejected}, skipped (mapped): {skipped}")
    if not skip_vlm:
        print("  Select stage marked complete (stages_completed includes 'select').")
    print(f"  Manifest: {out_manifest}")
    print(f"  Mapping: {mapping_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select stage: filter and ingest videos")
    parser.add_argument("--root_dir", type=str, default="examples/training")
    parser.add_argument("--manifest_name", type=str, default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--mapping_name", type=str, default="sample_id_to_source.json")
    parser.add_argument("--id_width", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--select-input-dir",
        type=str,
        default=None,
        help="Raw video folder (default: <root_dir>/video). Recursive mp4/mov/avi/mkv.",
    )
    parser.add_argument(
        "--select-symlink",
        action="store_true",
        help="Symlink instead of move when placing video under processed_trainable_data/.",
    )
    parser.add_argument(
        "--select-skip-filters",
        action="store_true",
        help="Skip step1/step2; still runs step3 unless --select-skip-vlm.",
    )
    parser.add_argument(
        "--select-skip-vlm",
        action="store_true",
        help="Skip step3 VLM (select stage will not be marked complete).",
    )
    parser.add_argument(
        "--select-yolo-model",
        type=str,
        default=DEFAULT_SELECT_YOLO_PATH,
        help=f"YOLO weights for step2 (default: {DEFAULT_SELECT_YOLO_PATH}).",
    )
    parser.add_argument("--select-min-duration", type=float, default=1.0)
    parser.add_argument("--select-max-duration", type=float, default=120.0)
    parser.add_argument("--select-min-side", type=int, default=240)
    parser.add_argument("--select-step1-frames", type=int, default=12)
    parser.add_argument("--select-step2-frames", type=int, default=16)
    parser.add_argument(
        "--select-vlm-model",
        type=str,
        default=DEFAULT_SELECT_VLM_MODEL,
        help="VLM model for step3 fine check.",
    )
    parser.add_argument("--select-vlm-frames", type=int, default=6)
    parser.add_argument("--select-vlm-max-side", type=int, default=512)
    parser.add_argument(
        "--select-vlm-vision-detail",
        type=str,
        default="low",
        choices=("low", "high", "auto", "original"),
    )
    parser.add_argument("--select-vlm-timeout", type=float, default=120.0)
    parser.add_argument("--select-vlm-max-retries", type=int, default=2)
    parser.add_argument(
        "--select-vlm-base-url",
        type=str,
        default=DEFAULT_LLM_BASE_URL,
    )
    parser.add_argument("--select-vlm-http-referer", type=str, default="")
    parser.add_argument("--select-vlm-x-title", type=str, default="video2smpl-select-vlm")
    parser.add_argument(
        "--select-workers",
        type=int,
        default=DEFAULT_STAGE_WORKERS,
        help=(
            f"Parallel filter workers per video (default: {DEFAULT_STAGE_WORKERS}). "
            "Use 1 for strictly serial filters."
        ),
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--link", type=str, default=None)
    parser.add_argument("--default_link", type=str, default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
