import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


def _setup_vendor_imports(vendor_root: Path) -> None:
    root_str = str(vendor_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _group_consecutive_frame_ids(frame_ids: torch.Tensor) -> List[List[int]]:
    if frame_ids.numel() == 0:
        return []
    grouped: List[List[int]] = []
    current = [int(frame_ids[0].item())]
    for value in frame_ids[1:]:
        idx = int(value.item())
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            grouped.append(current)
            current = [idx]
    grouped.append(current)
    return grouped


def _extract_first_frame(video_path: Path, output_jpg: Path) -> None:
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from video: {video_path}")
    cv2.imwrite(str(output_jpg), frame)


def _copy_rgb_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def _parse_sample_id_numeric(sample_id: str) -> Optional[int]:
    if sample_id.isdigit():
        return int(sample_id)
    return None


def _max_sample_id_from_dirs(bases: List[Path], id_width: int) -> int:
    m = 0
    for base in bases:
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if name.isdigit() and len(name) == id_width:
                m = max(m, int(name))
    return m


def _load_id_mapping(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("items") or [])


def _load_manifest_preserve(path: Path) -> Dict[str, Dict[str, str]]:
    """sample_id -> previous row (for merging text/source/link)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sid = row.get("sample_id")
        if sid:
            out[sid] = row
    return out


def _canonical_smpl_to_npz_dict(
    smpl_params_canonical: Dict[str, torch.Tensor],
    person_idx: int,
    num_persons: int,
) -> Dict[str, np.ndarray]:
    """If multi-person, select `person_idx`; all arrays float32 numpy for np.savez."""
    out: Dict[str, np.ndarray] = {}
    for key, val in smpl_params_canonical.items():
        if val is None:
            continue
        t = val.detach().cpu()
        if num_persons > 1:
            t = t[person_idx]
        out[key] = t.float().numpy().astype(np.float32)
    return out


def _betas_np_from_smoothed(
    smpl_smooth: Dict[str, torch.Tensor],
    T_canonical: int,
) -> np.ndarray:
    """Build (T, 10) float32 betas for canonical NPZ from smoothed incam tensors."""
    b = smpl_smooth.get("betas")
    if b is None:
        return np.zeros((T_canonical, 10), dtype=np.float32)
    t = b.detach().cpu().float()
    if t.ndim == 1:
        if t.numel() != 10:
            raise ValueError(f"Expected betas (10,), got {tuple(t.shape)}")
        row = t.numpy().astype(np.float32)
        return np.tile(row[None, :], (T_canonical, 1))
    if t.ndim != 2 or t.shape[1] != 10:
        raise ValueError(f"Expected betas (T,10), got {tuple(t.shape)}")
    arr = t.numpy().astype(np.float32)
    if arr.shape[0] == T_canonical:
        return arr
    if arr.shape[0] > T_canonical:
        return arr[:T_canonical]
    pad = np.tile(arr[-1:], (T_canonical - arr.shape[0], 1))
    return np.concatenate([arr, pad], axis=0)


def _smooth_smpl_for_one_person(
    person_smpl_params: Dict[str, torch.Tensor],
    frame_mask: Optional[torch.Tensor],
    smooth_window: int,
    echo_module,
) -> Dict[str, torch.Tensor]:
    seq_len = person_smpl_params["global_orient"].shape[0]
    if frame_mask is None:
        missing_groups: List[List[int]] = []
    else:
        mask = frame_mask[:seq_len].bool()
        missing_frame_ids = torch.where(~mask)[0]
        missing_groups = _group_consecutive_frame_ids(missing_frame_ids)

    betas = person_smpl_params.get("betas")
    smpl_dict = {
        "global_orient": person_smpl_params["global_orient"],
        "body_pose": person_smpl_params["body_pose"],
        "transl": person_smpl_params["transl"],
        "betas": betas if betas is not None else None,
    }
    smpl_6d = echo_module.smpl_dict_to_rot6d(smpl_dict)
    if missing_groups:
        smpl_6d = echo_module.linear_interpolate_frame_ids(smpl_6d, missing_groups)
    if smooth_window > 0:
        smpl_6d = echo_module.smooth_motion_rep(smpl_6d, kernel_size=smooth_window, sigma=1.0)
    return echo_module.rot6d_to_smpl_dict(smpl_6d)


def _resolve_manifest_link(args: argparse.Namespace) -> str:
    """Default for JSON `link` when a row has no stored link."""
    if getattr(args, "link", None) is not None:
        return str(args.link)
    return str(getattr(args, "default_link", "") or "")


def run(args: argparse.Namespace) -> None:
    if args.id_width < 1:
        raise ValueError("--id_width must be >= 1")

    wr = str(getattr(args, "weight_root", "") or "").strip()
    if wr:
        os.environ["VIDEO2SMPL_WEIGHT_ROOT"] = str(Path(wr).expanduser().resolve())
    else:
        os.environ["VIDEO2SMPL_WEIGHT_ROOT"] = ""

    manifest_source = str(args.source).strip()
    if not manifest_source:
        raise ValueError('--source is required and must be a non-empty string (dataset / provenance label for manifest JSON)')

    work_root = Path(args.root_dir).resolve()
    manifest_link = _resolve_manifest_link(args)
    rgb_dir = work_root / "rgb_videos"
    out_raw = work_root / "CameraHMR_smpl_results"
    out_smooth = work_root / "CameraHMR_smpl_results_smoothed"
    out_trainable = work_root / "processed_trainable_data"
    out_manifest = work_root / args.manifest_name

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {rgb_dir}")

    _setup_vendor_imports(Path(args.vendor_root))

    # Reuse EchoMotion pipeline pieces directly.
    from extract_motion import MotionExtractor  # type: ignore
    from scripts.data_processors.motion_alignment.retarget_mogen_db import (  # type: ignore
        smooth_motion_rep,
        smpl_dict_to_rot6d,
        rot6d_to_smpl_dict,
    )
    from scripts.data_processors.motion_alignment.seq_utils import linear_interpolate_frame_ids  # type: ignore

    class EchoModule:
        pass

    EchoModule.smooth_motion_rep = staticmethod(smooth_motion_rep)
    EchoModule.smpl_dict_to_rot6d = staticmethod(smpl_dict_to_rot6d)
    EchoModule.rot6d_to_smpl_dict = staticmethod(rot6d_to_smpl_dict)
    EchoModule.linear_interpolate_frame_ids = staticmethod(linear_interpolate_frame_ids)

    mapping_path = work_root / args.mapping_name
    id_mapping: List[Dict[str, str]] = _load_id_mapping(mapping_path)
    path_to_sample: Dict[str, str] = {}
    for item in id_mapping:
        rel = item.get("original_path_relative")
        sid = item.get("sample_id")
        if rel and sid:
            path_to_sample[rel] = sid

    max_id = 0
    for item in id_mapping:
        n = _parse_sample_id_numeric(item.get("sample_id", ""))
        if n is not None:
            max_id = max(max_id, n)
    max_id = max(
        max_id,
        _max_sample_id_from_dirs([out_raw, out_smooth, out_trainable], args.id_width),
    )
    next_id = max_id + 1

    prev_manifest = _load_manifest_preserve(out_manifest)

    extractor = MotionExtractor(
        device=torch.device(args.device) if args.device else None,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        use_shape=args.use_shape,
        overwrite=args.overwrite,
    )

    videos = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}])
    processed_this_run = 0
    skipped = 0

    for video_path in videos:
        rel = str(video_path.relative_to(work_root))
        if rel in path_to_sample and not args.overwrite:
            skipped += 1
            continue

        if rel in path_to_sample and args.overwrite:
            sample_id = path_to_sample[rel]
            id_mapping = [it for it in id_mapping if it.get("original_path_relative") != rel]
            path_to_sample.pop(rel, None)
        else:
            sample_id = f"{next_id:0{args.id_width}d}"
            next_id += 1

        sample_raw = out_raw / sample_id
        sample_smooth = out_smooth / sample_id
        sample_train = out_trainable / sample_id
        sample_raw.mkdir(parents=True, exist_ok=True)
        sample_smooth.mkdir(parents=True, exist_ok=True)
        sample_train.mkdir(parents=True, exist_ok=True)

        bbox_path = sample_raw / "bbox.pt"
        smpl_raw_path = sample_raw / "smpl_raw.pt"
        postprocess_path = sample_smooth / "motion_postprocess.pt"
        smpl_npz_path = sample_smooth / "smpls_smoothed_group.npz"
        smpl_canonical_npz_path = sample_smooth / "smpls_canonical_group.npz"

        bbx_xyxy, conf, frame_mask = extractor.extract_bbox(
            video_path=str(video_path),
            output_path=str(bbox_path),
            overwrite=args.overwrite,
        )
        smpl_data = extractor.extract_smpl(
            video_path=str(video_path),
            bbox_path=str(bbox_path),
            output_path=str(smpl_raw_path),
            overwrite=args.overwrite,
        )
        post_res = extractor.post_process(
            smpl_data=smpl_data,
            smooth_window=args.smooth_window,
            set_floor=args.set_floor,
            frame_mask=frame_mask,
            use_shape=args.use_shape,
        )
        torch.save(post_res, postprocess_path)

        person_idx = args.person_idx
        num_persons = int(smpl_data["smpl_params_incam"]["global_orient"].shape[0])
        smpl_incam = {k: v[person_idx].detach().cpu() for k, v in smpl_data["smpl_params_incam"].items()}
        smpl_smooth = _smooth_smpl_for_one_person(
            person_smpl_params=smpl_incam,
            frame_mask=frame_mask.detach().cpu(),
            smooth_window=args.smooth_window,
            echo_module=EchoModule,
        )

        canon_np = _canonical_smpl_to_npz_dict(
            post_res["smpl_params_canonical"], person_idx, num_persons
        )
        T_canon = int(canon_np["global_orient"].shape[0])
        canon_np["betas"] = _betas_np_from_smoothed(smpl_smooth, T_canon)
        np.savez(
            smpl_canonical_npz_path,
            **canon_np,
            intrinsic=smpl_data["intrinsic"].detach().cpu().numpy(),
            frame_mask=frame_mask.detach().cpu().numpy(),
            bbox_xyxy=bbx_xyxy.detach().cpu().numpy(),
            bbox_conf=conf.detach().cpu().numpy(),
            set_floor=np.array([int(args.set_floor)], dtype=np.int32),
            coord_note=np.bytes_("canonical_dart_smpl_axis_angle"),
        )

        betas_for_smooth_npz = smpl_smooth.get("betas")
        if betas_for_smooth_npz is not None:
            betas_smooth_out = betas_for_smooth_npz.detach().cpu().numpy().astype(np.float32)
        else:
            betas_smooth_out = np.zeros((10,), dtype=np.float32)
        np.savez(
            smpl_npz_path,
            global_orient=smpl_smooth["global_orient"].numpy(),
            body_pose=smpl_smooth["body_pose"].numpy(),
            transl=smpl_smooth["transl"].numpy(),
            betas=betas_smooth_out,
            intrinsic=smpl_data["intrinsic"].detach().cpu().numpy(),
            frame_mask=frame_mask.detach().cpu().numpy(),
            bbox_xyxy=bbx_xyxy.detach().cpu().numpy(),
            bbox_conf=conf.detach().cpu().numpy(),
            set_floor=np.array([int(args.set_floor)], dtype=np.int32),
        )

        rgb_out = sample_train / "rgb.mp4"
        first_frame = sample_train / "first_frame.jpg"
        _copy_rgb_video(video_path, rgb_out)
        _extract_first_frame(video_path, first_frame)

        map_row = {
            "sample_id": sample_id,
            "seq_index": "",
            "original_filename": video_path.name,
            "original_stem": video_path.stem,
            "original_path_relative": rel,
            "output_sample_dir": sample_id,
        }
        id_mapping.append(map_row)
        path_to_sample[rel] = sample_id
        processed_this_run += 1

    id_mapping.sort(key=lambda it: int(it["sample_id"]) if it.get("sample_id", "").isdigit() else 0)
    for i, item in enumerate(id_mapping, start=1):
        item["seq_index"] = str(i)

    manifest: List[Dict[str, str]] = []
    for item in id_mapping:
        sid = item["sample_id"]
        rgb_rel = f"processed_trainable_data/{sid}/rgb.mp4"
        ff_rel = f"processed_trainable_data/{sid}/first_frame.jpg"
        smpl_rel = f"CameraHMR_smpl_results_smoothed/{sid}/smpls_canonical_group.npz"
        smpl_incam_rel = f"CameraHMR_smpl_results_smoothed/{sid}/smpls_smoothed_group.npz"
        old = prev_manifest.get(sid, {})
        text_val = old.get("text", "")
        if not text_val:
            text_val = ""
        link_val = old.get("link")
        if link_val is None or str(link_val).strip() == "":
            link_val = manifest_link
        manifest.append(
            {
                "sample_id": sid,
                "original_video": item["original_filename"],
                "rgb_path": rgb_rel,
                "first_frame": ff_rel,
                "smpl_path": smpl_rel,
                "smpl_incam_smooth_path": smpl_incam_rel,
                "text": text_val,
                "type": "video",
                "source": manifest_source,
                "link": link_val,
            }
        )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

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

    print(f"Done. Total samples in mapping: {len(id_mapping)}")
    print(f"Processed this run: {processed_this_run}, skipped (already mapped): {skipped}")
    print(f'Manifest "source" label: {manifest_source}')
    print(f"Manifest written to: {out_manifest}")
    print(f"ID mapping written to: {mapping_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video2SMPL custom pipeline (steps 1-4 + empty-text manifest)")
    parser.add_argument("--root_dir", type=str, default="examples/training")
    parser.add_argument(
        "--weight_root",
        type=str,
        default="/data1/wjh/Video2SMPL",
        help="CameraHMR / SMPL / YOLO / Detectron 等权重所在目录（扁平放置）。传空字符串 \"\" 则仅用仓库内 third_party/.../data/",
    )
    parser.add_argument("--vendor_root", type=str, default="third_party")
    parser.add_argument("--manifest_name", type=str, default="train_stage4_empty_text.json")
    parser.add_argument(
        "--mapping_name",
        type=str,
        default="sample_id_to_source.json",
        help="JSON file under root_dir: seq id <-> original video filename/path",
    )
    parser.add_argument(
        "--id_width",
        type=int,
        default=6,
        help="Zero-pad width for sample folders (e.g. 6 -> 000001)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run videos that are already in the mapping (reuse the same sample_id). New videos always append after the current max id.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=500, help="Maximum number of frames to process per video")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--person_idx", type=int, default=0)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument(
        "--set-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Canonical DART 贴地（默认开启，适合站立/行走/地面操作等机器人常用动作）。使用 --no-set-floor 关闭。",
    )
    parser.add_argument("--use_shape", action="store_true")

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help='Required. Non-empty string written to every manifest row as field "source" (dataset or provenance label).',
    )
    parser.add_argument(
        "--link",
        type=str,
        default=None,
        help='Manifest field "link". Default: empty, or preserved from existing manifest.',
    )
    parser.add_argument(
        "--default_link",
        type=str,
        default="",
        help="Default link for new manifest rows when previous manifest has no link.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
