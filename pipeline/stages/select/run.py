"""
Select stage (passthrough for now).

Ingests videos from ``--select-input-dir``, assigns ``sample_id``, moves each clip into
``processed_trainable_data/<sample_id>/``, and registers paths in ``dataset_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.dataset_schema import DIR_PROCESSED, sample_dir_rel, sample_paths, sample_video_rel
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

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


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
    if not args.select_input_dir:
        raise ValueError(
            "--select-input-dir is required: folder of raw videos to ingest into "
            f"{DIR_PROCESSED}/<sample_id>/."
        )
    input_dir = Path(args.select_input_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = (work_root / input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Select input directory not found: {input_dir}")

    use_symlink = bool(getattr(args, "select_symlink", False))

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

    videos = _collect_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir} (recursive {VIDEO_SUFFIXES})")

    for video_path in videos:
        try:
            src_rel = str(video_path.relative_to(work_root))
        except ValueError:
            src_rel = str(video_path.resolve())

        if src_rel in path_to_sample and not args.overwrite:
            sid = path_to_sample[src_rel]
            skipped += 1
        elif src_rel in path_to_sample and args.overwrite:
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

        if src_rel not in path_to_sample:
            map_row = {
                "sample_id": sid,
                "seq_index": "",
                "original_filename": video_path.name,
                "original_stem": video_path.stem,
                "original_path_relative": src_rel,
                "output_sample_dir": sample_dir_rel(sid),
            }
            id_mapping.append(map_row)
            path_to_sample[src_rel] = sid

        old = manifest_rows.get(sid, {})
        link_val = old.get("link")
        if link_val is None or str(link_val).strip() == "":
            link_val = link_default

        paths = sample_paths(sid, dest_video.name)
        base = new_row_template(
            sample_id=sid,
            original_video=dest_video.name,
            source=source,
            link=str(link_val),
        )
        base["rgb_path"] = paths["video_path"]
        base["first_frame"] = paths["first_frame"]
        base["smpl_path"] = paths["smpl_path"]
        updated = apply_select_update(
            base,
            video_path=paths["video_path"],
            select_status="passed",
            select_notes="Ingested into processed_trainable_data/<sample_id>/; full select TBD.",
            original_video_path=src_rel,
        )
        updated["rgb_path"] = paths["video_path"]
        manifest_rows[sid] = merge_preserved_fields(updated, old)

    id_mapping.sort(key=lambda it: int(it["sample_id"]) if str(it.get("sample_id", "")).isdigit() else 0)
    for i, item in enumerate(id_mapping, start=1):
        item["seq_index"] = str(i)

    ordered = [manifest_rows[sid] for sid in sorted(manifest_rows.keys()) if sid.isdigit()]
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
    print(f"Select done. Videos ingested ({mode}): {len(videos)}")
    print(f"  New this run: {added}, skipped (already mapped): {skipped}")
    print(f"  Manifest: {out_manifest}")
    print(f"  Mapping: {mapping_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select stage: ingest videos into per-sample dirs")
    parser.add_argument("--root_dir", type=str, default="examples/training")
    parser.add_argument("--manifest_name", type=str, default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--mapping_name", type=str, default="sample_id_to_source.json")
    parser.add_argument("--id_width", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--select-input-dir",
        type=str,
        required=True,
        help="Folder of raw videos (searched recursively); each file -> processed_trainable_data/<id>/.",
    )
    parser.add_argument(
        "--select-symlink",
        action="store_true",
        help="Symlink instead of move when placing video under processed_trainable_data/.",
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--link", type=str, default=None)
    parser.add_argument("--default_link", type=str, default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
