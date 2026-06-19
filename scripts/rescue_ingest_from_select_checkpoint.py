#!/usr/bin/env python3
"""
Rescue ingest: materialize already-computed select results (passed/rejected)
from a select_filter_checkpoint.json into:
  - processed_trainable_data/<sample_id>/...
  - dataset_manifest.json (updated in place)
  - sample_id_to_source.json (updated in place)

This script does NOT call VLM/YOLO again. It only performs the "passed -> ingest"
part so you don't lose quota-paid compute when an old pipeline run is interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline.dataset_schema import DIR_PROCESSED, sample_dir_rel, sample_video_rel
from pipeline.manifest import (
    DEFAULT_MANIFEST_NAME,
    apply_select_update,
    load_manifest_by_id,
    load_manifest_list,
    manifest_path,
    new_row_template,
    merge_preserved_fields,
    save_manifest,
)
from pipeline.sample_renumber import renumber_sample_ids
from pipeline.stages.select.checkpoint import load_checkpoint

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def _parse_int(x: Any) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0


def _load_mapping(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("items") or [])


def _save_mapping(path: Path, work_root: Path, id_width: int, items: List[Dict[str, str]]) -> None:
    items = sorted(items, key=lambda it: _parse_int(it.get("sample_id", "")))
    for i, item in enumerate(items, start=1):
        # Keep seq_index stable but do not force it to be non-empty; downstream prune will overwrite.
        item.setdefault("seq_index", "")
        item["seq_index"] = item.get("seq_index", "")  # no-op, explicit
    payload = {
        "root_dir": str(work_root),
        "id_width": id_width,
        "count": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_raw_video(input_dir: Path, key: str) -> Path | None:
    """
    key example: "video/<filename>.mp4" (most common).
    Falls back to searching by filename under input_dir recursively.
    """
    k = str(key).replace("\\", "/").strip()
    if k.startswith("video/"):
        rel = k[len("video/") :]
        cand = (input_dir / rel).resolve()
        if cand.is_file():
            return cand

    # Fallback by filename.
    fname = Path(k).name
    if not fname:
        return None
    # Exact match first (fast).
    cand2 = (input_dir / fname).resolve()
    if cand2.is_file():
        return cand2

    # Recursive scan (slow but only for rescue keys).
    hits = list(input_dir.rglob(fname))
    if hits:
        hits.sort(key=lambda p: p.stat().st_size if p.exists() else 0)
        return hits[0].resolve()

    # If extension mismatch, try any suffix by stem.
    stem = Path(fname).stem
    for suf in VIDEO_SUFFIXES:
        hits = list(input_dir.rglob(stem + suf))
        if hits:
            hits.sort(key=lambda p: p.stat().st_size if p.exists() else 0)
            return hits[0].resolve()
    return None


def _normalize_src_rel(key: str) -> str:
    # Manifest field `original_video_path` stores a relative path like "video/<file>.mp4".
    return str(key or "").replace("\\", "/").strip().lstrip("/")


def _ingest_one(
    *,
    work_root: Path,
    sample_id: str,
    src_video: Path,
    use_symlink: bool,
    overwrite: bool,
) -> Path:
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


def _compute_next_sample_id(existing_rows: List[Dict[str, Any]], id_mapping: List[Dict[str, str]], id_width: int) -> str:
    m = 0
    for r in existing_rows:
        m = max(m, _parse_int(r.get("sample_id")))
    for it in id_mapping:
        m = max(m, _parse_int(it.get("sample_id")))
    return f"{m + 1:0{id_width}d}"


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rescue ingest passed select results from checkpoint (no VLM/YOLO).")
    p.add_argument("--hub-root", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--default-link", type=str, default="")
    p.add_argument("--link", type=str, default=None, help="Override link for rescued rows.")
    p.add_argument("--checkpoint", type=str, default=None, help="select_filter_checkpoint.json path.")
    p.add_argument("--checkpoint-keys-limit", type=int, default=None, help="Optional cap for testing.")

    p.add_argument("--manifest-name", type=str, default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--mapping-name", type=str, default="sample_id_to_source.json")
    p.add_argument("--id-width", type=int, default=6)

    p.add_argument("--select-input-dir", type=str, default=None, help="Raw video folder (default: <dataset_root>/video).")
    p.add_argument("--select-symlink", action="store_true", help="Symlink instead of move into processed_trainable_data/<id>/.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite destination files/dirs when conflicts.")

    p.add_argument("--renumber", action="store_true", help="Optionally re-contiguous renumber sample_id after ingest.")
    args = p.parse_args(argv)

    work_root = (Path(args.hub_root) / args.dataset).resolve()
    input_dir = Path(args.select_input_dir).expanduser().resolve() if args.select_input_dir else (work_root / "video").resolve()
    out_manifest = manifest_path(work_root, args.manifest_name)
    mapping_path = work_root / args.mapping_name

    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (work_root / "logs" / "select_filter_checkpoint.json").resolve()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"select input dir not found: {input_dir}")

    manifest_rows = load_manifest_list(out_manifest) if out_manifest.exists() else []
    prev_by_id = load_manifest_by_id(out_manifest) if out_manifest.exists() else {}
    id_mapping = _load_mapping(mapping_path) if mapping_path.exists() else []

    checkpoint_items = load_checkpoint(checkpoint_path)  # key -> "passed"/"rejected"
    passed = [(k, v) for k, v in checkpoint_items.items() if v == "passed"]
    passed.sort(key=lambda kv: kv[0])
    if args.checkpoint_keys_limit:
        passed = passed[: args.checkpoint_keys_limit]

    link_val = (args.link if args.link is not None else args.default_link) or ""

    # Quick duplicate detection: original_video_path stored as src_rel (string).
    existing_originals = {
        _normalize_src_rel(r.get("original_video_path", ""))
        for r in manifest_rows
        if _normalize_src_rel(r.get("original_video_path", "")) != ""
    }

    next_sid = _compute_next_sample_id(manifest_rows, id_mapping, args.id_width)
    sid_int = _parse_int(next_sid)

    added = 0
    for key, _ in passed:
        # Canonicalize and dedupe.
        src_key = _normalize_src_rel(key)
        if src_key in existing_originals:
            continue

        src_video = _find_raw_video(input_dir, key)
        if src_video is None:
            print(f"[rescue] WARN: raw video not found for key={key!r} (skip)")
            continue

        sid = f"{sid_int:0{args.id_width}d}"
        sid_int += 1

        dest_video = _ingest_one(
            work_root=work_root,
            sample_id=sid,
            src_video=src_video,
            use_symlink=bool(args.select_symlink),
            overwrite=bool(args.overwrite),
        )
        video_rel = str(dest_video.relative_to(work_root))

        base = new_row_template(
            sample_id=sid,
            original_video=dest_video.name,
            source=str(args.source),
            link=str(link_val),
        )
        updated = apply_select_update(
            base,
            video_path=video_rel,
            select_status="passed",
            select_notes="",
            original_video_path=str(key).replace("\\", "/").lstrip("/"),
            mark_complete=False,
        )

        old_row = prev_by_id.get(sid)
        row = merge_preserved_fields(updated, old_row)
        manifest_rows.append(row)
        prev_by_id[sid] = row

        id_mapping.append(
            {
                "sample_id": sid,
                "seq_index": "",
                "original_filename": dest_video.name,
                "original_stem": Path(dest_video.name).stem,
                "original_path_relative": src_key,
                "output_sample_dir": sample_dir_rel(sid),
            }
        )
        existing_originals.add(src_key)
        added += 1

    if not added:
        print("[rescue] No new passed rows ingested (nothing to do).")
        return 0

    # Optional: keep sample_id contiguous for downstream convenience.
    if args.renumber:
        # renumber_sample_ids expects: ordered_rows + mapping_items
        manifest_rows_sorted = sorted(manifest_rows, key=lambda r: _parse_int(r.get("sample_id")))
        # Filter mapping to only ids present.
        kept_ids = {str(r.get("sample_id", "")).strip() for r in manifest_rows_sorted if r.get("sample_id")}
        id_mapping = [it for it in id_mapping if str(it.get("sample_id", "")).strip() in kept_ids]
        manifest_rows_sorted, id_mapping, _id_remap = renumber_sample_ids(
            work_root,
            manifest_rows_sorted,
            id_mapping,
            id_width=args.id_width,
        )
        manifest_rows = manifest_rows_sorted

    # Deterministic ordering.
    manifest_rows = sorted(manifest_rows, key=lambda r: _parse_int(r.get("sample_id")))

    save_manifest(out_manifest, manifest_rows)
    _save_mapping(mapping_path, work_root, args.id_width, id_mapping)

    print(
        f"[rescue] Done. checkpoint_keys={len(checkpoint_items)} passed_ingested={added} "
        f"manifest_rows={len(manifest_rows)}"
    )
    print(f"[rescue] Manifest: {out_manifest}")
    print(f"[rescue] Mapping: {mapping_path}")
    print(f"[rescue] Processed dir: {work_root / DIR_PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

