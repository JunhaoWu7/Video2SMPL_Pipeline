"""
Remove samples with ``robot_learnable == false`` after the captions stage.

Deletes ``processed_trainable_data/<sample_id>/`` and drops manifest / mapping rows.
Kept rows no longer store ``robot_learnable`` (all remaining clips are learnable).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from pipeline.dataset_schema import DIR_PROCESSED, sample_dir_rel
from pipeline.manifest import (
    DEFAULT_MANIFEST_NAME,
    STAGE_PRUNE,
    captions_filled,
    load_manifest_list,
    manifest_path,
    mark_stage_completed,
    parse_robot_learnable,
    save_manifest,
)


def _load_mapping(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("items") or [])


def _save_mapping(path: Path, work_root: Path, id_width: int, items: List[Dict[str, str]]) -> None:
    items = sorted(items, key=lambda it: int(it["sample_id"]) if str(it.get("sample_id", "")).isdigit() else 0)
    for i, item in enumerate(items, start=1):
        item["seq_index"] = str(i)
    path.write_text(
        json.dumps(
            {
                "root_dir": str(work_root),
                "id_width": id_width,
                "count": len(items),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _sample_dir_abs(work_root: Path, sample_id: str) -> Path:
    return work_root / sample_dir_rel(sample_id)


def prune_non_learnable(
    work_root: Path,
    rows: List[Dict[str, Any]],
    *,
    dry_run: bool,
) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Returns (kept_rows, removed_ids, warn_ids).

    ``warn_ids``: caption-incomplete rows left in manifest (not deleted).
    """
    kept: List[Dict[str, Any]] = []
    removed_ids: List[str] = []
    warn_ids: List[str] = []

    for row in rows:
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            continue
        learnable = parse_robot_learnable(row.get("robot_learnable"))
        if learnable is False:
            removed_ids.append(sid)
            if not dry_run:
                sample_dir = _sample_dir_abs(work_root, sid)
                if sample_dir.is_dir():
                    shutil.rmtree(sample_dir)
            continue

        if not captions_filled(row):
            warn_ids.append(sid)
            kept.append(row)
            continue

        cleaned = dict(row)
        cleaned.pop("robot_learnable", None)
        if not dry_run:
            mark_stage_completed(cleaned, STAGE_PRUNE)
        kept.append(cleaned)

    return kept, removed_ids, warn_ids


def run(args: argparse.Namespace) -> None:
    work_root = Path(args.root_dir).resolve()
    out_manifest = manifest_path(work_root, args.manifest_name)
    mapping_path = work_root / args.mapping_name

    if not out_manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {out_manifest}")

    rows = load_manifest_list(out_manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {out_manifest}")

    kept, removed_ids, warn_ids = prune_non_learnable(work_root, rows, dry_run=args.dry_run)

    print(
        f"Prune {'(dry-run) ' if args.dry_run else ''}done. "
        f"kept={len(kept)}, removed(non-learnable)={len(removed_ids)}, "
        f"caption_incomplete_kept={len(warn_ids)}"
    )
    if removed_ids:
        preview = ", ".join(removed_ids[:12])
        extra = f" (+{len(removed_ids) - 12} more)" if len(removed_ids) > 12 else ""
        print(f"  removed sample_id: {preview}{extra}")
    if warn_ids:
        print(
            f"  WARN: {len(warn_ids)} sample(s) lack full captions but were not removed "
            f"(e.g. {', '.join(warn_ids[:5])})"
        )

    if args.dry_run:
        return

    save_manifest(out_manifest, kept)

    kept_ids = {str(r.get("sample_id", "")) for r in kept}
    mapping_items = [it for it in _load_mapping(mapping_path) if str(it.get("sample_id", "")) in kept_ids]
    _save_mapping(mapping_path, work_root, args.id_width, mapping_items)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune robot_learnable=false samples after captions")
    p.add_argument("--root_dir", type=str, default="examples/training")
    p.add_argument("--manifest_name", type=str, default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--mapping_name", type=str, default="sample_id_to_source.json")
    p.add_argument("--id_width", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
