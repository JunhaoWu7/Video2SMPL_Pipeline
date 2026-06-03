#!/usr/bin/env python3
"""
Export per-skill manifest lists after video2smpl.

Called by pipeline stage ``export_splits`` (``run.py``); keep logic here, not in video2smpl.

Writes under ``<root-dir>/splits/``:

  manipulation.json
  locomotion.json
  loco-manipulation.json

Each file is a JSON list of train-ready rows (SMPL + captions + skill_category).

Examples:
  python export_skill_splits.py --root-dir /data1/wjh/HumanRetarget/humanvid
  python export_skill_splits.py --root-dir examples/training --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.dataset_schema import SPLITS_DIR, export_train_row  # noqa: E402
from pipeline.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_NAME,
    SKILL_LOCO_MANIPULATION,
    SKILL_LOCOMOTION,
    SKILL_MANIPULATION,
    STAGE_VIDEO2SMPL,
    VALID_SKILL_CATEGORIES,
    captions_filled,
    load_manifest_list,
    manifest_path,
    normalize_skill_category,
    smpl_filled,
)

SKILL_SPLIT_FILES: dict[str, str] = {
    SKILL_MANIPULATION: "manipulation.json",
    SKILL_LOCOMOTION: "locomotion.json",
    SKILL_LOCO_MANIPULATION: "loco-manipulation.json",
}


def rows_ready_for_skill_export(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not captions_filled(row):
            continue
        if not smpl_filled(row):
            continue
        cat = normalize_skill_category(row.get("skill_category"))
        if cat not in VALID_SKILL_CATEGORIES:
            continue
        exported = export_train_row(row)
        exported.pop("robot_learnable", None)
        out.append(exported)
    return out


def group_by_skill(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {k: [] for k in SKILL_SPLIT_FILES}
    for row in rows:
        cat = normalize_skill_category(row.get("skill_category"))
        if cat in groups:
            groups[cat].append(row)
    for cat in groups:
        groups[cat].sort(key=lambda r: int(r["sample_id"]) if str(r.get("sample_id", "")).isdigit() else 0)
    return groups


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export splits/{manipulation,locomotion,loco-manipulation}.json after SMPL"
    )
    p.add_argument("--root-dir", type=Path, required=True, help="Dataset root (pipeline --root_dir)")
    p.add_argument("--manifest-name", type=str, default=DEFAULT_MANIFEST_NAME)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Default: <root-dir>/{SPLITS_DIR}",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write JSON files.",
    )
    args = p.parse_args(argv)

    root = args.root_dir.resolve()
    mpath = manifest_path(root, args.manifest_name)
    if not mpath.is_file():
        print(f"Manifest not found: {mpath}", file=sys.stderr)
        return 1

    all_rows = load_manifest_list(mpath)
    ready = rows_ready_for_skill_export(all_rows)
    groups = group_by_skill(ready)

    summary = {
        "root_dir": str(root),
        "manifest": str(mpath),
        "manifest_total": len(all_rows),
        "export_ready": len(ready),
        "by_skill": {cat: len(groups[cat]) for cat in SKILL_SPLIT_FILES},
        "output_dir": str(args.output_dir or (root / SPLITS_DIR)),
        "requires": ["captions_filled", "smpl_path", "skill_category"],
        "note": "Run after video2smpl; non-learnable samples should already be pruned.",
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    out_dir = args.output_dir or (root / SPLITS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat, filename in SKILL_SPLIT_FILES.items():
        path = out_dir / filename
        path.write_text(
            json.dumps(groups[cat], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary["files"] = {cat: str(out_dir / fn) for cat, fn in SKILL_SPLIT_FILES.items()}
    (out_dir / "skill_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
