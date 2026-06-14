#!/usr/bin/env python3
"""Check pilot / dataset readiness after each pipeline stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.dataset_schema import DIR_PROCESSED, DIR_RAW_VIDEO, MANIFEST_DEFAULT, SPLITS_DIR
from pipeline.hub import DEFAULT_HUB_ROOT, resolve_dataset_root
from pipeline.manifest import (
    captions_filled,
    load_manifest_list,
    manifest_path,
    smpl_filled,
    VALID_SKILL_CATEGORIES,
)


def _status_icon(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def verify_dataset(root: Path) -> dict:
    mpath = manifest_path(root)
    rows = load_manifest_list(mpath) if mpath.is_file() else []
    n = len(rows)

    select_done = sum(1 for r in rows if "select" in (r.get("stages_completed") or []))
    captions_done = sum(1 for r in rows if "captions" in (r.get("stages_completed") or []))
    prune_done = sum(1 for r in rows if "prune" in (r.get("stages_completed") or []))
    smpl_done = sum(1 for r in rows if "video2smpl" in (r.get("stages_completed") or []))
    caption_complete = sum(1 for r in rows if captions_filled(r))
    smpl_path_ok = sum(1 for r in rows if smpl_filled(r))

    video_in = len(list((root / DIR_RAW_VIDEO).glob("*.mp4"))) if (root / DIR_RAW_VIDEO).is_dir() else 0
    proc_dirs = len(list((root / DIR_PROCESSED).iterdir())) if (root / DIR_PROCESSED).is_dir() else 0

    split_files = {
        "manipulation.json": (root / SPLITS_DIR / "manipulation.json").is_file(),
        "locomotion.json": (root / SPLITS_DIR / "locomotion.json").is_file(),
        "loco-manipulation.json": (root / SPLITS_DIR / "loco-manipulation.json").is_file(),
        "skill_export_summary.json": (root / SPLITS_DIR / "skill_export_summary.json").is_file(),
    }

    pending = []
    for r in rows:
        sid = r.get("sample_id", "?")
        issues = []
        if "select" in (r.get("stages_completed") or []):
            vp = str(r.get("video_path", ""))
            if not vp.startswith(f"{DIR_PROCESSED}/"):
                issues.append("video_path not in processed_trainable_data")
        if captions_done and not captions_filled(r) and "prune" not in (r.get("stages_completed") or []):
            issues.append("captions incomplete")
        if smpl_done and not smpl_filled(r):
            issues.append("smpl_path empty")
        sc = str(r.get("skill_category", "")).strip()
        if "captions" in (r.get("stages_completed") or []) and sc not in VALID_SKILL_CATEGORIES:
            issues.append(f"invalid skill_category={sc!r}")
        if issues:
            pending.append({"sample_id": sid, "issues": issues})

    return {
        "root_dir": str(root),
        "manifest_rows": n,
        "video_dir_mp4": video_in,
        "processed_dirs": proc_dirs,
        "stages": {
            "select": select_done,
            "captions": captions_done,
            "prune": prune_done,
            "video2smpl": smpl_done,
        },
        "caption_complete_rows": caption_complete,
        "smpl_filled_rows": smpl_path_ok,
        "splits": split_files,
        "pending_issues": pending[:20],
        "e2e_pass": (
            n > 0
            and select_done == n
            and captions_done == n
            and prune_done == n
            and smpl_done == n
            and smpl_path_ok == n
            and all(split_files.values())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dataset / pilot pipeline status")
    parser.add_argument("--hub-root", type=str, default=DEFAULT_HUB_ROOT)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    root = resolve_dataset_root(args.hub_root, args.dataset)
    report = verify_dataset(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"\nE2E: {_status_icon(report['e2e_pass'])} "
        f"({report['stages']['video2smpl']}/{report['manifest_rows']} smpl, "
        f"splits={sum(report['splits'].values())}/4 files)"
    )
    return 0 if report["manifest_rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
