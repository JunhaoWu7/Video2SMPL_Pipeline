#!/usr/bin/env python3
"""
Create a small pilot dataset under HumanRetarget for end-to-end pipeline smoke tests.

Example:
  python scripts/create_pilot_dataset.py \
    --source-dataset charades \
    --pilot-name charades_pilot \
    --count 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.dataset_schema import DIR_PROCESSED, DIR_RAW_VIDEO, MANIFEST_DEFAULT, MAPPING_DEFAULT
from pipeline.hub import DEFAULT_HUB_ROOT, init_dataset_layout
from pipeline.manifest import normalize_row, save_manifest

PILOT_RESET_KEYS = {
    "video_path": "",
    "rgb_path": "",
    "first_frame": "",
    "smpl_path": "",
    "smpl_backend": "",
    "select_status": "",
    "select_notes": "",
    "original_video_path": "",
    "robot_learnable": None,
    "skill_category": "",
    "stages_completed": [],
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    import random

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[i] for i in indices]


def _build_pilot_row(src: dict[str, Any], new_sid: str, video_filename: str) -> dict[str, Any]:
    row = normalize_row(dict(src))
    row["sample_id"] = new_sid
    row["original_video"] = video_filename
    for key, val in PILOT_RESET_KEYS.items():
        row[key] = val if key != "stages_completed" else []
    # Keep Charades-provided text if present; captions stage will fill missing fields.
    return normalize_row(row)


def create_pilot(
    *,
    hub_root: Path,
    source_dataset: str,
    pilot_name: str,
    count: int,
    seed: int,
    overwrite: bool,
) -> Path:
    src_root = hub_root / source_dataset
    pilot_root = hub_root / pilot_name
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")
    manifest_path = src_root / MANIFEST_DEFAULT
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    src_rows = [_ for _ in _load_json(manifest_path) if isinstance(_, dict)]
    if not src_rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    picked = _pick_rows(src_rows, count, seed)
    if pilot_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Pilot dataset already exists: {pilot_root}. Pass --overwrite to recreate."
            )
        shutil.rmtree(pilot_root)

    init_dataset_layout(pilot_root, dataset_name=pilot_name)
    video_out = pilot_root / DIR_RAW_VIDEO
    proc_out = pilot_root / DIR_PROCESSED
    if proc_out.exists():
        shutil.rmtree(proc_out)
    proc_out.mkdir(parents=True, exist_ok=True)

    pilot_rows: list[dict[str, Any]] = []
    mapping_items: list[dict[str, str]] = []
    missing_videos: list[str] = []

    for i, src in enumerate(picked, start=1):
        sid = f"{i:06d}"
        rel = str(src.get("video_path", "")).strip().replace("\\", "/").lstrip("/")
        if not rel:
            raise ValueError(f"sample_id={src.get('sample_id')} has empty video_path")
        src_video = src_root / rel
        if not src_video.is_file():
            missing_videos.append(rel)
            continue

        dst_name = src_video.name
        dst_video = video_out / dst_name
        shutil.copy2(src_video, dst_video)

        row = _build_pilot_row(src, sid, dst_name)
        row["video_path"] = f"{DIR_RAW_VIDEO}/{dst_name}"
        row["source"] = pilot_name
        pilot_rows.append(row)

        mapping_items.append(
            {
                "sample_id": sid,
                "seq_index": str(i),
                "original_filename": dst_name,
                "original_stem": src_video.stem,
                "original_path_relative": f"{DIR_RAW_VIDEO}/{dst_name}",
                "output_sample_dir": f"{DIR_PROCESSED}/{sid}",
            }
        )

    if missing_videos:
        raise FileNotFoundError(
            "Missing source videos:\n" + "\n".join(missing_videos[:20])
        )
    if not pilot_rows:
        raise ValueError("No pilot rows created.")

    save_manifest(pilot_root / MANIFEST_DEFAULT, pilot_rows)
    (pilot_root / MAPPING_DEFAULT).write_text(
        json.dumps(
            {
                "root_dir": str(pilot_root),
                "id_width": 6,
                "count": len(mapping_items),
                "items": mapping_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = {
        "dataset": pilot_name,
        "root_dir": str(pilot_root),
        "layout_version": 2,
        "manifest": MANIFEST_DEFAULT,
        "mapping": MAPPING_DEFAULT,
        "id_width": 6,
        "raw_video_dir": f"{DIR_RAW_VIDEO}/",
        "train_sample_dir": f"{DIR_PROCESSED}/<sample_id>/",
        "pilot_of": source_dataset,
        "pilot_count": len(pilot_rows),
        "pilot_seed": seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Fresh pilot for end-to-end pipeline validation.",
    }
    (pilot_root / "hub.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = pilot_root / "PILOT_README.txt"
    readme.write_text(
        "\n".join(
            [
                f"Pilot dataset: {pilot_name}",
                f"Source: {source_dataset}",
                f"Samples: {len(pilot_rows)}",
                "",
                "Run full chain:",
                f"  cd /home/wujunhao/code/Video2SMPL_Pipeline",
                f"  conda activate video2smpl",
                f"  export TOKENROUTER_API_KEY=...   # or OPENAI_API_KEY",
                f"  python run.py --dataset {pilot_name} --stages select --overwrite",
                f"  python run.py --dataset {pilot_name} --from-stage captions",
                "",
                "Verify:",
                f"  python scripts/verify_pilot_dataset.py --dataset {pilot_name}",
            ]
        ),
        encoding="utf-8",
    )
    return pilot_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a small pilot dataset for pipeline E2E test")
    parser.add_argument("--hub-root", type=str, default=DEFAULT_HUB_ROOT)
    parser.add_argument("--source-dataset", type=str, default="charades")
    parser.add_argument("--pilot-name", type=str, default="charades_pilot")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pilot_root = create_pilot(
        hub_root=Path(args.hub_root).expanduser().resolve(),
        source_dataset=args.source_dataset.strip(),
        pilot_name=args.pilot_name.strip(),
        count=max(1, args.count),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Created pilot dataset: {pilot_root}")
    print(f"Videos: {len(list((pilot_root / DIR_RAW_VIDEO).glob('*.mp4')))}")
    print(f"Manifest: {pilot_root / MANIFEST_DEFAULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
