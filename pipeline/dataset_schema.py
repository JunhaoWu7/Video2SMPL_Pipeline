"""
Canonical unified dataset layout for all post-processed batches.

Each sample lives under ``processed_trainable_data/<sample_id>/``:
  <original_name>.mp4 (or mov/...), first_frame.jpg, smpl_canonical.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pipeline.manifest import (
    STAGE_CAPTIONS,
    STAGE_SELECT,
    STAGE_VIDEO2SMPL,
    captions_filled,
    get_action_caption,
    get_caption,
    normalize_row,
    smpl_filled,
)

MANIFEST_DEFAULT = "dataset_manifest.json"
MAPPING_DEFAULT = "sample_id_to_source.json"
SPLITS_DIR = "splits"

DIR_PROCESSED = "processed_trainable_data"
SPLITS_DIR_NAME = SPLITS_DIR

SMPL_CANONICAL_FILENAME = "smpl_canonical.npz"
FIRST_FRAME_FILENAME = "first_frame.jpg"

STANDARD_DATASET_DIRS: tuple[str, ...] = (
    DIR_PROCESSED,
    SPLITS_DIR,
)

CANONICAL_ROW_KEYS: tuple[str, ...] = (
    "sample_id",
    "original_video",
    "video_path",
    "rgb_path",
    "first_frame",
    "smpl_path",
    "caption",
    "action_caption",
    "robot_learnable",
    "skill_category",
    "type",
    "source",
    "link",
    "select_status",
    "select_notes",
    "original_video_path",
    "stages_completed",
)

TRAIN_EXPORT_KEYS: tuple[str, ...] = (
    "sample_id",
    "rgb_path",
    "first_frame",
    "smpl_path",
    "caption",
    "action_caption",
    "robot_learnable",
    "skill_category",
    "video_path",
    "type",
    "source",
    "link",
)


def sample_dir_rel(sample_id: str) -> str:
    return f"{DIR_PROCESSED}/{sample_id}"


def sample_video_rel(sample_id: str, video_filename: str) -> str:
    name = Path(video_filename).name
    if not name:
        raise ValueError("video_filename must be non-empty")
    return f"{sample_dir_rel(sample_id)}/{name}"


def sample_paths(sample_id: str, video_filename: str) -> Dict[str, str]:
    """Standard relative paths for one sample (video co-located with SMPL / first_frame)."""
    base = sample_dir_rel(sample_id)
    v = sample_video_rel(sample_id, video_filename)
    return {
        "video_path": v,
        "rgb_path": v,
        "first_frame": f"{base}/{FIRST_FRAME_FILENAME}",
        "smpl_path": f"{base}/{SMPL_CANONICAL_FILENAME}",
    }


def is_train_ready(
    row: Mapping[str, Any],
    *,
    require_captions: bool = False,
    require_stages: Optional[Sequence[str]] = None,
) -> bool:
    r = normalize_row(dict(row))
    if not r.get("sample_id"):
        return False
    if not r.get("rgb_path") or not r.get("smpl_path"):
        return False
    if require_captions and not captions_filled(r):
        return False
    if require_stages:
        done = set(r.get("stages_completed") or [])
        if not all(s in done for s in require_stages):
            return False
    return True


def export_train_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    r = normalize_row(dict(row))
    out: Dict[str, Any] = {k: r.get(k, "") for k in TRAIN_EXPORT_KEYS}
    out["sample_id"] = str(r["sample_id"])
    if not out.get("caption"):
        out["caption"] = get_caption(r)
    if not out.get("action_caption"):
        out["action_caption"] = get_action_caption(r)
    return out


def filter_train_ready(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_captions: bool = False,
    require_stages: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    allowed = {s.strip() for s in sources} if sources else None
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = normalize_row(dict(row))
        if allowed is not None and str(r.get("source", "")).strip() not in allowed:
            continue
        if not is_train_ready(
            r,
            require_captions=require_captions,
            require_stages=require_stages,
        ):
            continue
        out.append(export_train_row(r))
    return out


def layout_tree_text(root_name: str = "<root_dir>") -> str:
    return f"""{root_name}/
├── {MANIFEST_DEFAULT}
├── {MAPPING_DEFAULT}
├── {SPLITS_DIR}/
│   ├── manipulation.json         # export_skill_splits.py (after SMPL)
│   ├── locomotion.json
│   └── loco-manipulation.json
└── {DIR_PROCESSED}/<sample_id>/
    ├── <video>.mp4               # one video per sample (select moves here)
    ├── {FIRST_FRAME_FILENAME}
    └── {SMPL_CANONICAL_FILENAME}
"""
