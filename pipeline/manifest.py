"""
Unified pipeline manifest: one JSON list under ``root_dir`` that stages update in place.

Intended stage flow (see project diagram):
  1. select       -> video_path, rgb_path (per-sample dir), select_status, ...
  2. captions     -> caption, action_caption
  3. video2smpl   -> first_frame, smpl_path (canonical npz only)

Legacy manifests (``train_stage4_empty_text.json``, ``text`` field) are normalized on load.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

# Default single manifest filename (all stages read/write this unless overridden).
DEFAULT_MANIFEST_NAME = "dataset_manifest.json"

LEGACY_MANIFEST_NAMES = (
    "train_stage4_empty_text.json",
    "train_stage5_with_text.json",
)

STAGE_SELECT = "select"
STAGE_CAPTIONS = "captions"
STAGE_VIDEO2SMPL = "video2smpl"
STAGE_EXTERNAL_SMPL = "external_smpl"

# Fields owned by each stage (for merge / documentation).
STAGE_FIELDS: Dict[str, tuple[str, ...]] = {
    STAGE_SELECT: (
        "video_path",
        "select_status",
        "select_notes",
        "original_video_path",
    ),
    STAGE_CAPTIONS: (
        "caption",
        "action_caption",
    ),
    STAGE_VIDEO2SMPL: (
        "rgb_path",
        "first_frame",
        "smpl_path",
    ),
}

# Preserved across any stage rebuild when sample_id matches.
PRESERVE_KEYS = frozenset(
    {
        "caption",
        "action_caption",
        "text",  # legacy alias of caption
        "video_path",
        "select_status",
        "select_notes",
        "original_video_path",
        "link",
        "stages_completed",
        "action_label",  # reserved alias
    }
)


def manifest_path(root_dir: Path, manifest_name: Optional[str] = None) -> Path:
    name = manifest_name or DEFAULT_MANIFEST_NAME
    return Path(root_dir).resolve() / name


def load_manifest_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    return [normalize_row(x) for x in raw if isinstance(x, dict)]


def load_manifest_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(r["sample_id"]): r for r in load_manifest_list(path) if r.get("sample_id")}


def save_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [normalize_row(dict(r)) for r in rows]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _nonempty_str(val: Any) -> str:
    return str(val or "").strip()


def get_caption(row: Mapping[str, Any]) -> str:
    """Primary scene description (1–2 sentences)."""
    c = _nonempty_str(row.get("caption"))
    if c:
        return c
    return _nonempty_str(row.get("text"))


def get_action_caption(row: Mapping[str, Any]) -> str:
    """Short action phrase for the clip."""
    a = _nonempty_str(row.get("action_caption"))
    if a:
        return a
    return _nonempty_str(row.get("action_label"))


def resolve_video_rel(row: Mapping[str, Any]) -> str:
    """Relative clip path for caption stage (``video_path`` from select only)."""
    return _nonempty_str(row.get("video_path"))


def resolve_video_abs(root_dir: Path, row: Mapping[str, Any]) -> Path:
    rel = resolve_video_rel(row)
    if not rel:
        raise ValueError(
            f"sample_id={row.get('sample_id')!r} has no video_path; run the select stage first."
        )
    rel = rel.replace("\\", "/").lstrip("/")
    return (Path(root_dir).resolve() / rel).resolve()


def captions_filled(row: Mapping[str, Any]) -> bool:
    return bool(get_caption(row)) and bool(get_action_caption(row))


def smpl_filled(row: Mapping[str, Any]) -> bool:
    return bool(_nonempty_str(row.get("smpl_path")))


def normalize_row(row: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure unified schema; migrate legacy ``text`` -> ``caption``."""
    out: Dict[str, Any] = dict(row)
    sid = _nonempty_str(out.get("sample_id"))
    if sid:
        out["sample_id"] = sid

    cap = get_caption(out)
    if cap and not _nonempty_str(out.get("caption")):
        out["caption"] = cap
    act = get_action_caption(out)
    if act and not _nonempty_str(out.get("action_caption")):
        out["action_caption"] = act

    out.setdefault("caption", "")
    out.setdefault("action_caption", "")
    out.setdefault("video_path", "")
    out.setdefault("select_status", "")
    out.setdefault("select_notes", "")
    out.setdefault("original_video_path", "")
    out.setdefault("rgb_path", "")
    out.setdefault("first_frame", "")
    out.setdefault("smpl_path", "")
    out.setdefault("type", "video")
    # Drop deprecated debug / incam fields from exported rows.
    out.pop("smpl_incam_smooth_path", None)
    out.setdefault("source", "")
    out.setdefault("link", "")
    out.setdefault("stages_completed", [])

    sc = out.get("stages_completed")
    if not isinstance(sc, list):
        out["stages_completed"] = []
    else:
        out["stages_completed"] = [str(x) for x in sc]

    # Keep legacy ``text`` in sync when caption is set (downstream readers).
    if _nonempty_str(out.get("caption")):
        out["text"] = out["caption"]
    else:
        out.setdefault("text", "")

    return out


def merge_preserved_fields(new_row: Dict[str, Any], old_row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not old_row:
        return normalize_row(new_row)
    out = normalize_row(new_row)
    old = normalize_row(dict(old_row))
    for key in PRESERVE_KEYS:
        if key == "stages_completed":
            merged = list(old.get("stages_completed") or [])
            for s in out.get("stages_completed") or []:
                if s not in merged:
                    merged.append(s)
            out["stages_completed"] = merged
            continue
        old_val = old.get(key)
        if key in ("caption", "action_caption", "text"):
            if _nonempty_str(old_val) and not _nonempty_str(out.get(key)):
                out[key] = old_val
            continue
        if old_val is not None and old_val != "" and (out.get(key) in (None, "", [])):
            out[key] = old_val
    if _nonempty_str(out.get("caption")):
        out["text"] = out["caption"]
    return out


def mark_stage_completed(row: Dict[str, Any], stage: str) -> None:
    normalized = normalize_row(row)
    row.clear()
    row.update(normalized)
    stages = list(row.get("stages_completed") or [])
    if stage not in stages:
        stages.append(stage)
    row["stages_completed"] = stages


def new_row_template(
    *,
    sample_id: str,
    original_video: str,
    source: str,
    link: str = "",
) -> Dict[str, Any]:
    return normalize_row(
        {
            "sample_id": sample_id,
            "original_video": original_video,
            "video_path": "",
            "rgb_path": "",
            "first_frame": "",
            "smpl_path": "",
            "caption": "",
            "action_caption": "",
            "text": "",
            "type": "video",
            "source": source,
            "link": link,
            "select_status": "",
            "select_notes": "",
            "original_video_path": "",
            "stages_completed": [],
        }
    )


def build_video2smpl_row(
    *,
    sample_id: str,
    original_video: str,
    video_rel: str,
    first_frame_rel: str,
    smpl_rel: str,
    source: str,
    link: str,
    old_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """``video_rel`` / ``rgb_path`` point at the per-sample video under processed_trainable_data/."""
    base = new_row_template(
        sample_id=sample_id,
        original_video=original_video,
        source=source,
        link=link,
    )
    v = video_rel.strip().replace("\\", "/").lstrip("/")
    base["video_path"] = v
    base["rgb_path"] = v
    base["first_frame"] = first_frame_rel
    base["smpl_path"] = smpl_rel
    mark_stage_completed(base, STAGE_VIDEO2SMPL)
    return merge_preserved_fields(base, old_row)


def apply_select_update(
    row: Dict[str, Any],
    *,
    video_path: str,
    select_status: str = "passed",
    select_notes: str = "",
    original_video_path: str = "",
) -> Dict[str, Any]:
    """
    Reserved for the future ``select`` stage.

    Sets the clip path used by later caption / SMPL stages.
    """
    out = normalize_row(row)
    out["video_path"] = video_path
    out["rgb_path"] = video_path
    out["select_status"] = select_status
    out["select_notes"] = select_notes
    if original_video_path:
        out["original_video_path"] = original_video_path
    mark_stage_completed(out, STAGE_SELECT)
    return out


def apply_captions_update(
    row: Dict[str, Any],
    *,
    caption: str,
    action_caption: str,
) -> Dict[str, Any]:
    out = normalize_row(row)
    out["caption"] = caption.strip()
    out["action_caption"] = action_caption.strip()
    out["text"] = out["caption"]
    mark_stage_completed(out, STAGE_CAPTIONS)
    return out


def try_load_legacy_manifest(root_dir: Path, manifest_name: str) -> List[Dict[str, Any]]:
    """If the primary manifest is missing, try legacy filenames."""
    primary = manifest_path(root_dir, manifest_name)
    if primary.exists():
        return load_manifest_list(primary)
    for legacy in LEGACY_MANIFEST_NAMES:
        p = root_dir / legacy
        if p.exists():
            return load_manifest_list(p)
    return []
