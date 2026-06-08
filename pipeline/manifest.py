"""
Unified pipeline manifest: one JSON list under ``root_dir`` that stages update in place.

Intended stage flow (see project diagram):
  1. select       -> video_path, rgb_path (per-sample dir), select_status, ...
  2. captions     -> caption, action_caption, robot_learnable, skill_category
  3. prune        -> drop robot_learnable=false samples (+ on-disk dirs)
  4. video2smpl   -> first_frame, smpl_path (canonical npz only)
  5. export_splits -> splits/{manipulation,locomotion,loco-manipulation}.json (separate script)

Legacy manifests (``train_stage4_empty_text.json``) and old ``text`` fields are migrated to ``caption`` on load.
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
STAGE_PRUNE = "prune"
STAGE_VIDEO2SMPL = "video2smpl"
STAGE_EXPORT_SPLITS = "export_splits"
STAGE_EXTERNAL_SMPL = "external_smpl"

# Robot skill taxonomy (captions / VLM stage); exactly one category per clip.
SKILL_MANIPULATION = "manipulation"
SKILL_LOCOMOTION = "locomotion"
SKILL_LOCO_MANIPULATION = "loco-manipulation"

VALID_SKILL_CATEGORIES: frozenset[str] = frozenset(
    {SKILL_MANIPULATION, SKILL_LOCOMOTION, SKILL_LOCO_MANIPULATION}
)

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
        "robot_learnable",
        "skill_category",
    ),
    STAGE_PRUNE: (),
    STAGE_VIDEO2SMPL: (
        "rgb_path",
        "first_frame",
        "smpl_path",
        "smpl_backend",
    ),
}

# Preserved across any stage rebuild when sample_id matches.
PRESERVE_KEYS = frozenset(
    {
        "caption",
        "action_caption",
        "robot_learnable",
        "skill_category",
        "video_path",
        "select_status",
        "select_notes",
        "original_video_path",
        "link",
        "stages_completed",
        "action_label",  # reserved alias
        "smpl_backend",
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
    return _nonempty_str(row.get("caption"))


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


def normalize_skill_category(val: Any) -> str:
    """Map model output to one of ``VALID_SKILL_CATEGORIES`` or ``\"\"``."""
    s = str(val or "").strip().lower()
    if not s:
        return ""
    s = s.replace("_", "-").replace(" ", "-")
    if s in ("loco-manipulation", "locomanipulation", "loco-manip", "locomanip"):
        return SKILL_LOCO_MANIPULATION
    if s in ("manipulation", "manip"):
        return SKILL_MANIPULATION
    if s in ("locomotion", "loco", "locomote"):
        return SKILL_LOCOMOTION
    if s in VALID_SKILL_CATEGORIES:
        return s
    return ""


def parse_robot_learnable(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1", "learnable", "robot_learnable"):
        return True
    if s in ("false", "no", "n", "0", "not_learnable", "not-learnable"):
        return False
    return None


def captions_filled(row: Mapping[str, Any]) -> bool:
    if not get_caption(row) or not get_action_caption(row):
        return False
    if normalize_skill_category(row.get("skill_category")) not in VALID_SKILL_CATEGORIES:
        return False
    learnable = parse_robot_learnable(row.get("robot_learnable"))
    if learnable is False:
        return False
    # After prune, ``robot_learnable`` is stripped; treat as learnable if captions + skill ok.
    if learnable is None and "robot_learnable" in row:
        return False
    return True


def smpl_filled(row: Mapping[str, Any]) -> bool:
    return bool(_nonempty_str(row.get("smpl_path")))


def smpl_filled_for_backend(row: Mapping[str, Any], backend: str) -> bool:
    """True if ``smpl_path`` exists and ``smpl_backend`` matches (or is unset)."""
    if not smpl_filled(row):
        return False
    existing = _nonempty_str(row.get("smpl_backend"))
    if not existing:
        return True
    return existing == _nonempty_str(backend)


def get_hmr_text_prompt(row: Mapping[str, Any]) -> str:
    """Language prompt for PromptHMR (scene caption only)."""
    return get_caption(row)


def rows_caption_complete(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Samples with all caption-stage fields filled."""
    return [normalize_row(dict(r)) for r in rows if captions_filled(r)]


def rows_pending_smpl(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Caption-complete samples that still lack ``smpl_path``."""
    return [r for r in rows_caption_complete(rows) if not smpl_filled(r)]


def assert_all_captioned_have_smpl(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: str = "video2smpl",
) -> None:
    """Raise if any caption-complete row is missing SMPL (hard pipeline rule)."""
    pending = rows_pending_smpl(rows)
    if not pending:
        return
    ids = [str(r.get("sample_id", "?")) for r in pending[:20]]
    extra = f" (+{len(pending) - 20} more)" if len(pending) > 20 else ""
    raise RuntimeError(
        f"{context}: {len(pending)} caption-complete sample(s) still lack smpl_path "
        f"(e.g. {', '.join(ids)}{extra}). Re-run video2smpl or fix errors."
    )


def normalize_row(row: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure unified schema; migrate legacy ``text`` -> ``caption`` then drop ``text``."""
    out: Dict[str, Any] = dict(row)
    sid = _nonempty_str(out.get("sample_id"))
    if sid:
        out["sample_id"] = sid

    legacy_text = _nonempty_str(out.get("text"))
    if legacy_text and not _nonempty_str(out.get("caption")):
        out["caption"] = legacy_text
    out.pop("text", None)

    cap = get_caption(out)
    if cap and not _nonempty_str(out.get("caption")):
        out["caption"] = cap
    act = get_action_caption(out)
    if act and not _nonempty_str(out.get("action_caption")):
        out["action_caption"] = act

    out.setdefault("caption", "")
    out.setdefault("action_caption", "")
    out.setdefault("skill_category", "")
    if "robot_learnable" in out:
        if out["robot_learnable"] is not None and not isinstance(out["robot_learnable"], bool):
            out["robot_learnable"] = parse_robot_learnable(out["robot_learnable"])
    else:
        out.pop("robot_learnable", None)
    scat = normalize_skill_category(out.get("skill_category"))
    if scat:
        out["skill_category"] = scat
    out.setdefault("video_path", "")
    out.setdefault("select_status", "")
    out.setdefault("select_notes", "")
    out.setdefault("original_video_path", "")
    out.setdefault("rgb_path", "")
    out.setdefault("first_frame", "")
    out.setdefault("smpl_path", "")
    out.setdefault("smpl_backend", "")
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

    out.pop("text", None)
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
        if key in ("caption", "action_caption"):
            if _nonempty_str(old_val) and not _nonempty_str(out.get(key)):
                out[key] = old_val
            continue
        if key == "skill_category":
            if _nonempty_str(old_val) and not _nonempty_str(out.get(key)):
                out[key] = normalize_skill_category(old_val)
            continue
        if key == "robot_learnable":
            if isinstance(old_val, bool) and parse_robot_learnable(out.get(key)) is None:
                out[key] = old_val
            continue
        if old_val is not None and old_val != "" and (out.get(key) in (None, "", [])):
            out[key] = old_val
    out.pop("text", None)
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
            "robot_learnable": None,
            "skill_category": "",
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
    smpl_backend: str,
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
    base["smpl_backend"] = _nonempty_str(smpl_backend)
    mark_stage_completed(base, STAGE_VIDEO2SMPL)
    return merge_preserved_fields(base, old_row)


def apply_select_update(
    row: Dict[str, Any],
    *,
    video_path: str,
    select_status: str = "passed",
    select_notes: str = "",
    original_video_path: str = "",
    mark_complete: bool = False,
) -> Dict[str, Any]:
    """
    Partial select ingest (step1/step2): register ``video_path`` only.

    Leaves ``rgb_path`` empty and does not append ``select`` to
    ``stages_completed`` until the full select stage is finished.
    """
    out = normalize_row(row)
    out["video_path"] = video_path
    out["rgb_path"] = ""
    out["select_status"] = select_status
    out["select_notes"] = select_notes
    if original_video_path:
        out["original_video_path"] = original_video_path
    if mark_complete:
        mark_stage_completed(out, STAGE_SELECT)
    else:
        out["stages_completed"] = list(out.get("stages_completed") or [])
    return out


def apply_captions_update(
    row: Dict[str, Any],
    *,
    caption: str,
    action_caption: str,
    robot_learnable: bool,
    skill_category: str,
) -> Dict[str, Any]:
    out = normalize_row(row)
    out["caption"] = caption.strip()
    out["action_caption"] = action_caption.strip()
    out.pop("text", None)
    cat = normalize_skill_category(skill_category)
    if cat not in VALID_SKILL_CATEGORIES:
        raise ValueError(
            f"skill_category must be one of {sorted(VALID_SKILL_CATEGORIES)}, got {skill_category!r}"
        )
    out["skill_category"] = cat
    out["robot_learnable"] = bool(robot_learnable)
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
