"""
Renumber ``sample_id`` to contiguous ``000001``..``N`` after stages that drop samples.

Renames ``processed_trainable_data/<old_id>/`` directories and updates manifest /
mapping path fields that embed the sample directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from pipeline.dataset_schema import DIR_PROCESSED, sample_dir_rel

_PATH_FIELDS = ("video_path", "rgb_path", "first_frame", "smpl_path")
_RENUMBER_TMP_SUFFIX = "__renumber_tmp__"


def format_sample_id(seq: int, id_width: int) -> str:
    if seq < 1:
        raise ValueError(f"seq must be >= 1, got {seq}")
    return f"{seq:0{id_width}d}"


def _sort_key_sample_id(sample_id: str) -> int:
    sid = str(sample_id).strip()
    return int(sid) if sid.isdigit() else 0


def _replace_sample_dir_in_path(path: str, old_id: str, new_id: str) -> str:
    if not path:
        return path
    old_prefix = f"{DIR_PROCESSED}/{old_id}"
    new_prefix = f"{DIR_PROCESSED}/{new_id}"
    if old_prefix in path:
        return path.replace(old_prefix, new_prefix)
    return path


def _update_row_paths(row: MutableMapping[str, Any], old_id: str, new_id: str) -> None:
    row["sample_id"] = new_id
    for field in _PATH_FIELDS:
        val = row.get(field)
        if val is None:
            continue
        row[field] = _replace_sample_dir_in_path(str(val), old_id, new_id)


def renumber_sample_ids(
    work_root: Path,
    rows: Sequence[Mapping[str, Any]],
    mapping_items: Sequence[Mapping[str, str]],
    *,
    id_width: int = 6,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], Dict[str, str]]:
    """
    Return ``(new_rows, new_mapping, old_to_new)``.

    Rows are sorted by numeric ``sample_id`` before assigning ``000001..N``.
    """
    sorted_rows = sorted(
        (dict(r) for r in rows if str(r.get("sample_id", "")).strip()),
        key=lambda r: _sort_key_sample_id(str(r["sample_id"])),
    )
    if not sorted_rows:
        return [], [], {}

    old_to_new: Dict[str, str] = {}
    for i, row in enumerate(sorted_rows, start=1):
        old_id = str(row["sample_id"]).strip()
        new_id = format_sample_id(i, id_width)
        old_to_new[old_id] = new_id

    renames = [(old, new) for old, new in old_to_new.items() if old != new]
    processed_root = work_root / DIR_PROCESSED

    if renames and processed_root.is_dir() and not dry_run:
        for old_id, new_id in renames:
            old_dir = processed_root / old_id
            if not old_dir.is_dir():
                continue
            temp_dir = processed_root / f"{new_id}{_RENUMBER_TMP_SUFFIX}"
            if temp_dir.exists():
                raise FileExistsError(f"Renumber temp dir already exists: {temp_dir}")
            shutil.move(str(old_dir), str(temp_dir))

        for old_id, new_id in renames:
            temp_dir = processed_root / f"{new_id}{_RENUMBER_TMP_SUFFIX}"
            new_dir = processed_root / new_id
            if not temp_dir.is_dir():
                continue
            if new_dir.exists():
                raise FileExistsError(f"Renumber target dir already exists: {new_dir}")
            shutil.move(str(temp_dir), str(new_dir))

    new_rows: List[Dict[str, Any]] = []
    for row in sorted_rows:
        old_id = str(row["sample_id"]).strip()
        new_id = old_to_new[old_id]
        updated = dict(row)
        if old_id != new_id:
            _update_row_paths(updated, old_id, new_id)
        else:
            updated["sample_id"] = new_id
        new_rows.append(updated)

    kept_ids = set(old_to_new.keys())
    new_mapping: List[Dict[str, str]] = []
    for item in mapping_items:
        old_id = str(item.get("sample_id", "")).strip()
        if old_id not in kept_ids:
            continue
        new_id = old_to_new[old_id]
        updated = dict(item)
        updated["sample_id"] = new_id
        updated["output_sample_dir"] = sample_dir_rel(new_id)
        new_mapping.append(updated)

    new_mapping.sort(key=lambda it: _sort_key_sample_id(str(it.get("sample_id", ""))))
    for i, item in enumerate(new_mapping, start=1):
        item["seq_index"] = str(i)

    return new_rows, new_mapping, old_to_new
