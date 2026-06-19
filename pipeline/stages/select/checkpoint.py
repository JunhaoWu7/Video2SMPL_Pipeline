"""Select-stage filter checkpoint for crash-safe resume."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Literal, Tuple

SelectCheckpointStatus = Literal["passed", "rejected"]

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_REL = "logs/select_filter_checkpoint.json"


def canonical_source_key(work_root: Path, video_path: Path, src_rel: str) -> str:
    rel = str(src_rel or "").replace("\\", "/").strip().lstrip("/")
    if rel.startswith("video/"):
        return rel
    try:
        rel_to_root = str(video_path.resolve().relative_to(work_root.resolve())).replace("\\", "/")
        if rel_to_root.startswith("video/"):
            return rel_to_root
    except ValueError:
        pass
    return f"video/{video_path.name}"


def mapping_lookup_keys(rel: str, work_root: Path) -> list[str]:
    raw = str(rel or "").replace("\\", "/").strip()
    keys: list[str] = []
    for cand in (raw, raw.lstrip("/")):
        if cand and cand not in keys:
            keys.append(cand)
    path = Path(raw)
    if path.is_absolute():
        try:
            rel_to_root = str(path.resolve().relative_to(work_root.resolve())).replace("\\", "/")
            if rel_to_root and rel_to_root not in keys:
                keys.append(rel_to_root)
        except ValueError:
            pass
        abs_s = str(path.resolve())
        if abs_s not in keys:
            keys.append(abs_s)
    name = path.name
    if name:
        for cand in (name, f"video/{name}"):
            if cand not in keys:
                keys.append(cand)
    return keys


def resolve_checkpoint_path(work_root: Path, explicit: str | None = None) -> Path:
    if explicit and str(explicit).strip():
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (work_root / p).resolve()
        return p
    return (work_root / DEFAULT_CHECKPOINT_REL).resolve()


def checkpoint_jsonl_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".jsonl")


def clear_checkpoint_files(path: Path) -> None:
    for p in (path, checkpoint_jsonl_path(path)):
        if p.is_file():
            p.unlink()


def load_checkpoint(path: Path) -> Dict[str, SelectCheckpointStatus]:
    items, _vlm_called = load_checkpoint_with_vlm_called(path)
    return items


def load_checkpoint_with_vlm_called(
    path: Path,
) -> Tuple[Dict[str, SelectCheckpointStatus], Dict[str, bool]]:
    out: Dict[str, SelectCheckpointStatus] = {}
    vlm_called: Dict[str, bool] = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items") or {}
        for key, val in items.items():
            status = str(val.get("status") if isinstance(val, dict) else val).strip().lower()
            if status in ("passed", "rejected"):
                out[str(key)] = status  # type: ignore[assignment]
                if isinstance(val, dict) and "vlm_called" in val:
                    vlm_called[str(key)] = bool(val.get("vlm_called"))

    jsonl = checkpoint_jsonl_path(path)
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = str(rec.get("key", "")).strip()
                status = str(rec.get("status", "")).strip().lower()
                if key and status in ("passed", "rejected"):
                    out[key] = status  # type: ignore[assignment]
                    if "vlm_called" in rec:
                        vlm_called[key] = bool(rec.get("vlm_called"))
    return out, vlm_called


def save_checkpoint(
    path: Path,
    items: Dict[str, SelectCheckpointStatus],
    vlm_called: Dict[str, bool] | None = None,
) -> None:
    vlm_called = vlm_called or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(items),
        "vlm_called_count": sum(1 for k in items if vlm_called.get(k, False)),
        "vlm_called_known_count": sum(1 for k in items if k in vlm_called),
        "items": {
            k: (
                {"status": v, "vlm_called": bool(vlm_called[k])}
                if k in vlm_called
                else {"status": v}
            )
            for k, v in sorted(items.items())
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_checkpoint_entry(
    path: Path,
    key: str,
    status: SelectCheckpointStatus,
    *,
    vlm_called: bool = False,
) -> None:
    jsonl = checkpoint_jsonl_path(path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"key": key, "status": status, "vlm_called": bool(vlm_called)},
                ensure_ascii=False,
            )
            + "\n"
        )


class SelectCheckpointStore:
    """Thread-safe filter-result store; appends jsonl per item for fast crash-safe writes."""

    def __init__(
        self,
        path: Path,
        items: Dict[str, SelectCheckpointStatus] | None = None,
        vlm_called: Dict[str, bool] | None = None,
    ) -> None:
        self.path = path
        self.items: Dict[str, SelectCheckpointStatus] = dict(items or {})
        self.vlm_called: Dict[str, bool] = {
            k: bool(v) for k, v in dict(vlm_called or {}).items() if k in self.items
        }
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self.items)

    def get(self, key: str) -> SelectCheckpointStatus | None:
        return self.items.get(key)

    def get_vlm_called(self, key: str) -> bool:
        return bool(self.vlm_called.get(key, False))

    def vlm_called_count(self) -> int:
        return sum(1 for k in self.items if self.vlm_called.get(k, False))

    def vlm_called_known_count(self) -> int:
        return sum(1 for k in self.items if k in self.vlm_called)

    def set(self, key: str, status: SelectCheckpointStatus, *, vlm_called: bool = False) -> None:
        with self._lock:
            self.items[key] = status
            self.vlm_called[key] = bool(vlm_called)
            append_checkpoint_entry(self.path, key, status, vlm_called=vlm_called)

    def compact(self) -> None:
        with self._lock:
            if not self.items:
                clear_checkpoint_files(self.path)
                return
            save_checkpoint(self.path, self.items, self.vlm_called)
            jsonl = checkpoint_jsonl_path(self.path)
            if jsonl.is_file():
                jsonl.unlink()
