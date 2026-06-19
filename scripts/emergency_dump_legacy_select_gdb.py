"""GDB helper: dump legacy ``outcomes`` list from ``pipeline.stages.select.run.run``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    import gdb  # type: ignore
except ImportError as exc:  # pragma: no cover - only under gdb
    raise SystemExit("Run this file inside gdb (python ...)") from exc


def _py_obj_to_str(val) -> str:
    s = str(val)
    if s.startswith("'" ) and s.endswith("'"):
        return s[1:-1]
    return s


def _find_run_frame():
    inferior = gdb.selected_inferior()
    for thread in inferior.threads() or []:
        thread.switch()
        frame = gdb.newest_frame()
        while frame is not None:
            if frame.name() == "run":
                frame.select()
                return frame
            frame = frame.older()
    return None


def dump_outcomes(out_path: str) -> None:
    if _find_run_frame() is None:
        gdb.write("[dump] Could not find run() stack frame.\n")
        raise gdb.GdbError("run() frame not found")

    try:
        length = int(gdb.parse_and_eval("len(outcomes)"))
    except gdb.error as exc:
        gdb.write(f"[dump] outcomes not in scope: {exc}\n")
        raise gdb.GdbError("outcomes not found") from exc

    gdb.write(f"[dump] Found outcomes list, len={length}\n")

    items: dict[str, str] = {}
    for i in range(length):
        try:
            status = _py_obj_to_str(gdb.parse_and_eval(f"outcomes[{i}].status"))
            src_rel = _py_obj_to_str(gdb.parse_and_eval(f"outcomes[{i}].src_rel")).replace("\\\\", "/")
            if status not in ("passed", "rejected"):
                continue
            key = src_rel.lstrip("/")
            if not key.startswith("video/"):
                key = f"video/{Path(key).name}"
            items[key] = status
        except Exception as exc:  # pragma: no cover
            gdb.write(f"[dump] skip index {i}: {exc}\n")
            continue

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(items),
        "items": {k: {"status": v} for k, v in sorted(items.items())},
        "source": "emergency_gdb_dump",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    gdb.write(f"[dump] Saved {len(items)} filter results to {path}\n")
