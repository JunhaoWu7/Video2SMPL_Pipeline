#!/usr/bin/env bash
# Dump in-memory select filter outcomes from a LEGACY run (pre-checkpoint code)
# into logs/select_filter_checkpoint.json for resume with current pipeline code.
#
# Requires: sudo, gdb with Python support (python3-dbg / libc python gdb hooks)
#
# Usage:
#   sudo bash scripts/emergency_dump_legacy_select_gdb.sh <pid> <dataset_hub_root>
#
# Example:
#   sudo bash scripts/emergency_dump_legacy_select_gdb.sh 857098 /data1/wjh/HumanRetarget/kinetics_700_2020

set -euo pipefail

PID="${1:?pid required}"
WORK_ROOT="${2:?dataset hub root required}"
OUT="${WORK_ROOT}/logs/select_filter_checkpoint.json"
PY_HELPER="$(cd "$(dirname "$0")" && pwd)/emergency_dump_legacy_select_gdb.py"

if [[ ! -f "$PY_HELPER" ]]; then
  echo "Missing helper: $PY_HELPER" >&2
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID not found" >&2
  exit 1
fi

echo "[dump] Attaching gdb to PID $PID ..."
gdb -batch -p "$PID" \
  -ex "set pagination off" \
  -ex "python exec(open('${PY_HELPER}').read()); dump_outcomes('${OUT}')" \
  2>&1 | tee "${WORK_ROOT}/logs/emergency_gdb_dump.log"

if [[ -f "$OUT" ]]; then
  COUNT=$(python3 -c "import json; d=json.load(open('$OUT')); print(d.get('count', len(d.get('items',{}))))")
  echo "[dump] Wrote $OUT ($COUNT items)."
  echo "[dump] Restart select ONLY (no captions) to ingest passed clips without re-filtering:"
  echo "  cd $(dirname "$PY_HELPER")/.."
  echo "  python run.py --hub-root $(dirname "$WORK_ROOT") --dataset $(basename "$WORK_ROOT") \\"
  echo "    --stages select --select-symlink --select-workers 16 --source <SOURCE>"
else
  echo "[dump] FAILED — see ${WORK_ROOT}/logs/emergency_gdb_dump.log" >&2
  exit 1
fi
