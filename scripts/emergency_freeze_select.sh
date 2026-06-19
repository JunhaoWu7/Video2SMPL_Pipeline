#!/usr/bin/env bash
# Freeze a running select process to stop burning VLM API quota.
# Memory is preserved until you SIGCONT or kill the process.
#
# Usage:
#   ./scripts/emergency_freeze_select.sh <pid>
#   ./scripts/emergency_freeze_select.sh <pid> --cont   # resume

set -euo pipefail

PID="${1:?Usage: $0 <pid> [--cont]}"
ACTION="${2:-stop}"

if [[ "$ACTION" == "--cont" ]]; then
  kill -CONT "$PID"
  echo "[emergency] Sent SIGCONT to PID $PID (process resumed)."
  exit 0
fi

kill -STOP "$PID"
echo "[emergency] Sent SIGSTOP to PID $PID."
echo "[emergency] Process frozen — no new VLM calls; ~100k filter results stay in RAM."
echo "[emergency] Next steps:"
echo "  1) Try gdb dump (needs sudo):"
echo "       sudo bash scripts/emergency_dump_legacy_select_gdb.sh $PID /path/to/dataset/hub"
echo "  2) Or kill and restart select with new checkpoint code (loses in-RAM results)."
echo "  3) Resume frozen process: $0 $PID --cont"
