#!/usr/bin/env bash
# Copy PromptHMR into vendor_bundle; rename package pipeline -> phmr_pipeline (no name clash).
# Weights: absolute CKPT_ROOT (/data1/wjh/ckpt/PromptHMR).
#
# Usage: bash scripts/copy_prompthmr_vendor.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR="${REPO_ROOT}/pipeline/stages/video2smpl/vendor_bundle"

PROMPTHMR_SRC="${PROMPTHMR_SRC:-/home/wujunhao/code/PromptHMR}"
CKPT_ROOT="${CKPT_ROOT:-/data1/wjh/ckpt/PromptHMR}"

RSYNC_EXCLUDE=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.git/'
  --exclude 'results/'
)

if [ ! -d "${PROMPTHMR_SRC}/pipeline" ] || [ ! -d "${PROMPTHMR_SRC}/prompt_hmr" ]; then
  echo "ERROR: ${PROMPTHMR_SRC} must contain pipeline/ and prompt_hmr/" >&2
  exit 1
fi

echo "Copying PromptHMR into vendor_bundle..."
mkdir -p "${VENDOR}"
rm -rf "${VENDOR}/pipeline" "${VENDOR}/phmr_pipeline" "${VENDOR}/prompt_hmr" "${VENDOR}/data" 2>/dev/null || true

rsync -a "${RSYNC_EXCLUDE[@]}" "${PROMPTHMR_SRC}/pipeline/" "${VENDOR}/pipeline/"
rsync -a "${RSYNC_EXCLUDE[@]}" "${PROMPTHMR_SRC}/prompt_hmr/" "${VENDOR}/prompt_hmr/"
[ -f "${PROMPTHMR_SRC}/data_config.py" ] && cp -f "${PROMPTHMR_SRC}/data_config.py" "${VENDOR}/data_config.py"

echo "${CKPT_ROOT}" > "${VENDOR}/CKPT_ROOT.txt"

# MCS skip patch (before package rename)
python3 <<PY
from pathlib import Path
import os
repo = Path("${VENDOR}")
path = repo / "pipeline" / "pipeline.py"
text = path.read_text(encoding="utf-8")
if "VIDEO2SMPL_SKIP_MCS_EXPORT" not in text:
    old = """        joblib.dump(self.results, f'{seq_folder}/results.pkl')
        
        NUM_FRAMES = len(self.images)"""
    new = """        joblib.dump(self.results, f'{seq_folder}/results.pkl')

        if os.environ.get("VIDEO2SMPL_SKIP_MCS_EXPORT", "").strip() in ("1", "true", "yes"):
            return self.results
        
        NUM_FRAMES = len(self.images)"""
    if old not in text:
        raise SystemExit("ERROR: could not patch pipeline.py")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("Patched pipeline.py (skip MCS export)")
PY

python3 "${SCRIPT_DIR}/rename_phmr_pipeline_pkg.py"

cat > "${VENDOR}/README.md" <<EOF
# PromptHMR vendor bundle

- \`phmr_pipeline/\` — renamed from upstream \`pipeline/\` (avoids clash with Video2SMPL \`pipeline\` package)
- \`prompt_hmr/\` — PHMR model code
- \`data/\` — runtime symlinks to \`${CKPT_ROOT}\` (created on first inference)

Refresh: \`bash scripts/copy_prompthmr_vendor.sh\`
EOF

echo "Done. $(du -sh "${VENDOR}/phmr_pipeline" "${VENDOR}/prompt_hmr" 2>/dev/null | tr '\n' ' ')"
echo "Check: python -m pipeline.stages.video2smpl.prompthmr_weights"
