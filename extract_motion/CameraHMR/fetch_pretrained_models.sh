#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/../../third_party/extract_motion/CameraHMR/fetch_pretrained_models.sh"

bash "${TARGET_SCRIPT}"
