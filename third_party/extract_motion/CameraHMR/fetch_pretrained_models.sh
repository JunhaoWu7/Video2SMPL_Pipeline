#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PRETRAIN_DIR="${DATA_DIR}/pretrained-models"
YOLO_DIR="${DATA_DIR}/yolo"

echo -e "\nYou need to register at https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR):" username
read -p "Password (CameraHMR):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p "${PRETRAIN_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${YOLO_DIR}"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=cam_model_cleaned.ckpt' -O "${PRETRAIN_DIR}/cam_model_cleaned.ckpt" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=camerahmr_checkpoint_cleaned.ckpt' -O "${PRETRAIN_DIR}/camerahmr_checkpoint_cleaned.ckpt" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=model_final_f05665.pkl' -O "${PRETRAIN_DIR}/model_final_f05665.pkl" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=smpl_mean_params.npz' -O "${DATA_DIR}/smpl_mean_params.npz" --no-check-certificate --continue

# YOLOv8x weights for person tracking
wget 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt' -O "${YOLO_DIR}/yolov8x.pt" --continue

echo -e "\nAll pretrained models downloaded successfully!"
