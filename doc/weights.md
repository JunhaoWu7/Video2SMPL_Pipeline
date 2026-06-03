# CameraHMR 权重放置

路径均相对于仓库根目录：`third_party/extract_motion/CameraHMR/data/`

## 必需文件

| 文件 | 路径 |
|------|------|
| SMPL_NEUTRAL.pkl | `data/models/SMPL/SMPL_NEUTRAL.pkl` |
| cam_model_cleaned.ckpt | `data/pretrained-models/` |
| camerahmr_checkpoint_cleaned.ckpt | `data/pretrained-models/` |
| model_final_f05665.pkl | `data/pretrained-models/` |
| smpl_mean_params.npz | `data/` |
| yolov8x.pt | `data/yolo/yolov8x.pt` |

另：`smplx_root.pt` → `scripts/data_processors/motion_alignment/`

## 自动下载

在 [CameraHMR 官网](https://camerahmr.is.tue.mpg.de/) 注册后：

```bash
cd /path/to/Video2SMPL_Pipeline
bash third_party/extract_motion/CameraHMR/fetch_smpl_model.sh
bash third_party/extract_motion/CameraHMR/fetch_pretrained_models.sh
```

## 自检脚本

```bash
cd /path/to/Video2SMPL_Pipeline
python - <<'PY'
from pathlib import Path
root = Path("third_party/extract_motion/CameraHMR/data")
files = [
    root / "models/SMPL/SMPL_NEUTRAL.pkl",
    root / "pretrained-models/cam_model_cleaned.ckpt",
    root / "pretrained-models/camerahmr_checkpoint_cleaned.ckpt",
    root / "pretrained-models/model_final_f05665.pkl",
    root / "smpl_mean_params.npz",
    root / "yolo/yolov8x.pt",
]
ok = all(p.exists() and p.stat().st_size > 0 for p in files)
for p in files:
    print(("OK   " if p.exists() and p.stat().st_size else "MISS "), p)
print("COMPLETE" if ok else "INCOMPLETE")
PY
```

若报错 `third_party/extract_motion/data/...`，说明路径逻辑过旧，应使用 `CameraHMR/data/...`。
