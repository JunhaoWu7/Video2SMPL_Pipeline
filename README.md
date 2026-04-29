# Video2SMPL 权重放置说明

本文档用于说明：下载后的模型权重应该放到哪些路径，避免运行时报 `file not found`。

## 根目录约定

以下路径均相对于仓库根目录：

`/root/projects/Video2SMPL`

## 必需权重与目标路径

下载完成后，请确保这些文件存在且非空：

- `third_party/extract_motion/CameraHMR/data/models/SMPL/SMPL_NEUTRAL.pkl`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/cam_model_cleaned.ckpt`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/model_final_f05665.pkl`
- `third_party/extract_motion/CameraHMR/data/smpl_mean_params.npz`
- `third_party/extract_motion/CameraHMR/data/yolo/yolov8x.pt`

## 推荐下载方式（自动放到正确位置）

```bash
cd /root/projects/Video2SMPL
bash extract_motion/CameraHMR/fetch_smpl_model.sh
bash extract_motion/CameraHMR/fetch_pretrained_models.sh
```

> 需要先在 [CameraHMR 官网](https://camerahmr.is.tue.mpg.de/) 注册并准备好用户名。

## 手动下载放置规则

如果你是手动下载，请按“文件名 -> 目标目录”放置：

- `SMPL_NEUTRAL.pkl` -> `third_party/extract_motion/CameraHMR/data/models/SMPL/`
- `cam_model_cleaned.ckpt` -> `third_party/extract_motion/CameraHMR/data/pretrained-models/`
- `camerahmr_checkpoint_cleaned.ckpt` -> `third_party/extract_motion/CameraHMR/data/pretrained-models/`
- `model_final_f05665.pkl` -> `third_party/extract_motion/CameraHMR/data/pretrained-models/`
- `smpl_mean_params.npz` -> `third_party/extract_motion/CameraHMR/data/`
- `yolov8x.pt` -> `third_party/extract_motion/CameraHMR/data/yolo/`

## 一键自检

```bash
python - <<'PY'
from pathlib import Path
root = Path('/root/projects/Video2SMPL')
files = [
    root / 'third_party/extract_motion/CameraHMR/data/models/SMPL/SMPL_NEUTRAL.pkl',
    root / 'third_party/extract_motion/CameraHMR/data/pretrained-models/cam_model_cleaned.ckpt',
    root / 'third_party/extract_motion/CameraHMR/data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt',
    root / 'third_party/extract_motion/CameraHMR/data/pretrained-models/model_final_f05665.pkl',
    root / 'third_party/extract_motion/CameraHMR/data/smpl_mean_params.npz',
    root / 'third_party/extract_motion/CameraHMR/data/yolo/yolov8x.pt',
]
ok = True
for p in files:
    good = p.exists() and p.stat().st_size > 0
    print(('OK   ' if good else 'MISS '), p)
    ok = ok and good
print('COMPLETE' if ok else 'INCOMPLETE')
PY
```

