# pipeline介绍

### 1.video2smpl，查看Video2SMPL_Readme.md
### 2.外部smpl处理,查看process_external_smpl_README.md

# Video2SMPL 权重放置说明

本文档用于说明：下载后的模型权重应该放到哪些路径，避免运行时报 `file not found` 或权重文件损坏。

## 权重根目录

以下路径均相对于仓库根目录（`/root/projects/Video2SMPL`）：

`third_party/extract_motion/CameraHMR/data/`

## 必需权重与目标路径

请确保这些文件**存在且非空**：

- `third_party/extract_motion/CameraHMR/data/models/SMPL/SMPL_NEUTRAL.pkl`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/cam_model_cleaned.ckpt`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/model_final_f05665.pkl`
- `third_party/extract_motion/CameraHMR/data/smpl_mean_params.npz`
- `third_party/extract_motion/CameraHMR/data/yolo/yolov8x.pt`

## 推荐下载方式（自动放到正确位置）

需要先在 [CameraHMR 官网](https://camerahmr.is.tue.mpg.de/) 注册并准备好用户名密码：

```bash
cd /root/projects/Video2SMPL
bash extract_motion/CameraHMR/fetch_smpl_model.sh
bash extract_motion/CameraHMR/fetch_pretrained_models.sh
```

## 手动下载放置规则

- `SMPL_NEUTRAL.pkl` -> `third_party/extract_motion/CameraHMR/data/models/SMPL/`
- `cam_model_cleaned.ckpt` / `camerahmr_checkpoint_cleaned.ckpt` / `model_final_f05665.pkl`
  -> `third_party/extract_motion/CameraHMR/data/pretrained-models/`
- `smpl_mean_params.npz` -> `third_party/extract_motion/CameraHMR/data/`
- `yolov8x.pt` -> `third_party/extract_motion/CameraHMR/data/yolo/`

## 一键自检（存在性 + 基础可读性）

```bash
cd /root/projects/Video2SMPL
python - <<'PY'
from pathlib import Path
import torch
import numpy as np
import pickle

root = Path('/root/projects/Video2SMPL/third_party/extract_motion/CameraHMR/data')
items = [
    ('SMPL_NEUTRAL.pkl', root / 'models/SMPL/SMPL_NEUTRAL.pkl', 'pickle'),
    ('cam_model_cleaned.ckpt', root / 'pretrained-models/cam_model_cleaned.ckpt', 'torch'),
    ('camerahmr_checkpoint_cleaned.ckpt', root / 'pretrained-models/camerahmr_checkpoint_cleaned.ckpt', 'torch'),
    ('model_final_f05665.pkl', root / 'pretrained-models/model_final_f05665.pkl', 'pickle'),
    ('smpl_mean_params.npz', root / 'smpl_mean_params.npz', 'npz'),
    ('yolov8x.pt', root / 'yolo/yolov8x.pt', 'torch'),
]

def ok(msg): print('[OK] ', msg)
def miss(msg): print('[MISS]', msg)

all_ok = True
for name, p, mode in items:
    if not p.exists() or p.stat().st_size <= 0:
        miss(f'{name} missing/empty: {p}')
        all_ok = False
        continue
    try:
        if mode == 'pickle':
            with p.open('rb') as f:
                pickle.load(f, encoding='latin1')
        elif mode == 'npz':
            z = np.load(p, allow_pickle=True)
            _ = list(z.files)
            z.close()
        elif mode == 'torch':
            torch.load(p, map_location='cpu', weights_only=False)
        else:
            pass
        ok(f'{name} readable: {p}')
    except Exception as e:
        miss(f'{name} NOT readable: {e}')
        all_ok = False

print('COMPLETE' if all_ok else 'INCOMPLETE')
PY
```

