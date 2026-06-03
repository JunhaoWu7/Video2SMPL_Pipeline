# video2smpl 阶段

视频 → 每样本 SMPL npz + `first_frame.jpg`。支持两种 **HMR 后端**（仅本阶段内切换）：

| 后端 | 默认 | 输出文件 | 坐标 / 参数 |
|------|------|----------|-------------|
| **prompthmr** | 是 | `smpl_prompthmr.npz` | PromptHMR 重力世界系；`global_orient`(T,3) + `body_pose`(T,63) + `trans`(T,3) + `shape`(T,10) |
| **camerahmr** | 否 | `smpl_canonical.npz` | DART canonical；`global_orient` + `body_pose`(69) + `transl` + `betas` |

Manifest 字段：

- `smpl_path`：指向上述 npz 之一  
- `smpl_backend`：`"prompthmr"` | `"camerahmr"`

实现布局：

```
pipeline/stages/video2smpl/
  stage.py              # 编排入口（与其它 stage 一致）
  run.py                # 扫 manifest、分发后端
  common.py
  backends/
    prompthmr.py        # PromptHMR 分支
    camerahmr.py        # CameraHMR 分支
  vendor_bundle/        # bash scripts/copy_prompthmr_vendor.sh 拷贝（非软链）
  prompthmr_env.py          # 加载 phmr_pipeline（与 Video2SMPL pipeline 包不冲突）
  prompthmr_weights.py
```

## 处理流程

### prompthmr（默认）

1. 从 manifest 读取 `action_caption` + `caption` 拼成 **CLIP 文本 prompt**（缺则 **跳过并 WARN**）  
2. vendored `Pipeline`：检测跟踪 → 相机 → ViTPose → PHMR+PRHMR-Vid → **world** → post_opt  
3. 选最长 track，按视频时间轴写入 `smpl_prompthmr.npz`  

### camerahmr（可选）

1. YOLO 跟踪 → CameraHMR → 插值平滑 → `process_hmr_motion`（DART）  
2. 写入 `smpl_canonical.npz`  

## 命令

```bash
# 默认 PromptHMR：拷贝代码到 vendor_bundle；权重用绝对路径 /data1/wjh/ckpt/PromptHMR
bash scripts/copy_prompthmr_vendor.sh
python -m pipeline.stages.video2smpl.prompthmr_weights
# 非固定机位请加：
python -m pipeline.stages.video2smpl.prompthmr_weights --require-slam

python run.py --dataset humanvid --from-stage video2smpl

# 可选 CameraHMR
python run.py --dataset humanvid --from-stage video2smpl --hmr-backend camerahmr

# 固定机位（PromptHMR，默认已开启）
python run.py --dataset humanvid --from-stage video2smpl --static-camera
python run.py --dataset humanvid --from-stage video2smpl --no-static-camera
```

## 环境与权重

| 后端 | 环境 | 权重 |
|------|------|------|
| **prompthmr** | PromptHMR 官方 conda（`scripts/install.sh --world-video=true`） | 代码：`copy_prompthmr_vendor.sh`；权重：`/data1/wjh/ckpt/PromptHMR`（见 [prompthmr_vendor.md](prompthmr_vendor.md)） |
| **camerahmr** | 现有 `video2smpl` + `third_party` | [doc/weights.md](weights.md) |

权重检查：

```bash
python -m pipeline.stages.video2smpl.prompthmr_weights
```

| 文件 | 是否必需 |
|------|----------|
| `yolo11x.pt` | **否**（仅 `tracker: bytetrack`；默认 `sam2` + Detectron2） |
| `droid.pth` | **否**（SLAM 实际用 `droidcalib.pth`） |
| `droidcalib.pth` | **非固定机位必需**（`--no-static-camera`） |

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hmr-backend` | `prompthmr` | `prompthmr` 或 `camerahmr` |
| `--prompthmr-vendor` | `pipeline/stages/video2smpl/vendor_bundle` | vendored 代码与 `data/` 链接 |
| `--prompthmr-ckpt-root` | `/data1/wjh/ckpt/PromptHMR` | 权重根目录（校验用） |
| `--static-camera` | on | PromptHMR 固定机位 |
| `--max_frames` | 500 | 每视频最多帧数 |
| `--weight_root` | `/data1/wjh/Video2SMPL` | 仅 camerahmr |
| `--vendor_root` | `third_party` | 仅 camerahmr |
| `--set-floor` | on | 仅 camerahmr（DART 贴地） |

## 说明

- 通常排在 **prune** 之后；编排规则：同一轮若跑 prune，必须接着跑 video2smpl。  
- **prompthmr** 要求 captions 阶段已写入 `caption` 与 `action_caption`。  
- **export_splits** 为独立阶段，只要求 `smpl_path` 存在。
