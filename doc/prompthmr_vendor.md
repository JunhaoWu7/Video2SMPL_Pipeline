# PromptHMR 内嵌代码与权重（video2smpl）

video2smpl 默认后端 **prompthmr** 在仓库内自带一份 PromptHMR 源码拷贝，**不依赖** 外部 `~/code/PromptHMR` 目录（升级上游后需重新 copy）。

## 目录

```
pipeline/stages/video2smpl/
  vendor_bundle/          # 由 copy 脚本生成（真实文件）
    phmr_pipeline/        # 由原 upstream ``pipeline/`` 重命名，避免与 Video2SMPL ``pipeline`` 包冲突
    prompt_hmr/           # PHMR 模型
    CKPT_ROOT.txt         # 记录权重根路径
    README.md
  prompthmr_paths.py      # 权重绝对路径注入
  prompthmr_env.py        # sys.path + cwd + caption patch
  prompthmr_weights.py    # 权重自检
  backends/prompthmr.py
```

## 一次性准备

```bash
cd /path/to/Video2SMPL_Pipeline

# 1) 从上游 PromptHMR 拷贝代码（可重复执行以更新）
bash scripts/copy_prompthmr_vendor.sh
# 或指定源目录：
# PROMPTHMR_SRC=/path/to/PromptHMR bash scripts/copy_prompthmr_vendor.sh

# 2) 检查权重（绝对路径，默认 /data1/wjh/ckpt/PromptHMR）
python -m pipeline.stages.video2smpl.prompthmr_weights

# 非固定机位视频：
python -m pipeline.stages.video2smpl.prompthmr_weights --require-slam
```

**权重不在 vendor_bundle 里**；首次推理会在 `vendor_bundle/data/` 建立指向 ckpt 的软链（不复制权重文件）。

PromptHMR 视频管线 Python 包名为 **`phmr_pipeline`**（不是 `pipeline`），可与 Video2SMPL 的 `pipeline` 包在同一进程内加载，**无需子进程 worker**。

运行时权重路径：

- `/data1/wjh/ckpt/PromptHMR/pretrain/...`
- `/data1/wjh/ckpt/PromptHMR/body_models/...`
- `/data1/wjh/smpl_ckpt/smplx/SMPLX_NEUTRAL.npz`

可用 `--prompthmr-ckpt-root` 或环境变量 `VIDEO2SMPL_PROMPTHMR_CKPT` 覆盖。

## Conda / pip 环境

拷贝的只是 **Python 源码**；仍需 PromptHMR 官方环境（CUDA、detectron2、SAM2、smplx 等），见 PromptHMR `scripts/install.sh --world-video=true`。

本 pipeline 会设置 `VIDEO2SMPL_SKIP_MCS_EXPORT=1`，跳过 Meshcapade MCS/GLB 导出（无需 `smplcodec`）。

## 可选权重说明

| 文件 | 用途 | 缺失影响 |
|------|------|----------|
| `yolo11x.pt` | 仅 `tracker: bytetrack` 时 YOLO 人体检测 | **无影响**（默认 `sam2` + Detectron2） |
| `droid.pth` | 写在 `data_dict`，**当前 SLAM 未使用** | **无影响** |
| `droidcalib.pth` | 非固定机位 **DROID-SLAM** 权重 | `--no-static-camera` 时 **必需**，否则相机轨迹失败 |

## 集成自检（phmr_pt2.4）

```bash
conda activate phmr_pt2.4
python -m pipeline.stages.video2smpl.prompthmr_weights
python scripts/test_prompthmr_integration.py --max-frames 24   # 需空闲 GPU（SAM2 阶段约需数 GB）
```

## 与 CameraHMR 后端

`--hmr-backend camerahmr` 不使用 `vendor_bundle`，仍走 `third_party/extract_motion`，权重见 [weights.md](weights.md)。
