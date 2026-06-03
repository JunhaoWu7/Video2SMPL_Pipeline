# video2smpl 阶段

视频 → canonical SMPL（`smpl_canonical.npz`）+ `first_frame.jpg`。

实现：`pipeline/stages/video2smpl/run.py`

## 处理流程

1. YOLO 人体跟踪  
2. CameraHMR 回归 `global_orient` / `body_pose` / `betas` / `transl`  
3. 插值 + 时序平滑  
4. `process_hmr_motion` canonicalization / 地面对齐（`set_floor`，默认开启）  

中间产物（bbox、raw SMPL 等）仅在临时目录计算，**不落盘**。

## `smpl_canonical.npz`

- `global_orient` (T,3)、`body_pose` (T,69)、`transl` (T,3)、`betas`  
- 另附 `intrinsic`、`frame_mask`、`bbox_xyxy`、`bbox_conf`、`set_floor`、`coord_note`

## 说明

- 通常排在 **prune** 之后；编排硬规则：同一轮若跑了 prune，必须接着跑本阶段。
- 默认只处理 manifest 里 caption 已齐的样本；失败样本会 WARN，不强制 exit。
- **export_splits** 为独立阶段（`export_skill_splits.py`），默认全流程会跑，但未编码为硬规则。

## 命令

```bash
# 全流程需先 select（提供 --select-input-dir）
python run.py --dataset humanvid \
  --select-input-dir /path/to/raw_videos

# 仅 SMPL（manifest 中 caption 已齐的样本全部会处理）
python run.py --dataset humanvid --from-stage video2smpl

python run.py --root_dir examples/training --source test \
  --stages video2smpl \
  --vendor_root third_party \
  --smooth_window 5 \
  --id_width 6
```

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--source` | 必填 | 写入 manifest 的 `source` |
| `--max_frames` | 500 | 每视频最多处理帧数 |
| `--smooth_window` | 5 | 平滑窗口 |
| `--set-floor` / `--no-set-floor` | 贴地开启 | 坐姿/躺姿多时可 `--no-set-floor` |
| `--overwrite` | off | 覆盖已有 `smpl_path` |
| `--person_idx` | 0 | 多人时选取的人物索引 |

## 环境与依赖

见 [weights.md](weights.md)。

```bash
conda create -n video2smpl python=3.10 -y
conda activate video2smpl
cd /path/to/Video2SMPL_Pipeline
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
# ffmpeg：系统包或 conda-forge
```

## 输出

每个样本目录：

```
processed_trainable_data/000001/
├── <video>.mp4
├── first_frame.jpg
└── smpl_canonical.npz
```

映射表 `sample_id_to_source.json` 记录 `sample_id` ↔ 入库前源路径。
