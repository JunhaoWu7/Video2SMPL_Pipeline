examples/training/
└── processed_trainable_data/
    └── <000001>/
        ├── your_source_name.mp4      # select 从 --select-input-dir 移入
        ├── first_frame.jpg
        └── smpl_canonical.npz        # 唯一大文件产物（canonical SMPL）

`smpl_canonical.npz` 内容（与下游训练一致）：
    global_orient (T,3)、body_pose (T,69)、transl (T,3)、betas
    另附 intrinsic、frame_mask、bbox_xyxy、bbox_conf、set_floor、coord_note

默认**不保存** raw HMR（`smpl_raw.pt`）、incam 平滑 npz、`motion_postprocess.pt`；中间结果仅在临时目录计算。



步骤简介

1.yolo跟踪人（echomotion）
2.camerahmr 回归global_orient/body_pose/betas/transl 并且可以输出verts/joints,保留相机信息（echomotion）
3.后处理echomotion的办法对序列做插值和时序平滑（echomotion）
4.用 process_hmr_motion 做 canonicalization/坐标系变换/地面对齐（set_floor=True）（echomotion）
第 4 步起写入**同一份**渐进式 manifest（默认 `dataset_manifest.json`），各阶段在原文件上增补字段，不再每步新建 `train_stage4/5_*.json`。

单条样本示例（字段随阶段逐步填满）：

```json
{
  "sample_id": "000001",
  "original_video": "your_source_name.mp4",
  "video_path": "processed_trainable_data/000001/dance_01.mp4",
  "rgb_path": "processed_trainable_data/000001/dance_01.mp4",
  "first_frame": "processed_trainable_data/000001/first_frame.jpg",
  "caption": "A person walks forward across the room.",
  "action_caption": "walking forward",
  "smpl_path": "processed_trainable_data/000001/smpl_canonical.npz",
  "select_status": "",
  "type": "video",
  "source": "your_dataset_or_batch_label",
  "link": "",
  "stages_completed": ["video2smpl", "captions"],
  "text": "A person walks forward across the room."
}
```

| 阶段 | 写入字段 |
|------|----------|
| select | `video_path`, `rgb_path`, `select_status`, …（视频移入 `processed_trainable_data/<id>/`） |
| captions | `caption`, `action_caption` |
| video2smpl | `first_frame`, `smpl_path`（仅 canonical npz） |

`text` 为兼容旧下游保留，与 `caption` 同步。
5.调用大模型打标（本仓库脚本）（comovi）

依赖（可与 SMPL 环境分开装）：`pip install openai pillow httpx`；并设置 `OPENAI_API_KEY` 或 `OPENROUTER_API_KEY`（与脚本内 OpenRouter 默认 `base_url` 一致时可配 `OPENAI_BASE_URL`）。

在 **`--root_dir`** 下已有 `dataset_manifest.json`（且含 `video_path` 或 `rgb_path`）后，打标阶段从视频均匀抽帧，**原地**更新同文件中的 **`caption`**（1–2 句）与 **`action_caption`**（动作短语）。`generate_sequence_captions.py` 默认 `--output-manifest` 与 `--manifest` 相同（in-place）。

```bash
cd /path/to/Video2SMPL_Pipeline
# 只检查路径与抽帧（不调 API）
python generate_sequence_captions.py --dry-run \
  --manifest examples/training/dataset_manifest.json \
  --pipeline-root examples/training

python generate_sequence_captions.py \
  --manifest examples/training/dataset_manifest.json \
  --pipeline-root examples/training \
  --model openai/gpt-4o \
  --num-frames 12 \
  --caption-lang en \
  --resume
```

说明：

- `--workers`：并行请求数，**默认 4**；显式写 `--workers 1` 则退化为完全串行。若出现 **429 / rate limit**，把并发调小或略增大 `--sleep`。
- 已同时填好 `caption` 与 `action_caption` 的样本会跳过；`--force-recaption` 可强制重打。
- 若本机已装 **OpenCV**，优先从 `video_path` / `rgb_path` 抽多帧；否则 fallback `first_frame`。
- 旧 manifest（`train_stage4/5_*.json`、`text` 字段）首次加载会自动迁移到 `caption`。

6.组织成json格式（打标后 `text` 已写在输出 manifest 各条中，可直接用于训练/下游）（comovi）
[
  {
    "sample_id": "000001",
    "original_video": "your_source_name.mp4",
    "rgb_path": "processed_trainable_data/000001/your_source_name.mp4",
    "first_frame": "processed_trainable_data/000001/first_frame.jpg",
    "smpl_path": "processed_trainable_data/000001/smpl_canonical.npz",
    "caption": "A person performs a smooth yoga transition from plank to downward dog.",
    "action_caption": "yoga transition to downward dog",
    "type": "video",
    "source": "your_dataset_or_batch_label",
    "link": "optional_url_or_empty"
  }
]


多源外来 SMPL：先“对齐坐标语义”，再平滑。

---

## Pipeline 架构（多子阶段）

顶层编排入口（推荐）：

- `run.py`：按顺序执行一个或多个子阶段，支持 `--stages` 与 `--from-stage`

子阶段目录：

| 阶段名 | 路径 | 说明 |
|--------|------|------|
| `video2smpl` | `pipeline/stages/video2smpl/` | 视频 → SMPL（原 steps 1–4 + stage4 manifest） |
| `captions` | `pipeline/stages/captions/` | 大模型打标（封装 `generate_sequence_captions.py`） |
| `external_smpl` | `pipeline/stages/external_smpl/` | 外来 SMPL 对齐（独立链路） |

向后兼容（仍可用）：

- `pipeline/run_pipeline.py` → 等价于只跑 `video2smpl`
- `pipeline/process_external_smpl.py` → 等价于只跑 `external_smpl`

```bash
# 列出阶段与固定顺序
python run.py --list-stages

# 全流程（固定顺序：select -> captions -> video2smpl）
python run.py --root_dir examples/training --source my_dataset

# 从打标开始（跳过 select；manifest 需已有 video_path）
python run.py --root_dir examples/training --source my_dataset --from-stage captions

# 仅 SMPL（manifest 需已有 video_path，caption 可选）
python run.py --root_dir examples/training --source my_dataset --from-stage video2smpl

# 仅注册视频路径（select passthrough）
python run.py --root_dir examples/training --source my_dataset --from-stage select
```

`--stages` 若指定多个阶段，会**自动按** `select,captions,video2smpl` **排序**，不会按书写顺序执行。

---

已实现脚本：`pipeline/stages/video2smpl/run.py`（兼容入口：`pipeline/run_pipeline.py`）

用途：
- 复用 EchoMotion 的 1~4 步核心能力：
  - YOLO 跟踪
  - CameraHMR 回归
  - 线性插值 + 高斯时序平滑
  - `process_hmr_motion` canonicalization / 坐标变换 / set_floor（CLI 默认开启）
- 参考 CoMoVi 的组织方式，产出第4步后的空文本 JSON（第5/6步打标暂不执行）

## 环境与依赖（完整下载清单）

建议使用 Python 3.10 + CUDA 对齐的 PyTorch 环境（`torch/torchvision/torchaudio` 版本需互相匹配）。

1) 创建环境（示例）
```bash
conda create -n video2smpl python=3.10 -y
conda activate video2smpl
```

2) 安装 Python 依赖（本 pipeline 最小必需）
```bash
cd /root/projects/Video2SMPL
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt 
```

如果chumpy报错，可以单独这样下载

python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel

pip install chumpy==0.70 --no-build-isolation 
# 或者直接
pip install -r requirements.txt -i https://pypi.org/simple


3) 安装 detectron2（CameraHMR 需要）
```bash
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git" -i https://pypi.org/simple
```

4) 系统依赖（ffmpeg）
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg

conda clean -i
conda install -c conda-forge ffmpeg -y
ffmpeg -version
```

运行前准备：
- select 用 `--select-input-dir` 指定待入库视频目录（支持 `mp4/mov/avi/mkv`，递归扫描）；每条移入 `processed_trainable_data/<id>/`
- 输出目录名使用**递增序号**（默认 6 位零填充：`000001`、`000002`…），与原始文件名无关；排序按**文件名字典序**稳定遍历
- 映射表：`examples/training/sample_id_to_source.json`（序号 ↔ 原始路径/文件名），打标或合并数据时用该文件回溯
- 本仓库已内置依赖代码到 `third_party/`（无需再依赖外部 EchoMotion/CoMoVi 仓库）

模型参数下载（首次必须）：
```bash
cd /root/projects/Video2SMPL

# 兼容你给的命令路径（会转发到 third_party 下真实脚本）
bash extract_motion/CameraHMR/fetch_smpl_model.sh
bash extract_motion/CameraHMR/fetch_pretrained_models.sh
```
需要注册 https://camerahmr.is.tue.mpg.de/ 输入用户名

详细权重放置说明见：`WEIGHTS_PLACEMENT_README.md`


下载后文件位置：
- `third_party/extract_motion/CameraHMR/data/models/SMPL/SMPL_NEUTRAL.pkl`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/cam_model_cleaned.ckpt`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt`
- `third_party/extract_motion/CameraHMR/data/pretrained-models/model_final_f05665.pkl`
- `third_party/extract_motion/CameraHMR/data/smpl_mean_params.npz`
- `third_party/extract_motion/CameraHMR/data/yolo/yolov8x.pt`

说明：
- 代码加载权重已改为基于 `third_party/extract_motion/CameraHMR/core/constants.py` 的绝对路径拼接，不依赖外部仓库目录。
- `fetch_pretrained_models.sh` 现在会同时下载 YOLO 跟踪权重（`yolov8x.pt`），不需要首跑时再自动拉取。

常见报错排查：
- 若出现 `Path .../third_party/extract_motion/data/... does not exist`，说明使用了旧版路径逻辑；请更新到当前代码版本（应固定读取 `third_party/extract_motion/CameraHMR/data/...`）。
- 再执行一次权重自检（见下方脚本），确认 `SMPL_NEUTRAL.pkl` 与 `pretrained-models/*.ckpt` 均存在。

5) 一键检查权重是否齐全（可选）
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

运行示例：
```bash
cd /path/to/Video2SMPL_Pipeline
# 推荐：顶层编排
python run.py \
  --root_dir examples/training \
  --source "test" \
  --stages video2smpl \
  --vendor_root third_party \
  --smooth_window 5 \
  --id_width 6

# 或沿用旧入口
python pipeline/run_pipeline.py \
  --root_dir examples/training \
  --source "test" \
  --vendor_root third_party \
  --smooth_window 5 \
  --id_width 6
```
（`--set-floor` 默认已开启；若不要贴地请加 `--no-set-floor`。）

必选参数：

- `--source`：非空字符串，**必须在开始处理前通过命令行提供**；写入 `train_stage4_empty_text.json` 中每条样本的 `source` 字段（同一 `root_dir` 下本次生成的 manifest 共用该标签，重跑会按本次传入值覆盖各条目的 `source`）

可选参数：
- `--set-floor` / `--no-set-floor`：canonical 贴地，**默认开启**（`--set-floor`）；坐姿/躺姿多或不想抬到地面时用 `--no-set-floor`
- `--id_width`：序号零填充位数，**默认 6**（`000001` …）
- `--max_frames`：每个视频最多处理多少帧，**默认 500**
- `--mapping_name`：映射文件名，默认 `sample_id_to_source.json`
- **追加模式（默认）**：若 `sample_id_to_source.json` 已存在，已登记过的源路径（`original_path_relative`）会**跳过**；仅对新视频从「当前最大 sample 编号 + 1」继续编号
- `--overwrite`：对**已在映射中的**视频强制重跑，**沿用原 sample_id** 覆盖输出；新视频仍走追加编号
- 重建 `train_stage4_empty_text.json` 时会按映射合并全量条目，并尽量**保留**已有条目的 `text`；`source` 始终为本次运行的 `--source`；`link` 若旧条目非空则保留，否则用本次 `--link` / `--default_link`
- 请勿随意删除 `sample_id_to_source.json`：否则无法识别「旧视频对应哪个序号」，新跑可能给同一批视频分配新的编号（与已有目录重复风险）；备份该文件即可安全追加

输出：
- `processed_trainable_data/<000001>/` — 视频 + `first_frame.jpg` + `smpl_canonical.npz`
- `examples/training/sample_id_to_source.json`
  - `items[]`：`sample_id`、`seq_index`、`original_filename`、`original_stem`、`original_path_relative`、`output_sample_dir`
- `examples/training/dataset_manifest.json`
  - 全 pipeline 共用；含 `video_path`、`caption`、`action_caption`、`smpl_path` 等，随阶段逐步完善
