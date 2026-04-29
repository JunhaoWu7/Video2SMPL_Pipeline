examples/training/
├── CameraHMR_smpl_results/           # raw HMR results
└── CameraHMR_smpl_results_overlay/   # raw HMR re-projection results for sanity check
└── CameraHMR_smpl_results_smoothed/  # smoothed HMR results
└── rgb_videos/      
└── processed_trainable_data /


CameraHMR_smpl_results/<000001>/smpl_raw.pt（目录名为递增序号，见 `sample_id_to_source.json`）

这是直接从 extract_smpl 保存的原始结果，包含：
    smpl_params_incam（含 global_orient/body_pose/transl/betas）
    focal_length、width、height
    intrinsic
    cam_t
    verts、joints、fps 等

CameraHMR_smpl_results_smoothed/<000001>/smpls_smoothed_group.npz

    这是平滑后导出的包，包含：
    global_orient/body_pose/transl/betas
    intrinsic
    frame_mask、bbox_xyxy、bbox_conf
    set_floor



步骤简介

1.yolo跟踪人（echomotion）
2.camerahmr 回归global_orient/body_pose/betas/transl 并且可以输出verts/joints,保留相机信息（echomotion）
3.后处理echomotion的办法对序列做插值和时序平滑（echomotion）
4.用 process_hmr_motion 做 canonicalization/坐标系变换/地面对齐（set_floor=True）（echomotion）
第4步结束后组织成json格式，text字段暂时留空（comovi）
[
  {
    "sample_id": "000001",
    "original_video": "your_source_name.mp4",
    "rgb_path": "processed_trainable_data/000001/rgb.mp4",
    "first_frame": "processed_trainable_data/000001/first_frame.jpg",
    "smpl_path": "CameraHMR_smpl_results_smoothed/000001/smpls_smoothed_group.npz",
    "text": "",
    "type": "video",
    "source": "your_dataset_or_batch_label",
    "link": "optional_url_or_empty"
  }
]
5.调用大模型打标（分开做）（comovi）
6.组织成json格式（打标后确保文本被写入对应数据条）（comovi）
[
  {
    "sample_id": "000001",
    "original_video": "your_source_name.mp4",
    "rgb_path": "processed_trainable_data/000001/rgb.mp4",
    "first_frame": "processed_trainable_data/000001/first_frame.jpg",
    "smpl_path": "CameraHMR_smpl_results_smoothed/000001/smpls_smoothed_group.npz",
    "text": "A person performs a smooth yoga transition from plank to downward dog.",
    "type": "video",
    "source": "your_dataset_or_batch_label",
    "link": "optional_url_or_empty"
  }
]


多源外来 SMPL：先“对齐坐标语义”，再平滑。

---

已实现脚本：`pipeline/run_pipeline.py`

用途：
- 复用 EchoMotion 的 1~4 步核心能力：
  - YOLO 跟踪
  - CameraHMR 回归
  - 线性插值 + 高斯时序平滑
  - `process_hmr_motion` canonicalization / 坐标变换 / 可选 set_floor
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
python -m pip install -r requirements.txt -i 
```

3) 安装 detectron2（CameraHMR 需要）
```bash
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git" -i https://pypi.org/simple
```

4) 系统依赖（ffmpeg）
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

运行前准备：
- `examples/training/rgb_videos/` 下放待处理视频（支持 `mp4/mov/avi/mkv`）
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
cd /root/projects/Video2SMPL
python pipeline/run_pipeline.py \
  --root_dir examples/training \
  --source "my_batch_or_dataset_name" \
  --vendor_root third_party \
  --set_floor \
  --smooth_window 5 \
  --id_width 6 \
```

必选参数：

- `--source`：非空字符串，**必须在开始处理前通过命令行提供**；写入 `train_stage4_empty_text.json` 中每条样本的 `source` 字段（同一 `root_dir` 下本次生成的 manifest 共用该标签，重跑会按本次传入值覆盖各条目的 `source`）

可选参数：
- `--id_width`：序号零填充位数，**默认 6**（`000001` …）
- `--mapping_name`：映射文件名，默认 `sample_id_to_source.json`
- **追加模式（默认）**：若 `sample_id_to_source.json` 已存在，已登记过的 `rgb_videos` 下相对路径会**跳过**；仅对新视频从「当前最大 sample 编号 + 1」继续编号
- `--overwrite`：对**已在映射中的**视频强制重跑，**沿用原 sample_id** 覆盖输出；新视频仍走追加编号
- 重建 `train_stage4_empty_text.json` 时会按映射合并全量条目，并尽量**保留**已有条目的 `text`；`source` 始终为本次运行的 `--source`；`link` 若旧条目非空则保留，否则用本次 `--link` / `--default_link`
- 请勿随意删除 `sample_id_to_source.json`：否则无法识别「旧视频对应哪个序号」，新跑可能给同一批视频分配新的编号（与已有目录重复风险）；备份该文件即可安全追加

输出：
- `examples/training/CameraHMR_smpl_results/<000001>/`
  - `bbox.pt`
  - `smpl_raw.pt`
- `examples/training/CameraHMR_smpl_results_smoothed/<000001>/`
  - `motion_postprocess.pt`
  - `smpls_smoothed_group.npz`
- `examples/training/processed_trainable_data/<000001>/`
  - `rgb.mp4`
  - `first_frame.jpg`
- `examples/training/sample_id_to_source.json`
  - `items[]`：`sample_id`、`seq_index`、`original_filename`、`original_stem`、`original_path_relative`、`output_sample_dir`
- `examples/training/train_stage4_empty_text.json`
  - 额外字段：`sample_id`、`original_video`；以及 `rgb_path` / `first_frame` / `smpl_path` / `text=""` / `type` / `source` / `link`