# select 阶段

将 ``<dataset_root>/video/`` 中的视频经 **step1 → step2 → step3** 预筛后，登记进 `dataset_manifest.json`，并 move/symlink 到 `processed_trainable_data/<sample_id>/`。

实现：`pipeline/stages/select/run.py`、`pipeline/stages/select/filters/`

## 流程

1. **Step1** 基础预检（OpenCV）：时长、分辨率、保守静止检测  
2. **Step2** YOLO 粗检：无人 / 多人 / 人太小 / 第三人称启发式  
3. **Step3** 轻量 VLM 细检：**step1/2 的 `passed` 与 `deferred` 均执行**；判视角 + 动作存在性  
4. `rejected` → **直接舍弃**（不移动、不写 manifest）  
5. Step3 `passed` → 入库；`stages_completed` 写入 `["select"]`（select 阶段完结）

## 目录约定（Hub）

```
/data1/wjh/HumanRetarget/<dataset_name>/video/   # 待处理原始视频
```

## 命令

```bash
export OPENROUTER_API_KEY=...   # 或 OPENAI_API_KEY

python run.py --init-dataset humanvid
# 视频放入 /data1/wjh/HumanRetarget/humanvid/video/

python run.py --dataset humanvid --from-stage select

# 仅跳过 step1/2（仍跑 step3）
python run.py --dataset humanvid --from-stage select --select-skip-filters

# 开发：跳过 step3（select 不会 mark complete）
python run.py --dataset humanvid --from-stage select --select-skip-vlm
```

## 权重与 API

| 步骤 | 依赖 |
|------|------|
| Step2 YOLO | 默认 `/data1/wjh/ckpt/PromptHMR/pretrain/yolo11x.pt`；可覆盖 `--select-yolo-model` 或 `VIDEO2SMPL_SELECT_YOLO` |
| Step3 VLM | `OPENROUTER_API_KEY` / `OPENAI_API_KEY`；默认模型 `google/gemini-2.5-flash-lite`；base URL 同 captions（`http://47.94.22.126/v1`） |

自检：

```bash
python scripts/test_select_step3.py
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--select-input-dir` | `<root_dir>/video` | 递归扫描 `mp4/mov/avi/mkv` |
| `--select-skip-filters` | off | 跳过 step1/step2 |
| `--select-skip-vlm` | off | 跳过 step3 |
| `--select-yolo-model` | `/data1/wjh/ckpt/PromptHMR/pretrain/yolo11x.pt` | step2 YOLO |
| `--select-vlm-model` | `google/gemini-2.5-flash-lite` | step3 VLM |
| `--select-vlm-frames` | `6` | step3 抽帧数 |
| `--select-vlm-vision-detail` | `low` | 省 token |
| `--select-vlm-base-url` | captions 同款 | OpenAI 兼容 API |

## Manifest 写入（select 完成后）

| 字段 | 值 |
|------|-----|
| `video_path` | `processed_trainable_data/<id>/<file>.mp4` |
| `rgb_path` | `""` |
| `select_status` | `"passed"` |
| `select_notes` | `""` |
| `stages_completed` | `["select"]` |
