# 端到端验证指南

从零新建示例数据集，分阶段跑通 Pipeline，并在每步后用 `verify_pilot_dataset.py` 检查状态。

Hub 默认路径：`/data1/wjh/HumanRetarget`  
代码目录：`/home/wujunhao/code/Video2SMPL_Pipeline`

---

## 0. 一次性准备

```bash
cd /home/wujunhao/code/Video2SMPL_Pipeline

# VLM API（select step3 + captions 都需要）
export TOKENROUTER_API_KEY="你的key"   # 或 OPENAI_API_KEY

# 若 API 需代理（按需）
source ~/bin/proxy-on

# Hugging Face（可选，减少 video2smpl 下载限速 warning）
# export HF_TOKEN="hf_xxxx"

# PromptHMR vendor（首次或更新后）
bash scripts/copy_prompthmr_vendor.sh
conda activate phmr_pt2.4
python -m pipeline.stages.video2smpl.prompthmr_weights
```

---

## 1. 新建示例数据集

从已有数据集（如 `charades`）随机抽样，生成干净的 pilot 数据集：

```bash
conda activate video2smpl
cd /home/wujunhao/code/Video2SMPL_Pipeline

export HUB_ROOT=/data1/wjh/HumanRetarget
export PILOT_NAME=charades_pilot_e2e   # 新数据集名，勿与已有重复

python scripts/create_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --source-dataset charades \
  --pilot-name "$PILOT_NAME" \
  --count 30 \
  --seed 42 \
  --overwrite
```

生成结果：

- `$HUB_ROOT/$PILOT_NAME/video/` — 原始 mp4
- `$HUB_ROOT/$PILOT_NAME/dataset_manifest.json` — 已初始化（阶段字段为空）
- `$HUB_ROOT/$PILOT_NAME/sample_id_to_source.json` — 预写 mapping

**重要：** `create_pilot_dataset` 会预写 mapping。第一次跑 **select 必须加 `--overwrite`**，否则会全部 `skipped (mapped)` 且 manifest 可能被清空。见阶段 1。

---

## 2. 公共变量

后续各阶段命令共用：

```bash
export HUB_ROOT=/data1/wjh/HumanRetarget
export PILOT_NAME=charades_pilot_e2e
cd /home/wujunhao/code/Video2SMPL_Pipeline
export TOKENROUTER_API_KEY="你的key"
```

验证命令（每阶段后执行）：

```bash
python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

---

## 3. 分阶段单独跑

Pipeline 顺序：

```
select → captions → prune → video2smpl → export_splits
```

各阶段可分开跑，无强制同轮绑定（例如 prune 与 video2smpl 可分开两次执行）。

### 阶段 1：select

```bash
conda activate video2smpl

python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages select \
  --overwrite

python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

**期望：** `stages.select` = N/N；视频已进入 `processed_trainable_data/<id>/`；耗时分钟级（非 0.x 秒）。

**并行（默认）：** `--select-workers 8`（step1/2/3 按视频 8 路并行，入库仍顺序）。串行调试：

```bash
python run.py ... --stages select --overwrite --select-workers 1
```

**pilot 首次 select 务必 `--overwrite`**（见第 1 节说明）。

---

### 阶段 2：captions

```bash
conda activate video2smpl

python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages captions

python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

**期望：** `stages.captions` = N/N，`caption_complete_rows` = N/N。

**并行（默认）：** `--workers 8`（8 路 VLM API 并行）。

**默认行为：** 若 VLM 重试后仍 `Invalid skill_category`，自动删除该条样本（manifest + `processed_trainable_data/<id>/`）并重编号。保留失败样本：

```bash
python run.py ... --stages captions --no-drop-invalid-skill-category
```

补跑失败 / 提高重试次数：

```bash
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages captions \
  --caption-parse-retries 4
```

强制重打标（可选）：

```bash
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages captions \
  --force-recaption
```

---

### 阶段 3：prune

可先 dry-run 预览会删哪些样本，再单独执行 prune：

```bash
conda activate video2smpl

# 预览（不写 manifest、不删目录）
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages prune \
  --prune-dry-run

# 正式 prune
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages prune

python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

**期望：** `robot_learnable=false` 的样本从 manifest 与磁盘删除；若有剔除会重编号为 `000001`..`N`。

若全部 `robot_learnable=true`，prune 不会删样本，可跳过直接进入 video2smpl。

---

### 阶段 4：video2smpl

**必须使用 `phmr_pt2.4` 环境**（`video2smpl` 环境缺 `joblib` / `smplcodec` 会失败）。

```bash
conda activate phmr_pt2.4
cd /home/wujunhao/code/Video2SMPL_Pipeline

# 默认：8 worker × 8 GPU 并行（auto 取前 8 张可见卡）
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages video2smpl

python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

**期望：** `stages.video2smpl` = N/N，`smpl_filled_rows` = N/N；每条有 `smpl_prompthmr.npz`。

**多卡加速（显式指定 8 卡，与默认 auto 等价）：**

```bash
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages video2smpl \
  --video2smpl-workers 8 \
  --video2smpl-gpus 0,1,2,3,4,5,6,7
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--video2smpl-workers` | `0`（auto） | **8** worker（与 GPU 数取 min）；`1` = 串行 |
| `--video2smpl-gpus` | `auto` | 前 **8** 张可见 CUDA 卡：`0`–`7` |

启动时会打印：`video2smpl parallel: workers=8, gpus=[0, 1, ..., 7]`。已有 `smpl_prompthmr.npz` 的样本自动跳过。

重跑已完成样本：

```bash
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages video2smpl \
  --overwrite
```

单卡串行：

```bash
python run.py ... --stages video2smpl --video2smpl-workers 1
```

---

### 阶段 5：export_splits

```bash
conda activate video2smpl

python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages export_splits

# 或等价独立脚本
python export_skill_splits.py \
  --root-dir "$HUB_ROOT/$PILOT_NAME"

python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

**期望：** `e2e_pass: true`；`splits` 4/4 文件；`skill_export_summary.json` 中 `export_ready` = 当前样本数。

---

## 4. 推荐的两段式全流程

分阶段验证通过后，也可用两段命令一次跑完（环境切换与线上一致）：

```bash
# 前半：select + captions
conda activate video2smpl
export TOKENROUTER_API_KEY="你的key"
export HUB_ROOT=/data1/wjh/HumanRetarget
export PILOT_NAME=charades_pilot_e2e
cd /home/wujunhao/code/Video2SMPL_Pipeline

python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages select,captions \
  --overwrite

# 后半：prune → video2smpl → export_splits
conda activate video2smpl
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages prune

conda activate phmr_pt2.4
python run.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME" \
  --stages video2smpl,export_splits

python scripts/verify_pilot_dataset.py \
  --hub-root "$HUB_ROOT" \
  --dataset "$PILOT_NAME"
```

---

## 5. 阶段速查表

| 阶段 | conda 环境 | 能否单独跑 | 默认并行 | 依赖 API |
|------|-----------|-----------|----------|----------|
| select | `video2smpl` | ✅ | **`--select-workers 8`** | step3 VLM |
| captions | `video2smpl` | ✅ | **`--workers 8`** | 需要 |
| prune | `video2smpl` | ✅ | — | 否 |
| video2smpl | **`phmr_pt2.4`** | ✅ | **`8 worker × 8 GPU`** | 否 |
| export_splits | `video2smpl` | ✅ | — | 否 |

---

## 6. 各阶段后检查指标

| 阶段后 | 关键指标 |
|--------|----------|
| select | `stages.select == manifest_rows` |
| captions | `caption_complete_rows == manifest_rows` |
| prune | 无 `robot_learnable=false` 残留（或已剔除） |
| video2smpl | `smpl_filled_rows == manifest_rows` |
| export_splits | `e2e_pass: true` |

`verify_pilot_dataset.py` 输出示例（全部通过时）：

```json
{
  "e2e_pass": true,
  "stages": {
    "select": 19,
    "captions": 19,
    "prune": 19,
    "video2smpl": 19
  },
  "smpl_filled_rows": 19,
  "splits": {
    "manipulation.json": true,
    "locomotion.json": true,
    "loco-manipulation.json": true,
    "skill_export_summary.json": true
  }
}
```

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| select 0.x 秒、`skipped (mapped): N` | pilot 预写 mapping，未 `--overwrite` | 加 `--overwrite` 重跑；若 manifest 已空则重建 pilot |
| captions `Invalid skill_category` | VLM 偶发漏字段 | 默认自动删样本；或 `--caption-parse-retries 4` 补跑 |
| `HF Hub unauthenticated` warning | 匿名访问 Hugging Face | 可忽略；或 `export HF_TOKEN=...` |
| video2smpl 很慢 | 曾单卡串行 | 默认已 **8 worker × 8 GPU**；机器不足 8 卡时自动用全部可见卡 |

---

## 8. 相关文档

- [doc/select.md](doc/select.md)
- [doc/captions.md](doc/captions.md)
- [doc/prune.md](doc/prune.md)
- [doc/video2smpl.md](doc/video2smpl.md)
- [doc/export_splits.md](doc/export_splits.md)
- [doc/data_layout.md](doc/data_layout.md)
