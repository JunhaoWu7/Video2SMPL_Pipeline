# captions 阶段

从**动作 clip** 均匀抽帧，调用 VLM，**原地**更新 `dataset_manifest.json`。

## 写入字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `caption` | string | **一句**自然语言，描述该 clip **最显著的一个**动作（如「一个人在拿起一个水杯」）；不要多动作/先后叙述 |
| `action_caption` | string | 与 `caption` 同一动作的简短短语 |
| `robot_learnable` | boolean | 该动作是否适合机器人学习模仿 |
| `skill_category` | string | **唯一**类别：`manipulation` \| `locomotion` \| `loco-manipulation` |

### `skill_category` 定义

- **manipulation**：以上肢/手–物体交互为主，基座基本不动  
- **locomotion**：以全身位移为主（走、跑、转身、上下楼梯等），几乎无物体操作  
- **loco-manipulation**：位移与操作**明显结合**（边走边搬、边走近边抓取等）

### `robot_learnable`

- `true`：人形/移动操作机器人**有可能**从该 clip 学习并复现主技能  
- `false`：纯站立发呆、对话、镜头运动、严重遮挡、多人混乱、超出典型机器人能力等  

跳过条件（默认）：上述四字段均已填满。

**部分补全（默认）**：若 manifest 中已有部分字段（如 Charades 预填的 `caption` / `action_caption`），captions 阶段会将其作为提示词上下文，**仅调用 API 补全留空字段**，不覆盖已填值。四字段全空时则全量打标。用 `--force-recaption` 可强制重打并覆盖全部四字段。

## 依赖

```bash
pip install openai pillow httpx
export TOKENROUTER_API_KEY=...   # 或 OPENAI_API_KEY（默认走 TokenRouter https://api.tokenrouter.com/v1）
```

## 建议顺序

`captions` → `prune` → `video2smpl` → `export_splits`（默认全流程已包含）。  
编排上仅强制：**prune 后必须跑 video2smpl**（同一轮）。

## 命令

```bash
# 从打标开始（自动包含 video2smpl）
python run.py --dataset humanvid --from-stage captions

python generate_sequence_captions.py \
  --manifest /data1/wjh/HumanRetarget/humanvid/dataset_manifest.json \
  --pipeline-root /data1/wjh/HumanRetarget/humanvid \
  --model google/gemini-3.1-flash-image-preview \
  --num-frames 12 \
  --force-recaption   # 旧数据补 robot/skill 字段时
```

## Manifest 示例

```json
{
  "sample_id": "000001",
  "caption": "A person walks forward while holding a box.",
  "action_caption": "walking with box",
  "robot_learnable": true,
  "skill_category": "loco-manipulation"
}
```

## 常用选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `--caption-lang` | `en` | `caption` / `action_caption` 语言（`en` / `zh` / `bilingual`） |
| `--model` | `google/gemini-3.1-flash-image-preview` | TokenRouter 视觉打标模型 |
| `--base-url` | `https://api.tokenrouter.com/v1` | TokenRouter OpenAI 兼容端点 |
| `--workers` | 4 | 并行数 |
| `--num-frames` | 16 | 均匀抽帧数 |
| `--force-recaption` | off | 强制重打全部 caption 阶段字段 |
