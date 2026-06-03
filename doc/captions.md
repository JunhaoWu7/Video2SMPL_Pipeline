# captions 阶段

从**动作 clip** 均匀抽帧，调用 VLM，**原地**更新 `dataset_manifest.json`。

## 写入字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `caption` | string | 1–2 句场景/动作描述 |
| `action_caption` | string | 简短动作短语 |
| `robot_learnable` | boolean | 该动作是否适合机器人学习模仿 |
| `skill_category` | string | **唯一**类别：`manipulation` \| `locomotion` \| `loco-manipulation` |

### `skill_category` 定义

- **manipulation**：以上肢/手–物体交互为主，基座基本不动  
- **locomotion**：以全身位移为主（走、跑、转身、上下楼梯等），几乎无物体操作  
- **loco-manipulation**：位移与操作**明显结合**（边走边搬、边走近边抓取等）

### `robot_learnable`

- `true`：人形/移动操作机器人**有可能**从该 clip 学习并复现主技能  
- `false`：纯站立发呆、对话、镜头运动、严重遮挡、多人混乱、超出典型机器人能力等  

跳过条件（默认）：上述四字段均已填满；用 `--force-recaption` 强制重打。

## 依赖

```bash
pip install openai pillow httpx
export OPENAI_API_KEY=...   # 或 OPENROUTER_API_KEY
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
  --model openai/gpt-4o \
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
| `--workers` | 4 | 并行数 |
| `--num-frames` | 16 | 均匀抽帧数 |
| `--force-recaption` | off | 强制重打全部 caption 阶段字段 |
