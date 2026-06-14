# 数据集目录与 Manifest

Hub 根目录默认：`/data1/wjh/HumanRetarget`。每个数据集子目录结构相同。

- **待处理原始视频**：放在 `<dataset_root>/video/`（select 默认读此目录）
- **入库后样本**：select 通过 step1/step2 后 move 到 `processed_trainable_data/<sample_id>/`

## 目录树（跑完后）

```
/data1/wjh/HumanRetarget/<dataset_name>/
├── hub.json
├── dataset_manifest.json
├── sample_id_to_source.json
├── video/                          # 原始视频放这里（select 输入）
├── splits/
│   ├── manipulation.json           # export_skill_splits.py（SMPL 后）
│   ├── locomotion.json
│   ├── loco-manipulation.json
│   └── skill_export_summary.json
└── processed_trainable_data/
    └── 000001/
        ├── dance_01.mp4
        ├── first_frame.jpg
        ├── smpl_prompthmr.npz      # 默认 --hmr-backend prompthmr
        └── smpl_canonical.npz      # 可选 --hmr-backend camerahmr
```

## Manifest 字段

| 阶段 | 写入字段 |
|------|----------|
| select (step1/2/3) | `video_path`, `select_status=passed`；`rgb_path`/`select_notes` 留空；`stages_completed` 含 `select` |
| captions | `caption`, `action_caption`, `robot_learnable`, `skill_category` |
| prune | 删除 `robot_learnable=false` 样本；保留行保留 `robot_learnable` |
| video2smpl | `first_frame`, `smpl_path`, `smpl_backend`（`prompthmr` / `camerahmr`） |

`skill_category` 取值（每条 clip **仅一类**）：`manipulation`、`locomotion`、`loco-manipulation`。

示例：

```json
{
  "sample_id": "000001",
  "source": "humanvid",
  "video_path": "processed_trainable_data/000001/dance_01.mp4",
  "rgb_path": "processed_trainable_data/000001/dance_01.mp4",
  "first_frame": "processed_trainable_data/000001/first_frame.jpg",
  "smpl_path": "processed_trainable_data/000001/smpl_prompthmr.npz",
  "smpl_backend": "prompthmr",
  "caption": "...",
  "action_caption": "...",
  "robot_learnable": true,
  "skill_category": "loco-manipulation"
}
```

## `source` 字段

- 非空字符串，标明数据来源（数据集/批次名）。
- Hub 模式：`python run.py --dataset humanvid` 时默认 `source=humanvid`。
- 也可用 `--source` 显式覆盖。
- 导出训练集：`export_train_splits.py --source-filter humanvid,sports`。

## 导出 splits

**按技能类别**（SMPL 完成后，独立脚本）：

```bash
python export_skill_splits.py --root-dir /data1/wjh/HumanRetarget/humanvid
```

**按 train/val/test 随机划分**（可选）：

```bash
python export_train_splits.py --root-dir /data1/wjh/HumanRetarget/humanvid --require-captions
```
