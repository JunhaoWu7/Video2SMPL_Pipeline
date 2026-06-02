# 统一数据集结构（Hub + 训练精简布局）

每个样本一个目录：`processed_trainable_data/<sample_id>/` 下放**视频**、`first_frame.jpg`、`smpl_canonical.npz`。

**select** 从外部目录（`--select-input-dir`）把视频 **移动** 到对应样本目录（可用 `--select-symlink` 改为硬链）。

## 每个数据集子目录（跑完后）

```
<dataset_root>/
├── dataset_manifest.json
├── sample_id_to_source.json
├── splits/                         # 预留
└── processed_trainable_data/
    └── 000001/
        ├── dance_01.mp4            # 保留原文件名
        ├── first_frame.jpg
        └── smpl_canonical.npz
```

## Manifest 示例

```json
{
  "sample_id": "000001",
  "source": "humanvid",
  "video_path": "processed_trainable_data/000001/dance_01.mp4",
  "rgb_path": "processed_trainable_data/000001/dance_01.mp4",
  "first_frame": "processed_trainable_data/000001/first_frame.jpg",
  "smpl_path": "processed_trainable_data/000001/smpl_canonical.npz",
  "caption": "...",
  "action_caption": "..."
}
```

训练读视频：`rgb_path`。读动作：`smpl_path`。

## 常用命令

```bash
python run.py --init-dataset humanvid
python run.py --dataset humanvid --select-input-dir /data/.../raw_clips
python run.py --dataset humanvid --from-stage captions   # 跳过 select，无需 --select-input-dir
```

仅追加新视频时，再次指定同一或新的 `--select-input-dir`；已出现在 `sample_id_to_source.json` 的源路径会跳过。
