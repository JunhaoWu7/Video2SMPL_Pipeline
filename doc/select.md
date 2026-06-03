# select 阶段

将外部目录中的视频登记进 `dataset_manifest.json`，并**移动**（或硬链）到 `processed_trainable_data/<sample_id>/`。

实现：`pipeline/stages/select/run.py`

## 命令

```bash
python run.py --dataset humanvid \
  --select-input-dir /path/to/raw_videos

# 硬链而非移动（源目录需保留时）
python run.py --dataset humanvid \
  --select-input-dir /path/to/raw_videos \
  --select-symlink

# 仅 select
python run.py --root_dir examples/training --source my_data \
  --from-stage select \
  --select-input-dir /path/to/videos
```

## 参数

| 参数 | 说明 |
|------|------|
| `--select-input-dir` | **必填**。递归扫描 `mp4/mov/avi/mkv` |
| `--source` | **必填**。写入 manifest 的 `source` 标签 |
| `--id_width` | 默认 `6` → `000001` |
| `--overwrite` | 已映射源路径可重新入库 |
| `--select-symlink` | 用 symlink 代替 move |

## 行为

- 新视频：从当前最大 `sample_id + 1` 编号。
- 已出现在 `sample_id_to_source.json` 的 `original_path_relative` 默认跳过。
- 输出：`processed_trainable_data/<id>/<原名>.mp4`，manifest 中 `video_path` / `rgb_path` 指向该路径。
