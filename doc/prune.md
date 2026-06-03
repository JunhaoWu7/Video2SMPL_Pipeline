# prune 阶段

在 **captions 之后、video2smpl 之前** 执行，删除 `robot_learnable == false` 的样本。

## 为什么要删 `processed_trainable_data/<id>/` 整目录？

select 阶段已经把**视频文件**放进这个目录，例如：

```
processed_trainable_data/000042/
└── dance_01.mp4
```

prune 时 SMPL **还没跑**，目录里通常只有视频（没有 `smpl_canonical.npz`）。

若只删 manifest 行、不删目录：

- 磁盘上仍留着不可学习 clip 的视频，**白占空间**
- 容易和保留样本混淆，不利于后续排查

因此对 `robot_learnable=false` 的样本：**manifest / mapping 删掉 + 整目录 `shutil.rmtree`**，从数据集里彻底清掉。

保留的样本会去掉 manifest 里的 `robot_learnable` 字段（隐含均为可学习），再进入 video2smpl。

## 行为摘要

| 操作 | 说明 |
|------|------|
| manifest | 移除不可学习行 |
| mapping | 同步移除 |
| 磁盘 | 删除 `processed_trainable_data/<id>/` |
| 保留行 | 去掉 `robot_learnable` 字段 |

## 命令

```bash
python run.py --dataset humanvid --select-input-dir /path/to/videos   # 全流程含 prune

python run.py --dataset humanvid --from-stage prune
python run.py --dataset humanvid --from-stage prune --prune-dry-run
```

## 编排硬规则

同一轮若跑 **prune**，必须接着跑 **video2smpl**（`run.py` 会校验）。
