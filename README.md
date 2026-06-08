# Video2SMPL Pipeline

多阶段流水线：视频入库 → 大模型打标 → 视频转 SMPL（默认 **PromptHMR** 世界系，可选 **CameraHMR** DART），统一写入 `dataset_manifest.json`。

**入口**：`run.py`

## 快速开始

```bash
# Hub（默认 /data1/wjh/HumanRetarget/<dataset>/video/ 放原始视频）
python run.py --init-dataset humanvid
python run.py --list-datasets
# 视频放入 /data1/wjh/HumanRetarget/humanvid/video/ 后：
python run.py --dataset humanvid --from-stage select

# 从打标或 SMPL 续跑
python run.py --dataset humanvid --from-stage captions
# SMPL（默认 prompthmr；需先 bash scripts/copy_prompthmr_vendor.sh）
python run.py --dataset humanvid --from-stage video2smpl
python run.py --dataset humanvid --from-stage video2smpl --hmr-backend camerahmr

# 单目录开发（默认读 examples/training/video/）
python run.py --root_dir examples/training --source my_data --from-stage select
```

`--source`：manifest 中的来源标签；Hub 下默认与 `--dataset` 同名。详见 [doc/data_layout.md](doc/data_layout.md)。

## 阶段顺序

固定顺序：`select` → `captions` → `prune` → `video2smpl` → `export_splits`（`--stages` 会自动重排）。

**编排硬规则（仅此一条）**：同一轮若跑 **prune**，必须接着跑 **video2smpl**。

建议顺序仍为全流程；`export_splits`、各阶段衔接等见 `doc/`，未在代码里强制。

| 阶段 | 说明文档 |
|------|----------|
| select | [doc/select.md](doc/select.md) |
| captions | [doc/captions.md](doc/captions.md) |
| prune | [doc/prune.md](doc/prune.md) |
| video2smpl | [doc/video2smpl.md](doc/video2smpl.md) |
| export_splits | [doc/export_splits.md](doc/export_splits.md) |
| external_smpl（旁路） | [doc/external_smpl.md](doc/external_smpl.md) |

## 文档索引

| 文档 | 内容 |
|------|------|
| [doc/data_layout.md](doc/data_layout.md) | 目录树、manifest、`source`、导出 splits |
| [doc/weights.md](doc/weights.md) | CameraHMR 权重下载与自检 |
| [doc/select.md](doc/select.md) | 视频入库 |
| [doc/captions.md](doc/captions.md) | 打标 |
| [doc/video2smpl.md](doc/video2smpl.md) | SMPL 提取 |
| [doc/prompthmr_vendor.md](doc/prompthmr_vendor.md) | PromptHMR 内嵌代码与权重 |
| [doc/external_smpl.md](doc/external_smpl.md) | 外来 SMPL 对齐 |

## 仓库结构

```
Video2SMPL_Pipeline/
├── run.py                          # 顶层编排
├── generate_sequence_captions.py   # captions 实现
├── export_train_splits.py          # 导出 train/val/test JSON
├── export_skill_splits.py          # export_splits 阶段调用的实现（勿并入 video2smpl）
├── requirements.txt
├── doc/                            # 各阶段与数据说明
├── pipeline/
│   ├── manifest.py, dataset_schema.py, hub.py
│   └── stages/
│       ├── select/
│       ├── captions/
│       ├── prune/
│       ├── video2smpl/             # backends + vendor_bundle/（PromptHMR 拷贝）
│       ├── export_splits/
│       └── external_smpl/
└── third_party/                    # CameraHMR / EchoMotion  vendored
```

## 其他命令

```bash
python run.py --list-stages
python run.py --dataset humanvid --from-stage export_splits
python export_train_splits.py --root-dir /data1/wjh/HumanRetarget/humanvid
```

**video2smpl**：默认 PromptHMR → `bash scripts/copy_prompthmr_vendor.sh`（代码进 `vendor_bundle/`，权重绝对路径 `/data1/wjh/ckpt/PromptHMR`）。CameraHMR 见 [doc/weights.md](doc/weights.md)。详见 [doc/video2smpl.md](doc/video2smpl.md)、[doc/prompthmr_vendor.md](doc/prompthmr_vendor.md)。






