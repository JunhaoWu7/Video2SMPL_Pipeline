# external_smpl 阶段（旁路）

将**外来 SMPL** 按与 video2smpl 相近的后处理做对齐；**不**写入主链 `dataset_manifest.json`，输出在独立 run 目录。

实现：`pipeline/stages/external_smpl/run.py`

## 命令

```bash
python run.py --root_dir examples/training --stages external_smpl \
  --external_smpl_dir /path/to/external_smpl \
  --self_check_confirm

# 仅预检
python run.py --stages external_smpl \
  --root_dir examples/training \
  --external_smpl_dir /path/to/external_smpl \
  --glob "*.npz" \
  --self_check_confirm \
  --check_only
```

## 输入格式（`.npz` / `.pt` / `.pth`）

必需：

- `global_orient`、`body_pose`、`transl`、`betas`

可选：

- `intrinsic`、`frame_mask`

处理前须传 `--self_check_confirm`（表示已人工确认轴系、单位、关节拓扑等）。建议先 `--check_only`。

## 输出

```
<root_dir>/external_smpl_runs/external_smpl_run_YYYYmmdd_HHMMSS/
├── precheck_report.json
├── external_smpl_mapping.json
└── CameraHMR_smpl_results_smoothed/<sample_id>/...
```

权重路径与 video2smpl 相同，见 [weights.md](weights.md)。
