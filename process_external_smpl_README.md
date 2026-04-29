# process_external_smpl.py 使用说明

`process_external_smpl.py` 用于将“外来 SMPL”按 `run_pipeline.py` 的后处理思路做对齐：

1. 先按同链路进行插值与时序平滑（`rot6d` 路径）
2. 再走 `process_hmr_motion` 做坐标语义变换 / canonicalization（可选 `set_floor`）
3. 输出目录结构对齐到 CameraHMR 后处理产物
4. 每次运行都新建一个新的 run 文件夹，不覆盖历史结果

依赖路径约定：
- `--vendor_root` 默认是 `third_party`
- CameraHMR 权重统一读取 `third_party/extract_motion/CameraHMR/data/...`
- 若出现 `third_party/extract_motion/data/...` 报错，说明代码版本过旧，请先更新代码

---

## 输入格式要求（外来 SMPL）

支持文件后缀：`.npz` / `.pt` / `.pth`

必须包含以下字段：

- `global_orient`：每帧根节点旋转，支持轴角或旋转矩阵
- `body_pose`：每帧 23 关节旋转，支持轴角或旋转矩阵
- `transl`：每帧平移 `(T, 3)`
- `betas`：`(10,)` 或 `(T,10)`（脚本取第一帧形状）

可选字段：

- `intrinsic`：`3x3` 内参矩阵（若无则使用命令行 `--fx --fy --cx --cy`）
- `frame_mask`：帧有效性掩码（若无默认全 1）

---

## 处理前“自查”要求（必须）

脚本要求显式传 `--self_check_confirm`，否则会拒绝处理。  
这表示你已经手动确认以下事项：
- 轴系定义：确认是右手系，明确哪个轴是 up / forward / right
- 参数语义：`transl` 必须是全局根位移，不是局部偏移
- 关节拓扑：与 SMPL 约定一致（关节顺序、数量一致）
- 单位/尺度：同一批数据单位一致（通常米）
- 旋转表达无误（轴角单位为弧度；矩阵为正交旋转）
- 帧序连续且与 `frame_mask` 语义一致
- `intrinsic` 合理（若缺失，默认内参是否可接受）

建议先跑一次 `--check_only`，仅做兼容性预检与报告输出。

---

## 运行示例

### 1) 只做预检（推荐先跑）

```bash
cd /root/projects/Video2SMPL
python pipeline/process_external_smpl.py \
  --root_dir examples/training \
  --vendor_root third_party \
  --external_smpl_dir /path/to/external_smpl \
  --glob "*.npz" \
  --self_check_confirm \
  --check_only
```

### 2) 正式处理（变换 + 插值平滑 + 对齐输出）

```bash
cd /root/projects/Video2SMPL
python pipeline/process_external_smpl.py \
  --root_dir examples/training \
  --vendor_root third_party \
  --external_smpl_dir /path/to/external_smpl \
  --glob "*.npz" \
  --smooth_window 5 \
  --set_floor \
  --self_check_confirm
```

---

## 输出结构

每次运行都会创建一个新目录：

`<root_dir>/external_smpl_runs/external_smpl_run_YYYYmmdd_HHMMSS/`

其中包含：

- `precheck_report.json`：输入兼容性预检结果
- `external_smpl_mapping.json`：本次处理映射
- `CameraHMR_smpl_results/<sample_id>/smpl_raw.pt`
- `CameraHMR_smpl_results_smoothed/<sample_id>/motion_postprocess.pt`
- `CameraHMR_smpl_results_smoothed/<sample_id>/smpls_smoothed_group.npz`

> `sample_id` 默认从 `--start_id` 开始按 `--id_width` 零填充递增。

