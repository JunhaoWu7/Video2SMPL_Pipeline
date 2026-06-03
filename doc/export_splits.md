# export_splits 阶段

SMPL 跑完后**自动**导出按技能分类的 manifest 列表；逻辑在独立脚本 `export_skill_splits.py`，**不**写在 `video2smpl` 里。

## 输出

```
splits/
├── manipulation.json
├── locomotion.json
├── loco-manipulation.json
└── skill_export_summary.json
```

每条为 train-ready 行（caption + `smpl_path` + `skill_category`）。

## 命令

```bash
# 全流程（默认含本阶段）
python run.py --dataset humanvid --select-input-dir /path/to/videos

# 仅从 SMPL 之后续跑（含 export_splits）
python run.py --dataset humanvid --from-stage video2smpl

# 单独脚本（与 pipeline 阶段等价）
python export_skill_splits.py --root-dir /data1/wjh/HumanRetarget/humanvid
```

## 说明

默认全流程在 `video2smpl` 之后运行本阶段；也可单独 `--from-stage export_splits` 或调用 `export_skill_splits.py`。
