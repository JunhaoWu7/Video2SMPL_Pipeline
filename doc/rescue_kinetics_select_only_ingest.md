# kinetics select 额度挽救：只入库（不续跑 select/VLM/YOLO）

目标：把旧进程内存里已经算完的 `select` 结果（passed/rejected）导出成 checkpoint，然后**只把 passed 的样本入库**到：
* `processed_trainable_data/<sample_id>/...`
* `dataset_manifest.json`（updated in place）
* `sample_id_to_source.json`（updated in place）

关键点：此流程**不再调用 VLM/YOLO**，因此不会继续消耗额度；同时也避免“续跑 select 会从头再花钱”。

适用前提：你能在另一台设备上用 `root/sudo` attach 到当前正在跑 legacy `select` 的旧进程，并能从内存导出 `select_filter_checkpoint.json`。

---

## 0. 路径（按需替换）

* 仓库（代码）：`/home/wujunhao/code/Video2SMPL_Pipeline`
* hub 根：`/data1/wjh/HumanRetarget`
* 数据集：`kinetics_700_2020`
* 数据集根：`/data1/wjh/HumanRetarget/kinetics_700_2020`
* raw 视频目录：`/data1/wjh/HumanRetarget/kinetics_700_2020/video`

---

## 1) 找到旧进程 PID（legacy pipeline 的 python run.py）

```bash
PID=$(pgrep -af 'python run.py.*--dataset kinetics_700_2020' | head -1 | awk '{print $1}')
echo "PID=$PID"
ps -p "$PID" -o pid,stat,wchan,cmd
```

确认 `PID` 不为空，并且 cmd 行包含 `--dataset kinetics_700_2020`。

---

## 2) 冻结旧进程（暂停消耗额度）

```bash
sudo kill -STOP "$PID"
sleep 2
ps -p "$PID" -o pid,stat,wchan,cmd
```

如果看到 `STAT` 类似 `Tl+` 或 `T`，表示冻结成功。

---

## 3) 从内存导出 select checkpoint（生成 passed/rejected）

```bash
sudo bash /home/wujunhao/code/Video2SMPL_Pipeline/scripts/emergency_dump_legacy_select_gdb.sh \
  "$PID" \
  /data1/wjh/HumanRetarget/kinetics_700_2020
```

导出成功后应生成（通常在）：

* `/data1/wjh/HumanRetarget/kinetics_700_2020/logs/select_filter_checkpoint.json`

检查文件存在：

```bash
ls -la /data1/wjh/HumanRetarget/kinetics_700_2020/logs/select_filter_checkpoint.json*
```

---

## 4) 结束旧进程（避免干扰 + 释放资源）

```bash
sudo kill -9 "$PID"
```

---

## 5) 只入库：checkpoint -> processed_trainable_data + manifest + mapping

> 下面这一步不会调用 VLM/YOLO，只做 `passed` 的落盘与 manifest 更新。

```bash
conda activate video2smpl
cd /home/wujunhao/code/Video2SMPL_Pipeline

python scripts/rescue_ingest_from_select_checkpoint.py \
  --hub-root /data1/wjh/HumanRetarget \
  --dataset kinetics_700_2020 \
  --source Kinetics_700-2020 \
  --checkpoint /data1/wjh/HumanRetarget/kinetics_700_2020/logs/select_filter_checkpoint.json \
  --select-symlink \
  --manifest-name dataset_manifest.json \
  --mapping-name sample_id_to_source.json \
  --id-width 6
```

如果你希望覆盖目标冲突目录/文件（慎用），再加 `--overwrite`。

---

## 6) 验证是否挽救成功（重点检查 select_status=passed）

```bash
python3 - <<'PY'
import json
p="/data1/wjh/HumanRetarget/kinetics_700_2020/dataset_manifest.json"
data=json.load(open(p))
passed=sum(1 for r in data if str(r.get("select_status","")).strip()=="passed")
print("manifest rows:", len(data))
print("select_status=passed:", passed)
PY

ls -la /data1/wjh/HumanRetarget/kinetics_700_2020/processed_trainable_data | head
```

---

## 7) 若 gdb 导出失败

把失败的最后 30 行输出贴出来，常见原因包括：
* ptrace 权限限制（`ptrace_scope`）
* 脚本找不到预期的 `run()` 栈变量

可先查看：

```bash
cat /proc/sys/kernel/yama/ptrace_scope
```

