# PromptHMR vendor bundle (copied, not symlinked)

Copied from `/home/wujunhao/code/PromptHMR` by `scripts/copy_prompthmr_vendor.sh`.

- `pipeline/` — video world-coordinate Pipeline (SAM2, ViTPose, PHMR, SLAM, …)
- `prompt_hmr/` — PHMR model code

**Weights** are NOT stored here. Runtime loads from absolute paths:

```
/data1/wjh/ckpt/PromptHMR
```

(set via `--prompthmr-ckpt-root` or `VIDEO2SMPL_PROMPTHMR_CKPT`)

Re-copy after upgrading PromptHMR upstream:

```bash
bash scripts/copy_prompthmr_vendor.sh
```
