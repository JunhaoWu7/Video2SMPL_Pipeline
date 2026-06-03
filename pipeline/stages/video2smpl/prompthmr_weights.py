"""Check PromptHMR vendor_bundle + checkpoint layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

DEFAULT_CKPT_ROOT = Path("/data1/wjh/ckpt/PromptHMR")
DEFAULT_SMPL_CKPT = Path("/data1/wjh/smpl_ckpt")
DEFAULT_VENDOR = Path(__file__).resolve().parent / "vendor_bundle"


def _vendor_root(vendor_root: str | None) -> Path:
    if vendor_root:
        return Path(vendor_root).expanduser().resolve()
    return DEFAULT_VENDOR.resolve()


def required_weight_paths(ckpt: Path, smpl_ckpt: Path) -> List[Tuple[str, Path, bool]]:
    """Return (label, path, required) tuples — all paths under CKPT_ROOT / smpl_ckpt."""
    pre = ckpt / "pretrain"
    bm = ckpt / "body_models"
    return [
        ("phmr checkpoint", pre / "phmr/checkpoint.ckpt", True),
        ("phmr_vid ckpt", pre / "phmr_vid/prhmr_release_002.ckpt", True),
        ("phmr_vid yaml", pre / "phmr_vid/prhmr_release_002.yaml", True),
        ("sam vit-h", pre / "sam_vit_h_4b8939.pth", True),
        ("vitpose-h", pre / "vitpose-h-coco_25.pth", True),
        ("sam2 tiny", pre / "sam2_ckpts/sam2_hiera_tiny.pt", True),
        ("detectron2 kprcnn", pre / "sam2_ckpts/keypoint_rcnn_5ad38f.pkl", True),
        ("camcalib spec", pre / "camcalib_sa_biased_l2.ckpt", True),
        ("droidcalib slam", pre / "droidcalib.pth", False),
        ("smplx slim npz", bm / "smplx/SMPLX_neutral_array_f32_slim.npz", True),
        ("smplx2smpl", bm / "smplx2smpl.pkl", True),
        ("SMPLX_NEUTRAL", smpl_ckpt / "smplx/SMPLX_NEUTRAL.npz", True),
        ("yolo11x (ByteTrack only)", pre / "yolo11x.pt", False),
        ("droid.pth (unused by SLAM)", pre / "droid.pth", False),
    ]


def check_weights(
    vendor_root: str | None = None,
    ckpt_root: str | None = None,
    smpl_ckpt_root: str | None = None,
    *,
    require_slam: bool = False,
) -> Tuple[bool, List[str]]:
    vend = _vendor_root(vendor_root)
    if (vend / "phmr_pipeline").exists():
        print(f"vendor code: {vend} (phmr_pipeline)")
    elif (vend / "pipeline").exists():
        print(f"WARN: vendor still has legacy pipeline/ — run: bash scripts/copy_prompthmr_vendor.sh")
    ckpt = Path(ckpt_root or DEFAULT_CKPT_ROOT).expanduser().resolve()
    smpl_ckpt = Path(smpl_ckpt_root or DEFAULT_SMPL_CKPT).expanduser().resolve()
    print(f"ckpt root: {ckpt}")
    missing: List[str] = []
    warnings: List[str] = []
    for label, path, required in required_weight_paths(ckpt, smpl_ckpt):
        if "droidcalib" in label:
            required = require_slam
        ok = path.is_file() and path.stat().st_size > 0
        if not ok and not required:
            warnings.append(f"WARN (optional) {label}: {path}")
            continue
        if not ok:
            missing.append(f"MISS {label}: {path}")
        else:
            print(f"OK   {label}: {path}")
    for w in warnings:
        print(w)
    return len(missing) == 0, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PromptHMR vendor weights")
    parser.add_argument("--vendor-root", type=str, default=None)
    parser.add_argument("--ckpt-root", type=str, default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--smpl-ckpt-root", type=str, default=str(DEFAULT_SMPL_CKPT))
    parser.add_argument(
        "--require-slam",
        action="store_true",
        help="Treat droidcalib.pth as required (moving camera / --no-static-camera).",
    )
    args = parser.parse_args()
    ok, missing = check_weights(
        args.vendor_root,
        args.ckpt_root,
        args.smpl_ckpt_root,
        require_slam=args.require_slam,
    )
    if missing:
        print("\nMissing required files:")
        for m in missing:
            print(m)
        return 1
    if not ok:
        return 1
    print("\nRequired weights: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
