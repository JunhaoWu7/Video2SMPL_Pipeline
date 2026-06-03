"""Checkpoint layout for vendored PromptHMR (package ``phmr_pipeline``, not ``pipeline``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

DEFAULT_CKPT_ROOT = Path("/data1/wjh/ckpt/PromptHMR")
DEFAULT_SMPL_CKPT = Path("/data1/wjh/smpl_ckpt")
PHMR_PKG = "phmr_pipeline"


def resolve_ckpt_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("VIDEO2SMPL_PROMPTHMR_CKPT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CKPT_ROOT.resolve()


def resolve_smpl_ckpt_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return DEFAULT_SMPL_CKPT.resolve()


def _symlink_force(link_path: Path, target: Path) -> None:
    target = target.resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == target:
            return
        link_path.unlink()
    os.symlink(target, link_path)


def ensure_vendor_data_tree(
    vendor_root: Union[str, Path],
    ckpt_root: Optional[str] = None,
    smpl_ckpt_root: Optional[str] = None,
) -> Path:
    """``vendor_bundle/data/`` -> CKPT_ROOT (symlinks only)."""
    vendor = Path(vendor_root).resolve()
    ckpt = resolve_ckpt_root(ckpt_root)
    smpl_ckpt = resolve_smpl_ckpt_root(smpl_ckpt_root)
    data = vendor / "data"
    _symlink_force(data / "pretrain", ckpt / "pretrain")
    _symlink_force(data / "body_models", ckpt / "body_models")
    neutral = smpl_ckpt / "smplx" / "SMPLX_NEUTRAL.npz"
    neutral_dst = data / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if neutral.is_file() and not neutral_dst.exists():
        neutral_dst.parent.mkdir(parents=True, exist_ok=True)
        _symlink_force(neutral_dst, neutral)
    return ckpt


def smplx_neutral_npz(ckpt_root: Path, smpl_ckpt: Path) -> Path:
    candidates = [
        smpl_ckpt / "smplx" / "SMPLX_NEUTRAL.npz",
        ckpt_root / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "SMPLX_NEUTRAL.npz not found under smpl_ckpt or ckpt body_models/smplx"
    )


def apply_absolute_path_patches(
    ckpt_root: Optional[str] = None,
    smpl_ckpt_root: Optional[str] = None,
    vendor_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Patch ``phmr_pipeline`` to load weights from CKPT_ROOT."""
    vendor = Path(vendor_root).resolve() if vendor_root else None
    if vendor is not None:
        ckpt = ensure_vendor_data_tree(vendor, ckpt_root, smpl_ckpt_root)
    else:
        ckpt = resolve_ckpt_root(ckpt_root)
    smpl_ckpt = resolve_smpl_ckpt_root(smpl_ckpt_root)
    pre = ckpt / "pretrain"
    bm = ckpt / "body_models"
    os.environ["VIDEO2SMPL_PROMPTHMR_CKPT"] = str(ckpt)

    import phmr_pipeline.pipeline as pl_mod
    from smplx import SMPLX

    smplx_npz = str(smplx_neutral_npz(ckpt, smpl_ckpt))
    cfg_path = f"{PHMR_PKG}/config.yaml"

    def _pipeline_init(self, static_cam: bool = False) -> None:
        from omegaconf import OmegaConf

        self.images = None
        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.static_cam = static_cam
        self.data_dict = {
            "droid": str(pre / "droid.pth"),
            "sam": str(pre / "sam_vit_h_4b8939.pth"),
            "sam2": str(pre / "sam2_ckpts"),
            "yolo": str(pre / "yolo11x.pt"),
            "vitpose": str(pre / "vitpose-h-coco_25.pth"),
        }
        self.smplx = SMPLX(
            smplx_npz,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        )

    pl_mod.Pipeline.__init__ = _pipeline_init  # type: ignore[method-assign]

    import phmr_pipeline.phmr_vid as pmv
    from omegaconf import OmegaConf
    from phmr_pipeline.gvhmr.hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

    phmr_ckpt = str(pre / "phmr" / "checkpoint.ckpt")
    phmr_vid_yaml = str(pre / "phmr_vid" / "prhmr_release_002.yaml")
    phmr_vid_ckpt = str(pre / "phmr_vid" / "prhmr_release_002.ckpt")
    slim_npz = bm / "smplx" / "SMPLX_neutral_array_f32_slim.npz"
    if not slim_npz.is_file():
        slim_npz = Path(smplx_npz)

    def _load_video_head():
        phmr_vid_cfg = OmegaConf.load(phmr_vid_yaml)
        vid_head = DemoPL(
            pipeline=phmr_vid_cfg.model.pipeline,
            smplx_path=str(slim_npz),
        )
        vid_head = vid_head.eval().cuda()
        vid_head.load_pretrained_model(phmr_vid_ckpt)
        return vid_head

    pmv.load_video_head = _load_video_head

    def _phmr_video_init(self) -> None:
        from prompt_hmr import load_model as load_phmr

        self.model = load_phmr(phmr_ckpt)
        self.vid_head = _load_video_head()

    pmv.PromptHMR_Video.__init__ = _phmr_video_init  # type: ignore[method-assign]

    import phmr_pipeline.camera.masked_droid_slam as mds

    mds.slam_args.weights = str(pre / "droidcalib.pth")  # type: ignore[attr-defined]

    return ckpt
