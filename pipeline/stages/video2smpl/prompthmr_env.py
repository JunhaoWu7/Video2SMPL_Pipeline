"""Prepare sys.path + cwd for vendored PromptHMR (package ``phmr_pipeline``)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

# Minimal imports needed before ``phmr_pipeline.pipeline`` loads.
_PROMPTHMR_RUNTIME_MODULES = ("joblib", "smplcodec")

from pipeline.stages.video2smpl.prompthmr_paths import (
    PHMR_PKG,
    apply_absolute_path_patches,
)

VENDOR_BUNDLE_DIR = Path(__file__).resolve().parent / "vendor_bundle"


def vendor_bundle_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("VIDEO2SMPL_PROMPTHMR_VENDOR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return VENDOR_BUNDLE_DIR.resolve()


def verify_prompthmr_runtime(
    *,
    vendor_root: Optional[str] = None,
    ckpt_root: Optional[str] = None,
) -> None:
    """
    Fail fast when the active conda env lacks PromptHMR runtime deps.

    Recommended env: ``phmr_pt2.4`` (see doc/prompthmr_vendor.md).
    """
    missing = [
        name for name in _PROMPTHMR_RUNTIME_MODULES if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "PromptHMR backend missing Python package(s): "
            + ", ".join(missing)
            + "\nSwitch to the PromptHMR env before video2smpl:\n"
            + "  conda activate phmr_pt2.4\n"
            + "  python run.py ... --stages video2smpl\n"
            + "Or install deps into the current env (may need more than joblib/smplcodec)."
        )
    setup_prompthmr_runtime(vendor_root=vendor_root, ckpt_root=ckpt_root)
    import phmr_pipeline.pipeline  # noqa: F401


def setup_prompthmr_runtime(
    *,
    vendor_root: Optional[str] = None,
    ckpt_root: Optional[str] = None,
) -> Path:
    """
    Add vendor_bundle to sys.path; chdir there for ``data/`` and config paths.

    Uses ``phmr_pipeline`` package name so it does not clash with Video2SMPL ``pipeline``.
    """
    root = vendor_bundle_dir(vendor_root)
    phmr_root = root / PHMR_PKG
    if not phmr_root.is_dir():
        raise FileNotFoundError(
            f"Missing {phmr_root}. Run: bash scripts/copy_prompthmr_vendor.sh"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    gvhmr = phmr_root / "gvhmr"
    if gvhmr.is_dir():
        gvhmr_str = str(gvhmr)
        if gvhmr_str not in sys.path:
            sys.path.insert(0, gvhmr_str)
    os.chdir(root)
    os.environ["VIDEO2SMPL_SKIP_MCS_EXPORT"] = "1"
    apply_absolute_path_patches(ckpt_root=ckpt_root, vendor_root=root)
    return root


def apply_caption_text_patch(text_prompt: str, primary_track_id: Optional[int] = None) -> None:
    """Attach manifest caption strings to PHMR batch ``text`` field."""
    import phmr_pipeline.phmr_vid as pmv

    prompt = (text_prompt or "").strip()
    primary_tid = primary_track_id
    OriginalDataset = pmv.PromptHMRVideoDataset

    class CaptionPromptHMRVideoDataset(OriginalDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            track_ids = item.get("track_ids") or []
            texts: List[str] = []
            for tid in track_ids:
                if prompt and (primary_tid is None or tid == primary_tid):
                    texts.append(prompt)
                else:
                    texts.append("NULL")
            if texts:
                item["text"] = texts
            return item

    pmv.PromptHMRVideoDataset = CaptionPromptHMRVideoDataset
