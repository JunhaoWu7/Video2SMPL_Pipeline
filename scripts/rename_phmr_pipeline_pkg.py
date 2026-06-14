#!/usr/bin/env python3
"""Rename vendored PromptHMR ``pipeline`` package to ``phmr_pipeline`` (avoid clash with Video2SMPL)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "pipeline/stages/video2smpl/vendor_bundle"
OLD = VENDOR / "pipeline"
NEW = VENDOR / "phmr_pipeline"

# Order matters: longer patterns first
REPLACEMENTS = [
    ("from pipeline.", "from phmr_pipeline."),
    ("import pipeline.", "import phmr_pipeline."),
    ("sys.path.insert(0, 'pipeline/", "sys.path.insert(0, 'phmr_pipeline/"),
    ('sys.path.insert(0, "pipeline/', 'sys.path.insert(0, "phmr_pipeline/'),
    ("'pipeline/", "'phmr_pipeline/"),
    ('"pipeline/', '"phmr_pipeline/'),
    ("OmegaConf.load(\"pipeline/", 'OmegaConf.load("phmr_pipeline/'),
    ("OmegaConf.load('pipeline/", "OmegaConf.load('phmr_pipeline/"),
    ("np.loadtxt('pipeline/", "np.loadtxt('phmr_pipeline/"),
    ("_target_=pipeline.", "_target_=phmr_pipeline."),
]


def patch_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    orig = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    if not OLD.is_dir() and NEW.is_dir():
        print(f"Already renamed: {NEW}")
        return
    if not OLD.is_dir():
        raise SystemExit(f"Missing {OLD}; run copy_prompthmr_vendor.sh first")

    if NEW.exists():
        shutil.rmtree(NEW)
    OLD.rename(NEW)
    print(f"Renamed {OLD.name} -> {NEW.name}")

    n = 0
    for py in NEW.rglob("*.py"):
        if patch_file(py):
            n += 1
    print(f"Patched {n} Python files under phmr_pipeline/")


if __name__ == "__main__":
    main()
