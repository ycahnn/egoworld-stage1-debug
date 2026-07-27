#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

REQUIRED_PATHS = [
    "inputs/exo.jpg",
    "outputs/depth_raw.npy",
    "outputs/K_exo.npy",
    "outputs/hand_bboxes.json",
    "outputs/hamer/hamer_metadata.json",
    "outputs/hamer_projection/final_projection_debug.png",
    "outputs/hamer_depth/hand_depth.npy",
    "outputs/hamer_depth/hand_mask.png",
    "outputs/scaled_depth/depth_scaled.npy",
    "outputs/scaled_depth/exo_point_cloud_scaled.ply",
]

SHAPE_PATHS = [
    "outputs/depth_raw.npy",
    "outputs/scaled_depth/depth_scaled.npy",
    "outputs/hamer_depth/hand_depth.npy",
]


def check_required_files(root: Path):
    missing = []
    for rel_path in REQUIRED_PATHS:
        path = root / rel_path
        if not path.exists():
            missing.append(rel_path)
    return missing


def load_shape(path: Path):
    try:
        array = np.load(str(path))
        return array.shape
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def main():
    project_root = Path(__file__).resolve().parent.parent
    print(f"Workspace root: {project_root}")

    missing_files = check_required_files(project_root)
    if missing_files:
        print("\nMissing required files:")
        for missing in missing_files:
            print(f" - {missing}")
        print("\nPlease restore the missing final outputs before running the full pipeline verification.")
        sys.exit(1)

    print("\nAll required files exist.")
    print("\nShapes:")
    for rel_path in SHAPE_PATHS:
        path = project_root / rel_path
        shape = load_shape(path)
        print(f" - {rel_path}: {shape}")

    print("\nWorkspace integrity check passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
