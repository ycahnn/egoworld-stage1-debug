#!/usr/bin/env python3
"""Rebuild camera extrinsics from saved task directions with zero gaze weight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_camera_pose_pipeline.geometry import camera_extrinsics, stabilize_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.8)
    parser.add_argument("--deadzone-degrees", type=float, default=2.0)
    parser.add_argument("--max-step-degrees", type=float, default=3.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def valid_direction(direction: np.ndarray) -> bool:
    return bool(np.isfinite(direction).all() and np.linalg.norm(direction) > 1e-8)


def main() -> None:
    args = parse_args()
    source_run = resolve(args.source_run)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_run / "camera_trajectory.npz"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = np.load(source_path)
    centers = source["ego_cam_point_exo"].astype(np.float64)
    task_directions = source["task_direction_exo"].astype(np.float64)
    gaze_directions = source["gaze_direction_exo"].astype(np.float64)
    if not (len(centers) == len(task_directions) == len(gaze_directions)):
        raise ValueError("Trajectory arrays have different frame counts")

    orientations = []
    rotations = []
    camera_to_exo = []
    exo_to_camera = []
    fallback_frames = []
    previous = None
    for index, (center, task, gaze) in enumerate(zip(centers, task_directions, gaze_directions)):
        if valid_direction(task):
            current = task / np.linalg.norm(task)
        elif valid_direction(gaze):
            current = gaze / np.linalg.norm(gaze)
            fallback_frames.append(index)
        else:
            raise ValueError(f"Frame {index} has neither a task nor gaze direction")
        if previous is not None:
            current = stabilize_direction(
                previous,
                current,
                smoothing=args.smoothing,
                deadzone_degrees=args.deadzone_degrees,
                max_step_degrees=args.max_step_degrees,
            )
        rotation, transform_camera_to_exo, transform_exo_to_camera = camera_extrinsics(
            center, current
        )
        orientations.append(current)
        rotations.append(rotation)
        camera_to_exo.append(transform_camera_to_exo)
        exo_to_camera.append(transform_exo_to_camera)
        previous = current

    payload = {key: source[key] for key in source.files}
    payload.update(
        camera_direction_exo=np.asarray(orientations),
        rotation_camera_to_exo=np.asarray(rotations),
        transform_camera_to_exo=np.asarray(camera_to_exo),
        transform_exo_to_camera=np.asarray(exo_to_camera),
    )
    np.savez_compressed(out_dir / "camera_trajectory.npz", **payload)

    summary = {
        "source_run": str(source_run),
        "orientation_mode": "task-only",
        "frame_count": len(orientations),
        "gaze_weight": 0.0,
        "task_weight": 1.0,
        "smoothing": args.smoothing,
        "deadzone_degrees": args.deadzone_degrees,
        "max_step_degrees": args.max_step_degrees,
        "gaze_fallback_frames": fallback_frames,
        "ego_intrinsics": source["ego_intrinsics"].tolist(),
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
