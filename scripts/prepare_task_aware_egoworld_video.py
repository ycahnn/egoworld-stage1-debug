#!/usr/bin/env python3
"""Prepare sparse RGB and aligned hand-pose frames for EgoWorld video inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_camera_pose_pipeline.geometry import backproject_pixel, sample_depth_median, transform_points


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGER_COLORS_RGB = (
    (239, 68, 68),
    (234, 179, 8),
    (34, 197, 94),
    (6, 182, 212),
    (217, 70, 239),
)
DEFAULT_PROMPT = (
    "A chef in a white uniform is standing at a counter, cooking eggs in a small frying pan on a portable electric stove. "
    "He grips the pan with tongs or a handle while using a red silicone spatula to stir and fold the eggs, carefully "
    "controlling their texture as he prepares a simple scrambled egg dish."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-run",
        type=Path,
        default=Path("outputs/ego_camera_pose_exo_task_aware"),
    )
    parser.add_argument(
        "--pointcloud-run",
        type=Path,
        help="Defaults to CAMERA_RUN/ego_pointcloud_dense.",
    )
    parser.add_argument(
        "--observation-run",
        type=Path,
        help="Run containing depth and hand_landmarks. Defaults to CAMERA_RUN.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("gaze-aware", "task-aware"),
        help="Defaults to the camera run summary, or gaze-aware for legacy runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/egoworld_task_aware_video/prepared_inputs"),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--sample-radii", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--line-width", type=int, default=6)
    parser.add_argument("--joint-radius", type=int, default=7)
    parser.add_argument("--fps", type=float)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-4)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if valid.any():
        projected = points[valid] @ intrinsics.T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def draw_pose(
    hands_pixels: np.ndarray,
    hands_valid: np.ndarray,
    width: int,
    height: int,
    line_width: int,
    joint_radius: int,
) -> tuple[np.ndarray, int]:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    visible_count = 0
    for points, valid in zip(hands_pixels, hands_valid):
        inside = (
            valid
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        visible_count += int(inside.sum())
        integer_points = np.rint(points).astype(np.int32)
        for edge_index, (start, end) in enumerate(HAND_EDGES):
            if inside[start] and inside[end]:
                rgb = FINGER_COLORS_RGB[edge_index // 4]
                cv2.line(
                    canvas,
                    tuple(integer_points[start]),
                    tuple(integer_points[end]),
                    rgb[::-1],
                    line_width,
                    cv2.LINE_AA,
                )
        for joint_index, point in enumerate(integer_points):
            if not inside[joint_index]:
                continue
            rgb = (248, 250, 252) if joint_index == 0 else FINGER_COLORS_RGB[(joint_index - 1) // 4]
            cv2.circle(canvas, tuple(point), joint_radius, rgb[::-1], -1, cv2.LINE_AA)
    return canvas, visible_count


def create_video(frame_paths: list[Path], output: Path, fps: float) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video: {output}")
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(path)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    camera_run = resolve(args.camera_run)
    pointcloud_run = resolve(args.pointcloud_run) if args.pointcloud_run else camera_run / "ego_pointcloud_dense"
    observation_run = resolve(args.observation_run) if args.observation_run else camera_run
    out_dir = resolve(args.out)
    sparse_dir = out_dir / "sparse"
    pose_dir = out_dir / "pose"
    text_dir = out_dir / "text"
    pose_3d_dir = out_dir / "pose_3d"
    for directory in (sparse_dir, pose_dir, text_dir, pose_3d_dir):
        directory.mkdir(parents=True, exist_ok=True)

    orientation_mode = args.orientation_mode
    if orientation_mode is None:
        summary_path = camera_run / "pipeline_summary.json"
        if summary_path.is_file():
            camera_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            orientation_mode = camera_summary.get("orientation_mode")
        orientation_mode = orientation_mode or "gaze-aware"

    trajectory = np.load(camera_run / "camera_trajectory.npz")
    hand_timeline = np.load(observation_run / "hand_landmarks" / "hand_timeline.npz")
    hand_pixels_exo = hand_timeline["hand_landmarks_pixels"]
    hand_count = hand_timeline["hand_count"]
    frame_indices = trajectory["frame_index"].astype(np.int64)
    transforms = trajectory["transform_exo_to_camera"]
    exo_intrinsics = trajectory["intrinsics"]
    ego_intrinsics = (
        trajectory["ego_intrinsics"] if "ego_intrinsics" in trajectory.files else exo_intrinsics
    )
    if len(frame_indices) != len(hand_pixels_exo):
        raise ValueError("Camera trajectory and hand timeline have different frame counts")

    prompt = resolve(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt
    timestamps = trajectory["timestamp_seconds"]
    fps = args.fps or (1.0 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 30.0)
    width = int(round(ego_intrinsics[0, 2] * 2.0))
    height = int(round(ego_intrinsics[1, 2] * 2.0))
    sparse_paths = []
    pose_paths = []
    rows = []

    for sequence_index, frame_index in enumerate(frame_indices):
        output_stem = f"{sequence_index:06d}"
        source_stem = f"frame_{int(frame_index):06d}"
        sparse_source = pointcloud_run / "projection_rgb" / f"{source_stem}.png"
        depth_source = observation_run / "depth" / f"{source_stem}.npz"
        if not sparse_source.is_file():
            raise FileNotFoundError(sparse_source)
        if not depth_source.is_file():
            raise FileNotFoundError(depth_source)
        sparse_destination = sparse_dir / f"{output_stem}.png"
        shutil.copy2(sparse_source, sparse_destination)
        sparse_paths.append(sparse_destination)
        (text_dir / f"{output_stem}.txt").write_text(prompt + "\n", encoding="utf-8")

        depth = np.load(depth_source)["depth"]
        count = int(hand_count[sequence_index])
        hands_exo = np.full((2, 21, 3), np.nan, dtype=np.float64)
        hands_ego = np.full((2, 21, 3), np.nan, dtype=np.float64)
        hands_projected = np.full((2, 21, 2), np.nan, dtype=np.float64)
        hands_valid = np.zeros((2, 21), dtype=bool)
        for hand_index in range(count):
            for joint_index, pixel in enumerate(hand_pixels_exo[sequence_index, hand_index]):
                value, _ = sample_depth_median(depth, pixel, tuple(args.sample_radii))
                if np.isfinite(value):
                    hands_exo[hand_index, joint_index] = backproject_pixel(
                        pixel, value, exo_intrinsics
                    )
            hands_ego[hand_index] = transform_points(
                hands_exo[hand_index], transforms[sequence_index]
            )
            hands_projected[hand_index], hands_valid[hand_index] = project_points(
                hands_ego[hand_index], ego_intrinsics
            )

        pose, visible_count = draw_pose(
            hands_projected,
            hands_valid,
            width,
            height,
            args.line_width,
            args.joint_radius,
        )
        pose_path = pose_dir / f"{output_stem}.png"
        if not cv2.imwrite(str(pose_path), pose):
            raise RuntimeError(f"Failed to write pose: {pose_path}")
        pose_paths.append(pose_path)
        np.savez_compressed(
            pose_3d_dir / f"{output_stem}.npz",
            joints_exo=hands_exo,
            joints_ego=hands_ego,
            joints_ego_pixels=hands_projected,
            joints_valid=hands_valid,
            transform_exo_to_ego=transforms[sequence_index],
            ego_intrinsics=ego_intrinsics,
        )
        rows.append(
            {
                "frame_index": int(frame_index),
                "hand_count": count,
                "valid_joints": int(hands_valid.sum()),
                "visible_joints": visible_count,
            }
        )
        print(
            f"[{sequence_index + 1:03d}/{len(frame_indices):03d}] {source_stem}: "
            f"hands={count}, visible_joints={visible_count}/42"
        )

    create_video(sparse_paths, out_dir / "sparse_preview.mp4", fps)
    create_video(pose_paths, out_dir / "pose_preview.mp4", fps)
    visible_counts = np.asarray([row["visible_joints"] for row in rows])
    summary = {
        "camera_run": str(camera_run),
        "pointcloud_run": str(pointcloud_run),
        "observation_run": str(observation_run),
        "orientation_mode": orientation_mode,
        "frame_count": len(rows),
        "fps": fps,
        "image_size": [width, height],
        "sparse_source": f"{orientation_mode} ego point-cloud z-buffer RGB projection",
        "pose_source": (
            "MediaPipe 21 joints per hand, depth-lifted in exo and transformed "
            f"to {orientation_mode} ego"
        ),
        "visible_joints": {
            "min": int(visible_counts.min()),
            "max": int(visible_counts.max()),
            "mean": float(visible_counts.mean()),
        },
        "prompt": prompt,
        "ego_intrinsics": ego_intrinsics.tolist(),
    }
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
