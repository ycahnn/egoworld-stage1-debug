#!/usr/bin/env python3
"""Render predicted 3D hand joints projected into an egocentric camera view."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = [
    (239, 68, 68),
    (234, 179, 8),
    (34, 197, 94),
    (6, 182, 212),
    (217, 70, 239),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/vit_handpose_exo_best/joints_2x21x3.npy", type=Path)
    parser.add_argument("--out", default="outputs/vit_handpose_exo_best/predicted_hand_pose_egoview.png", type=Path)
    parser.add_argument(
        "--intrinsics",
        nargs=6,
        type=float,
        metavar=("FX", "FY", "CX", "CY", "WIDTH", "HEIGHT"),
        default=(636.6593017578125, 636.251953125, 635.283881879317, 366.8740353496978, 1280, 720),
    )
    parser.add_argument("--line-width", default=5, type=int)
    parser.add_argument("--joint-radius", default=6, type=int)
    return parser.parse_args()


def load_joints(path: Path) -> np.ndarray:
    joints = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    joints = np.asarray(joints, dtype=np.float64)
    if joints.size == 126:
        return joints.reshape(2, 21, 3)
    if joints.size == 63:
        return joints.reshape(1, 21, 3)
    if joints.ndim == 3 and joints.shape[1:] == (21, 3):
        return joints
    raise ValueError(f"Expected one or two 21x3 hands, got shape {joints.shape} from {path}")


def project(joints: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    z = joints[..., 2]
    valid = np.isfinite(joints).all(axis=-1) & (z > 1e-8)
    points_2d = np.full(joints.shape[:2] + (2,), np.nan, dtype=np.float64)
    points_2d[..., 0] = fx * joints[..., 0] / z + cx
    points_2d[..., 1] = fy * joints[..., 1] / z + cy
    return points_2d, valid


def draw_hand(draw: ImageDraw.ImageDraw, points: np.ndarray, valid: np.ndarray) -> None:
    for edge_index, (a, b) in enumerate(HAND_EDGES):
        if valid[a] and valid[b]:
            color = FINGER_COLORS[edge_index // 4]
            draw.line(
                [tuple(points[a]), tuple(points[b])],
                fill=color,
                width=args.line_width,
                joint="curve",
            )

    wrist_color = (248, 250, 252)
    for index, point in enumerate(points):
        if not valid[index]:
            continue
        radius = args.joint_radius
        x, y = point
        color = wrist_color if index == 0 else FINGER_COLORS[(index - 1) // 4]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


if __name__ == "__main__":
    args = parse_args()
    fx, fy, cx, cy, width, height = args.intrinsics
    width, height = int(width), int(height)

    joints_3d = load_joints(args.input)
    joints_2d, valid = project(joints_3d, fx, fy, cx, cy)

    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for hand_points, hand_valid in zip(joints_2d, valid):
        draw_hand(draw, hand_points, hand_valid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)

    visible = (
        valid
        & (joints_2d[..., 0] >= 0)
        & (joints_2d[..., 0] < width)
        & (joints_2d[..., 1] >= 0)
        & (joints_2d[..., 1] < height)
    )
    print(f"Saved: {args.out}")
    print(f"Projected joints inside frame: {int(visible.sum())}/{visible.size}")
