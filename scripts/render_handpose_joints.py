#!/usr/bin/env python3
"""Render one or two 21-joint hand poses from 3D coordinates."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = {
    range(1, 5): "#ef4444",
    range(5, 9): "#eab308",
    range(9, 13): "#22c55e",
    range(13, 17): "#06b6d4",
    range(17, 21): "#d946ef",
}


def joint_color(index: int) -> str:
    if index == 0:
        return "#f8fafc"
    for indices, color in FINGER_COLORS.items():
        if index in indices:
            return color
    return "#94a3b8"


def load_joints(path: Path) -> np.ndarray:
    joints = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    joints = np.asarray(joints, dtype=np.float32)
    if joints.size == 126:
        joints = joints.reshape(2, 21, 3)
    elif joints.size == 63:
        joints = joints.reshape(1, 21, 3)
    else:
        raise ValueError(f"Expected 63 or 126 values, got {joints.size}: {path}")
    return joints


def draw_hand_3d(ax, hand: np.ndarray, label: str) -> None:
    for a, b in HAND_EDGES:
        ax.plot(
            hand[[a, b], 0],
            hand[[a, b], 1],
            hand[[a, b], 2],
            color=joint_color(b),
            linewidth=2.5,
        )
    for index, point in enumerate(hand):
        ax.scatter(*point, color=joint_color(index), s=24, edgecolor="#111827", linewidth=0.4)
    ax.set_title(label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def draw_hand_xy(ax, hand: np.ndarray, label: str) -> None:
    for a, b in HAND_EDGES:
        ax.plot(
            hand[[a, b], 0],
            hand[[a, b], 1],
            color=joint_color(b),
            linewidth=2.5,
        )
    for index, point in enumerate(hand):
        ax.scatter(
            point[0],
            point[1],
            color=joint_color(index),
            s=24,
            edgecolor="#111827",
            linewidth=0.4,
        )
    ax.set_title(label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()


def set_equal_3d_axes(axes, joints: np.ndarray) -> None:
    mins = joints.min(axis=(0, 1))
    maxs = joints.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render predicted 3D hand joints.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    joints = load_joints(args.input)
    labels = ["Left hand", "Right hand"] if len(joints) == 2 else ["Hand"]
    columns = len(joints)
    figure = plt.figure(figsize=(7 * columns, 11), facecolor="#0b1020")
    three_d_axes = []

    for index, (hand, label) in enumerate(zip(joints, labels)):
        ax_3d = figure.add_subplot(2, columns, index + 1, projection="3d")
        ax_xy = figure.add_subplot(2, columns, columns + index + 1)
        three_d_axes.append(ax_3d)
        draw_hand_3d(ax_3d, hand, f"{label} - 3D")
        draw_hand_xy(ax_xy, hand, f"{label} - XY view")

    set_equal_3d_axes(three_d_axes, joints)
    for ax in figure.axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#cbd5e1")
        ax.title.set_color("#f8fafc")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_color("#cbd5e1")
        ax.grid(True, color="#334155", alpha=0.45)

    figure.suptitle("Predicted Hand Pose", color="#f8fafc", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
