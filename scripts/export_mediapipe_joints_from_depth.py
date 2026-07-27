#!/usr/bin/env python3
"""Lift MediaPipe hand landmarks into the exocentric point-cloud frame.

This script does not perform scale fitting. It samples the provided depth map
at MediaPipe hand landmark pixels and backprojects those pixels with the given
camera intrinsics, producing joints in the same coordinate frame as a point
cloud generated from that depth map and K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MediaPipe hand landmarks as 3D points in the depth point-cloud frame."
    )
    parser.add_argument("--image", default="inputs/exo.jpg", help="Input RGB image.")
    parser.add_argument("--depth", default="outputs/depth_raw.npy", help="Depth map used to create the point cloud.")
    parser.add_argument("--K", default="outputs/K_exo.npy", help="Camera intrinsic matrix used for point cloud backprojection.")
    parser.add_argument("--out", default="outputs/mediapipe_pointcloud_joints", help="Output directory.")
    parser.add_argument("--max_num_hands", type=int, default=2, help="Maximum number of hands to detect.")
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    parser.add_argument("--sample_radii", type=int, nargs="+", default=[0, 2, 4, 8, 12])
    parser.add_argument("--no_flip_handedness", action="store_false", dest="flip_handedness")
    return parser.parse_args()


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def corrected_handedness(raw: str | None, flip: bool) -> str:
    if raw is None:
        return "unknown"
    label = raw.capitalize()
    if not flip:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def sample_depth_median(depth: np.ndarray, u: float, v: float, radii: list[int]) -> tuple[float | None, int | None]:
    h, w = depth.shape
    x = int(round(u))
    y = int(round(v))
    for radius in radii:
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        vals = depth[y0:y1, x0:x1]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            return float(np.median(vals)), radius
    return None, None


def backproject(u: float, v: float, z: float, K: np.ndarray) -> list[float]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return [x, y, z]


def detect_hands(image_bgr: np.ndarray, max_num_hands: int, min_detection_confidence: float) -> list[dict[str, Any]]:
    mp_hands = mp.solutions.hands
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    hands: list[dict[str, Any]] = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    ) as detector:
        result = detector.process(image_rgb)
    if not result.multi_hand_landmarks:
        return hands

    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        raw_label = None
        score = None
        if result.multi_handedness and idx < len(result.multi_handedness):
            classifications = result.multi_handedness[idx].classification
            if classifications:
                raw_label = classifications[0].label
                score = float(classifications[0].score)

        pixels = []
        for lm in hand_landmarks.landmark:
            u = min(max(float(lm.x) * w, 0.0), float(w - 1))
            v = min(max(float(lm.y) * h, 0.0), float(h - 1))
            pixels.append([u, v])

        hands.append(
            {
                "index": idx,
                "raw_handedness": raw_label,
                "confidence": score,
                "landmarks_px": pixels,
            }
        )
    return hands


def draw_debug(image_bgr: np.ndarray, joints_2d: np.ndarray, valid: np.ndarray, labels: list[str]) -> np.ndarray:
    out = image_bgr.copy()
    colors = [(0, 180, 255), (0, 255, 80), (255, 255, 255)]
    for hand_idx in range(joints_2d.shape[0]):
        color = colors[hand_idx % len(colors)]
        pts = joints_2d[hand_idx]
        hand_valid = valid[hand_idx]
        for a, b in HAND_CONNECTIONS:
            if hand_valid[a] and hand_valid[b]:
                p0 = tuple(np.round(pts[a]).astype(int))
                p1 = tuple(np.round(pts[b]).astype(int))
                cv2.line(out, p0, p1, color, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            x, y = np.round(pt).astype(int)
            if hand_valid[i]:
                cv2.circle(out, (int(x), int(y)), 3 if i else 5, color, -1, cv2.LINE_AA)
        finite = pts[np.isfinite(pts).all(axis=1)]
        if finite.size:
            x0, y0 = np.min(finite, axis=0)
            cv2.putText(out, labels[hand_idx], (int(x0), max(12, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    image_path = resolve(args.image, root)
    depth_path = resolve(args.depth, root)
    k_path = resolve(args.K, root)
    out_dir = resolve(args.out, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    depth = np.load(str(depth_path))
    K = np.load(str(k_path))
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got shape {depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K.shape}")
    if image_bgr.shape[:2] != depth.shape:
        raise ValueError(f"Image shape {image_bgr.shape[:2]} does not match depth shape {depth.shape}")

    hands = detect_hands(image_bgr, args.max_num_hands, args.min_detection_confidence)
    if not hands:
        raise RuntimeError("No MediaPipe hands detected.")

    hand_count = len(hands)
    joints_2d = np.full((hand_count, 21, 2), np.nan, dtype=np.float32)
    joints_3d = np.full((hand_count, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((hand_count, 21), dtype=bool)
    depth_samples = np.full((hand_count, 21), np.nan, dtype=np.float32)
    used_radii = np.full((hand_count, 21), -1, dtype=np.int32)
    labels = []
    records = []

    for hand_idx, hand in enumerate(hands):
        handedness = corrected_handedness(hand["raw_handedness"], args.flip_handedness)
        labels.append(handedness.lower())
        for joint_idx, (u, v) in enumerate(hand["landmarks_px"]):
            joints_2d[hand_idx, joint_idx] = [u, v]
            z, radius = sample_depth_median(depth, u, v, args.sample_radii)
            if z is None:
                continue
            joints_3d[hand_idx, joint_idx] = backproject(u, v, z, K)
            valid[hand_idx, joint_idx] = True
            depth_samples[hand_idx, joint_idx] = z
            used_radii[hand_idx, joint_idx] = int(radius)
        records.append(
            {
                "index": int(hand_idx),
                "raw_handedness": hand["raw_handedness"],
                "corrected_handedness": handedness,
                "confidence": hand["confidence"],
                "valid_joint_count": int(valid[hand_idx].sum()),
            }
        )

    np.save(out_dir / "joints_3d_pointcloud.npy", joints_3d)
    np.save(out_dir / "joints_2d_pixels.npy", joints_2d)
    np.save(out_dir / "joints_valid.npy", valid)
    np.save(out_dir / "joint_depth_samples.npy", depth_samples)
    np.save(out_dir / "joint_depth_sample_radii.npy", used_radii)

    for hand_idx, label in enumerate(labels):
        np.savetxt(out_dir / f"hand_{hand_idx:02d}_{label}_joints_3d.txt", joints_3d[hand_idx], fmt="%.8f")
        np.savetxt(out_dir / f"hand_{hand_idx:02d}_{label}_joints_2d.txt", joints_2d[hand_idx], fmt="%.3f")

    debug = draw_debug(image_bgr, joints_2d, valid, labels)
    cv2.imwrite(str(out_dir / "mediapipe_joints_debug.png"), debug)

    metadata = {
        "coordinate_system": "same_as_point_cloud_from_depth_and_K",
        "method": "MediaPipe 2D landmarks sampled from depth map and backprojected with K; no scale fitting",
        "image": str(image_path.relative_to(root)) if image_path.is_relative_to(root) else str(image_path),
        "depth": str(depth_path.relative_to(root)) if depth_path.is_relative_to(root) else str(depth_path),
        "K": str(k_path.relative_to(root)) if k_path.is_relative_to(root) else str(k_path),
        "outputs": {
            "joints_3d": "joints_3d_pointcloud.npy",
            "joints_2d": "joints_2d_pixels.npy",
            "valid": "joints_valid.npy",
            "depth_samples": "joint_depth_samples.npy",
            "debug": "mediapipe_joints_debug.png",
        },
        "joint_order": [f"mediapipe_hand_landmark_{i:02d}" for i in range(21)],
        "hands": records,
        "sample_radii": args.sample_radii,
        "flip_handedness": bool(args.flip_handedness),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Detected {hand_count} hand(s)")
    print(f"Saved 3D joints: {out_dir / 'joints_3d_pointcloud.npy'}")
    print(f"Saved debug overlay: {out_dir / 'mediapipe_joints_debug.png'}")
    print(f"Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
