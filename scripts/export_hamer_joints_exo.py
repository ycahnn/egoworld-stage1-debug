#!/usr/bin/env python3
"""Export HaMeR joints in the exocentric camera coordinate frame.

This uses the same camera metadata and left-hand 2D mirror correction used by
the mesh-depth renderer, but exports joints instead of MediaPipe depth-lifted
landmarks.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MANO_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

HAND_COLORS = {
    "left": (0, 128, 255),
    "right": (0, 255, 0),
    "unknown": (255, 255, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export HaMeR joints in exo camera coordinates")
    parser.add_argument("--image", default="inputs/exo.jpg", help="Input exo image")
    parser.add_argument("--hamer_dir", default="outputs/hamer", help="HaMeR output directory")
    parser.add_argument("--scaled_depth", default="outputs/scaled_depth/depth_scaled.npy", help="Scaled depth map for scale diagnostics")
    parser.add_argument("--out", default="outputs/hamer_pose_exo", help="Output directory")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve(path_str, project_root):
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def normalize_focal(focal):
    if focal is None:
        raise ValueError("Missing focal value")
    arr = np.asarray(focal, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    if arr.size >= 2:
        return float(arr[0]), float(arr[1])
    raise ValueError("Invalid focal value")


def normalize_cam_t(cam_t):
    if cam_t is None:
        raise ValueError("Missing camera translation")
    arr = np.asarray(cam_t, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("Camera translation must contain 3 values")
    return arr[:3].astype(float)


def camera_center(preprocessing, image_shape, img_size):
    if img_size is not None and len(img_size) >= 2:
        return float(img_size[0]) / 2.0, float(img_size[1]) / 2.0
    width = preprocessing.get("original_image_width", image_shape[1])
    height = preprocessing.get("original_image_height", image_shape[0])
    return float(width) / 2.0, float(height) / 2.0


def project_points(points_cam, focal, center):
    fx, fy = focal
    cx, cy = center
    z = np.where(points_cam[:, 2] == 0.0, 1e-6, points_cam[:, 2])
    u = fx * points_cam[:, 0] / z + cx
    v = fy * points_cam[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def mirror_2d_in_bbox(points_2d, bbox):
    left, _top, right, _bottom = [float(v) for v in bbox]
    mirrored = points_2d.copy()
    mirrored[:, 0] = left + right - mirrored[:, 0]
    return mirrored


def backproject_points(points_2d, z, focal, center):
    fx, fy = focal
    cx, cy = center
    x = (points_2d[:, 0] - cx) * z / fx
    y = (points_2d[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def draw_skeleton(image, points_2d, valid, color, label):
    h, w = image.shape[:2]
    for a, b in MANO_EDGES:
        if a >= len(points_2d) or b >= len(points_2d) or not (valid[a] and valid[b]):
            continue
        p0 = points_2d[a]
        p1 = points_2d[b]
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color, 2, lineType=cv2.LINE_AA)
    for idx, pt in enumerate(points_2d):
        if not valid[idx] or not np.isfinite(pt).all():
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 4 if idx == 0 else 3, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(image, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, lineType=cv2.LINE_AA)
    finite = points_2d[np.isfinite(points_2d).all(axis=1)]
    if finite.size:
        x0, y0 = np.min(finite, axis=0)
        cv2.putText(image, label, (int(x0), max(12, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)


def sample_depth(depth, points_2d):
    if depth is None:
        return np.full((len(points_2d),), np.nan, dtype=float)
    h, w = depth.shape[:2]
    out = np.full((len(points_2d),), np.nan, dtype=float)
    for idx, (u, v) in enumerate(points_2d):
        x = int(round(u))
        y = int(round(v))
        x0, x1 = max(0, x - 3), min(w, x + 4)
        y0, y1 = max(0, y - 3), min(h, y + 4)
        if x0 >= x1 or y0 >= y1:
            continue
        vals = depth[y0:y1, x0:x1]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            out[idx] = float(np.median(vals))
    return out


def save_report(path, metadata):
    lines = [
        "# HaMeR Exo Joint Export Report",
        "",
        f"coordinate_system: {metadata['coordinate_system']}",
        f"source: {metadata['source']}",
        f"combined_joints: {metadata['combined_joints_exo']}",
        f"combined_projection: {metadata['combined_joints_2d']}",
        f"debug_overlay: {metadata['debug_overlay']}",
        f"debug_black: {metadata['debug_black']}",
        "",
    ]
    for hand in metadata["hands"]:
        lines.extend([
            f"## Hand {hand['index']} ({hand['hand_side_for_hamer']})",
            f"joints_exo: {hand['joints_exo']}",
            f"joints_2d: {hand['joints_2d']}",
            f"camera_translation_source: {hand['camera_translation_source']}",
            f"left_bbox_2d_mirror_applied: {hand['left_bbox_2d_mirror_applied']}",
            f"z_range: {hand['z_range']}",
            f"scaled_depth_minus_joint_z_median: {hand['scaled_depth_minus_joint_z_median']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    image_path = resolve(args.image, project_root)
    hamer_dir = resolve(args.hamer_dir, project_root)
    out_dir = resolve(args.out, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    scaled_depth_path = resolve(args.scaled_depth, project_root)
    scaled_depth = np.load(str(scaled_depth_path)) if scaled_depth_path.exists() else None

    hamer_metadata = load_json(hamer_dir / "hamer_metadata.json")
    outputs = hamer_metadata.get("outputs", [])
    if not outputs:
        raise ValueError("No HaMeR outputs found")

    overlay = image.copy()
    black = np.zeros_like(image)
    all_joints_exo = []
    all_joints_2d = []
    all_valid = []
    hand_records = []

    for entry in outputs:
        idx = int(entry.get("index", len(hand_records)))
        hand_side = str(entry.get("hand_side_for_hamer", "unknown")).lower()
        hand_name = hand_side if hand_side in HAND_COLORS else "unknown"
        color = HAND_COLORS[hand_name]

        ham_out = entry.get("hamer_output", {})
        preprocessing = ham_out.get("preprocessing", {})
        img_size = preprocessing.get("img_size") or entry.get("img_size")
        scaled_focal = ham_out.get("scaled_focal_length") or entry.get("scaled_focal_length")
        focal = normalize_focal(scaled_focal or ham_out.get("focal_length") or ham_out.get("focal"))
        center = camera_center(preprocessing, image.shape, img_size)
        cam_t = ham_out.get("pred_cam_t_full") or entry.get("pred_cam_t_full") or ham_out.get("full_img_cam_t")
        camera_translation_source = "pred_cam_t_full" if cam_t is not None else "pred_cam_t"
        if cam_t is None:
            cam_t = ham_out.get("pred_cam_t")
        cam_t = normalize_cam_t(cam_t)

        joints_local = np.load(str(resolve(entry["joints"], project_root))).astype(float)
        if joints_local.shape != (21, 3):
            raise ValueError(f"Expected 21x3 joints for hand {idx}, got {joints_local.shape}")

        joints_cam_original = joints_local + cam_t.reshape(1, 3)
        joints_2d = project_points(joints_cam_original, focal, center)
        mirror_applied = False
        if hand_side == "left":
            joints_2d = mirror_2d_in_bbox(joints_2d, entry.get("bbox_xyxy", [0, 0, 0, 0]))
            mirror_applied = True

        joints_exo = backproject_points(joints_2d, joints_cam_original[:, 2], focal, center)
        valid = np.isfinite(joints_exo).all(axis=1) & (joints_exo[:, 2] > 0)

        stem = f"hand_{idx:02d}_{hand_name}"
        joints_exo_path = out_dir / f"{stem}_joints_exo.npy"
        joints_2d_path = out_dir / f"{stem}_joints_2d.npy"
        valid_path = out_dir / f"{stem}_joints_valid.npy"
        np.save(joints_exo_path, joints_exo)
        np.save(joints_2d_path, joints_2d)
        np.save(valid_path, valid)

        draw_skeleton(overlay, joints_2d, valid, color, f"{stem} HaMeR joints")
        draw_skeleton(black, joints_2d, valid, color, f"{stem} HaMeR joints")

        depth_at_joint = sample_depth(scaled_depth, joints_2d)
        dz = depth_at_joint - joints_exo[:, 2]
        dz_finite = dz[np.isfinite(dz)]
        dz_median = float(np.median(dz_finite)) if dz_finite.size else None

        hand_records.append({
            "index": idx,
            "hand_side_for_hamer": hand_side,
            "joints_exo": str(joints_exo_path.relative_to(project_root)),
            "joints_2d": str(joints_2d_path.relative_to(project_root)),
            "joints_valid": str(valid_path.relative_to(project_root)),
            "camera_translation_source": camera_translation_source,
            "left_bbox_2d_mirror_applied": mirror_applied,
            "focal": [float(focal[0]), float(focal[1])],
            "center": [float(center[0]), float(center[1])],
            "z_range": [float(np.min(joints_exo[:, 2])), float(np.max(joints_exo[:, 2]))],
            "scaled_depth_minus_joint_z_median": dz_median,
        })
        all_joints_exo.append(joints_exo)
        all_joints_2d.append(joints_2d)
        all_valid.append(valid)

    combined_joints = np.concatenate(all_joints_exo, axis=0)
    combined_2d = np.concatenate(all_joints_2d, axis=0)
    combined_valid = np.concatenate(all_valid, axis=0)

    combined_joints_path = out_dir / "hamer_joints_exo.npy"
    combined_2d_path = out_dir / "hamer_joints_2d.npy"
    combined_valid_path = out_dir / "hamer_joints_valid.npy"
    overlay_path = out_dir / "hamer_joints_exo_debug.png"
    black_path = out_dir / "hamer_joints_exo_black.png"
    metadata_path = out_dir / "hamer_joints_exo_metadata.json"
    report_path = out_dir / "hamer_joints_exo_report.md"

    np.save(combined_joints_path, combined_joints)
    np.save(combined_2d_path, combined_2d)
    np.save(combined_valid_path, combined_valid)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(black_path), black)

    metadata = {
        "coordinate_system": "exo_camera_from_hamer_mesh_projection_scale",
        "source": "HaMeR pred_keypoints_3d translated by pred_cam_t_full; left hand follows the same 2D mirror correction used for mesh depth rendering",
        "source_hamer_metadata": str((hamer_dir / "hamer_metadata.json").relative_to(project_root)),
        "scaled_depth_reference": str(scaled_depth_path.relative_to(project_root)) if scaled_depth_path.exists() else None,
        "joint_order": [f"hamer_mano_joint_{i:02d}" for i in range(21)],
        "hand_order": [record["hand_side_for_hamer"] for record in hand_records],
        "combined_joints_exo": str(combined_joints_path.relative_to(project_root)),
        "combined_joints_2d": str(combined_2d_path.relative_to(project_root)),
        "combined_joints_valid": str(combined_valid_path.relative_to(project_root)),
        "debug_overlay": str(overlay_path.relative_to(project_root)),
        "debug_black": str(black_path.relative_to(project_root)),
        "hands": hand_records,
        "note": "These are HaMeR/MANO joints in the same exo camera scale as the rendered HaMeR hand depth, not MediaPipe depth-lifted landmarks.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_report(report_path, metadata)

    print(f"Saved HaMeR exo joints to {combined_joints_path}")
    print(f"Saved 2D projection to {combined_2d_path}")
    print(f"Saved debug overlay to {overlay_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
