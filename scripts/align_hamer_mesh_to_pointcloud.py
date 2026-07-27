#!/usr/bin/env python3
"""Align a HaMeR hand mesh to MediaPipe depth-lifted point-cloud joints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a similarity transform from HaMeR mesh/joints to MediaPipe point-cloud joints."
    )
    parser.add_argument("--mediapipe_joints", required=True, help="MediaPipe joints_3d_pointcloud.npy")
    parser.add_argument("--mediapipe_valid", default=None, help="Optional MediaPipe joints_valid.npy")
    parser.add_argument("--hamer_metadata", required=True, help="HaMeR hamer_metadata.json")
    parser.add_argument("--hand_index", type=int, default=0, help="HaMeR hand output index to align")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--joint_indices", default=None, help="Comma-separated joint indices for fitting. Default: all valid.")
    parser.add_argument("--robust", action="store_true", help="Refit after dropping large residual outliers.")
    parser.add_argument("--image", default=None, help="Optional image for projection debug overlay")
    parser.add_argument("--depth", default=None, help="Optional target depth map for mesh z-buffer validation")
    parser.add_argument("--K", default=None, help="Optional K_exo.npy for projection/depth validation")
    return parser.parse_args()


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_joint_indices(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        return None
    return np.asarray(values, dtype=np.int32)


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t such that target ~= s * source @ R.T + t."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("At least 3 corresponding points are required")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    covariance = (tgt_centered.T @ src_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    src_variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if src_variance <= 0:
        raise ValueError("Degenerate source points; variance is zero")
    scale = float(np.sum(singular_values * np.diag(correction)) / src_variance)
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation.reshape(1, 3)


def robust_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    scale, rotation, translation = umeyama_similarity(source, target)
    aligned = apply_similarity(source, scale, rotation, translation)
    residual = np.linalg.norm(aligned - target, axis=1)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    threshold = median + 2.5 * max(mad * 1.4826, 1e-8)
    keep = residual <= threshold
    if int(keep.sum()) >= 6 and int(keep.sum()) < len(keep):
        scale, rotation, translation = umeyama_similarity(source[keep], target[keep])
    else:
        keep = np.ones_like(residual, dtype=bool)
    return scale, rotation, translation, keep


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# HaMeR mesh aligned to point-cloud frame\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces.astype(np.int64):
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.where(points[:, 2] == 0.0, 1e-8, points[:, 2])
    u = float(k[0, 0]) * points[:, 0] / z + float(k[0, 2])
    v = float(k[1, 1]) * points[:, 1] / z + float(k[1, 2])
    return np.stack([u, v], axis=1)


def draw_joints(image: np.ndarray, points_2d: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    h, w = image.shape[:2]
    for a, b in HAND_EDGES:
        p0 = points_2d[a]
        p1 = points_2d[b]
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color, 2, cv2.LINE_AA)
    for idx, pt in enumerate(points_2d):
        if not np.isfinite(pt).all():
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 4 if idx == 0 else 3, color, -1, cv2.LINE_AA)
    finite = points_2d[np.isfinite(points_2d).all(axis=1)]
    if finite.size:
        x0, y0 = np.min(finite, axis=0)
        cv2.putText(image, label, (int(x0), max(14, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_wireframe(image: np.ndarray, projected: np.ndarray, faces: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    edges = set()
    for face in faces:
        a, b, c = [int(x) for x in face]
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    for a, b in edges:
        p0 = projected[a]
        p1 = projected[b]
        if (
            np.isfinite(p0).all()
            and np.isfinite(p1).all()
            and 0 <= p0[0] < w
            and 0 <= p0[1] < h
            and 0 <= p1[0] < w
            and 0 <= p1[1] < h
        ):
            cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color, 1, cv2.LINE_AA)


def rasterize_depth(projected: np.ndarray, z_values: np.ndarray, faces: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    for face in faces:
        i0, i1, i2 = [int(x) for x in face]
        pts = projected[[i0, i1, i2]]
        zs = z_values[[i0, i1, i2]]
        if not np.isfinite(pts).all() or not np.isfinite(zs).all():
            continue
        min_x = max(int(np.floor(np.min(pts[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(pts[:, 0]))), width - 1)
        min_y = max(int(np.floor(np.min(pts[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(pts[:, 1]))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        p0, p1, p2 = pts
        denom = ((p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]))
        if abs(float(denom)) < 1e-9:
            continue
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px = x + 0.5
                py = y + 0.5
                w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
                w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = float(w0 * zs[0] + w1 * zs[1] + w2 * zs[2])
                if z > 0.0 and z < zbuffer[y, x]:
                    zbuffer[y, x] = z
    depth = np.full_like(zbuffer, np.nan, dtype=np.float32)
    valid = np.isfinite(zbuffer)
    depth[valid] = zbuffer[valid]
    return depth


def array_stats(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "median_abs": float(np.median(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    out_dir = resolve(args.out, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    mediapipe_joints_all = np.load(resolve(args.mediapipe_joints, root)).astype(float)
    mediapipe_joints = mediapipe_joints_all[0] if mediapipe_joints_all.ndim == 3 else mediapipe_joints_all
    if mediapipe_joints.shape != (21, 3):
        raise ValueError(f"Expected MediaPipe joints shape 21x3, got {mediapipe_joints.shape}")

    if args.mediapipe_valid:
        valid_all = np.load(resolve(args.mediapipe_valid, root)).astype(bool)
        mediapipe_valid = valid_all[0] if valid_all.ndim == 2 else valid_all
    else:
        mediapipe_valid = np.isfinite(mediapipe_joints).all(axis=1) & (mediapipe_joints[:, 2] > 0)

    metadata_path = resolve(args.hamer_metadata, root)
    metadata = load_json(metadata_path)
    outputs = metadata.get("outputs", [])
    if not outputs:
        raise ValueError(f"No HaMeR outputs in {metadata_path}")
    entry = outputs[args.hand_index]
    hamer_joints = np.load(resolve(entry["joints"], root)).astype(float)
    hamer_vertices = np.load(resolve(entry["vertices"], root)).astype(float)
    faces = np.load(resolve(entry["faces"], root)).astype(np.int64)

    requested_indices = parse_joint_indices(args.joint_indices)
    fit_mask = mediapipe_valid & np.isfinite(hamer_joints).all(axis=1)
    if requested_indices is not None:
        requested_mask = np.zeros(21, dtype=bool)
        requested_mask[requested_indices] = True
        fit_mask &= requested_mask
    fit_indices = np.where(fit_mask)[0]
    if fit_indices.size < 3:
        raise ValueError(f"Need at least 3 valid joint correspondences, got {fit_indices.size}")

    source = hamer_joints[fit_indices]
    target = mediapipe_joints[fit_indices]
    if args.robust:
        scale, rotation, translation, robust_keep_local = robust_fit(source, target)
        kept_indices = fit_indices[robust_keep_local]
    else:
        scale, rotation, translation = umeyama_similarity(source, target)
        kept_indices = fit_indices

    aligned_joints = apply_similarity(hamer_joints, scale, rotation, translation)
    aligned_vertices = apply_similarity(hamer_vertices, scale, rotation, translation)
    residuals = aligned_joints - mediapipe_joints
    distances = np.linalg.norm(residuals, axis=1)
    eval_mask = mediapipe_valid & np.isfinite(aligned_joints).all(axis=1)

    np.save(out_dir / "hamer_joints_aligned_to_pointcloud.npy", aligned_joints)
    np.save(out_dir / "hamer_vertices_aligned_to_pointcloud.npy", aligned_vertices)
    np.save(out_dir / "hamer_faces.npy", faces)
    np.save(out_dir / "alignment_residual_xyz.npy", residuals)
    np.save(out_dir / "alignment_joint_distances.npy", distances)
    np.save(out_dir / "similarity_scale.npy", np.asarray(scale, dtype=np.float64))
    np.save(out_dir / "similarity_rotation.npy", rotation)
    np.save(out_dir / "similarity_translation.npy", translation)
    save_obj(out_dir / "hamer_mesh_aligned_to_pointcloud.obj", aligned_vertices, faces)

    report = {
        "method": "Umeyama similarity alignment from HaMeR raw joints to MediaPipe depth-lifted point-cloud joints",
        "hamer_metadata": str(metadata_path),
        "hand_index": int(args.hand_index),
        "hand_side_for_hamer": entry.get("hand_side_for_hamer"),
        "fit_joint_indices_initial": fit_indices.astype(int).tolist(),
        "fit_joint_indices_used": kept_indices.astype(int).tolist(),
        "scale": float(scale),
        "rotation": rotation.round(10).tolist(),
        "translation": translation.round(10).tolist(),
        "joint_distance_stats_all_valid": array_stats(distances[eval_mask]),
        "joint_distance_stats_fit_joints": array_stats(distances[kept_indices]),
        "outputs": {
            "aligned_mesh_obj": "hamer_mesh_aligned_to_pointcloud.obj",
            "aligned_vertices": "hamer_vertices_aligned_to_pointcloud.npy",
            "aligned_joints": "hamer_joints_aligned_to_pointcloud.npy",
            "joint_distances": "alignment_joint_distances.npy",
            "residual_xyz": "alignment_residual_xyz.npy",
        },
    }

    if args.image and args.K:
        image = cv2.imread(str(resolve(args.image, root)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {args.image}")
        k = np.load(resolve(args.K, root)).astype(float)
        hamer_joints_2d = project(aligned_joints, k)
        mediapipe_joints_2d = project(mediapipe_joints, k)
        mesh_2d = project(aligned_vertices, k)
        overlay = image.copy()
        draw_wireframe(overlay, mesh_2d, faces, (255, 0, 255))
        draw_joints(overlay, mediapipe_joints_2d, (0, 180, 255), "MediaPipe depth")
        draw_joints(overlay, hamer_joints_2d, (0, 255, 0), "Aligned HaMeR")
        cv2.imwrite(str(out_dir / "aligned_mesh_projection_debug.png"), overlay)
        report["outputs"]["projection_debug"] = "aligned_mesh_projection_debug.png"

        if args.depth:
            depth = np.load(resolve(args.depth, root)).astype(float)
            mesh_depth = rasterize_depth(mesh_2d, aligned_vertices[:, 2], faces, depth.shape)
            valid_depth = np.isfinite(mesh_depth) & (mesh_depth > 0) & np.isfinite(depth) & (depth > 0)
            depth_delta = depth[valid_depth] - mesh_depth[valid_depth]
            np.save(out_dir / "aligned_mesh_rendered_depth.npy", mesh_depth)
            np.save(out_dir / "aligned_mesh_depth_valid_mask.npy", valid_depth)
            report["mesh_depth_minus_pointcloud_depth_convention"] = "depth_delta = target_depth - aligned_mesh_rendered_depth"
            report["mesh_depth_delta_stats"] = array_stats(depth_delta)
            report["outputs"]["aligned_mesh_rendered_depth"] = "aligned_mesh_rendered_depth.npy"
            report["outputs"]["aligned_mesh_depth_valid_mask"] = "aligned_mesh_depth_valid_mask.npy"

    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# HaMeR Mesh to Point-Cloud Alignment Report",
        "",
        f"method: {report['method']}",
        f"hand_index: {report['hand_index']}",
        f"hand_side_for_hamer: {report['hand_side_for_hamer']}",
        f"fit_joint_indices_used: {report['fit_joint_indices_used']}",
        f"scale: {report['scale']:.8f}",
        f"translation: {[round(x, 8) for x in report['translation']]}",
        "",
        "## Joint Distance Stats",
    ]
    for key, value in report["joint_distance_stats_all_valid"].items():
        lines.append(f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value}")
    if "mesh_depth_delta_stats" in report:
        lines.extend(["", "## Mesh Depth Delta Stats", "depth_delta = target_depth - aligned_mesh_rendered_depth"])
        for key, value in report["mesh_depth_delta_stats"].items():
            lines.append(f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value}")
    lines.extend(["", "## Outputs"])
    for key, value in report["outputs"].items():
        lines.append(f"{key}: {value}")
    (out_dir / "alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
