#!/usr/bin/env python3
"""Compare MediaPipe depth-lifted joints with HaMeR projected point-cloud joints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mediapipe_dir", required=True)
    parser.add_argument("--hamer_dir", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def sample_depth(depth: np.ndarray, u: float, v: float, radii: tuple[int, ...] = (0, 2, 4, 8, 12)) -> tuple[float, int]:
    h, w = depth.shape
    x = int(round(float(u)))
    y = int(round(float(v)))
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
    return float("nan"), -1


def backproject(points_2d: np.ndarray, depth: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    joints = np.full((points_2d.shape[0], 3), np.nan, dtype=float)
    samples = np.full((points_2d.shape[0],), np.nan, dtype=float)
    radii = np.full((points_2d.shape[0],), -1, dtype=np.int32)
    for idx, (u, v) in enumerate(points_2d):
        z, radius = sample_depth(depth, float(u), float(v))
        samples[idx] = z
        radii[idx] = radius
        if np.isfinite(z) and z > 0:
            joints[idx] = [(float(u) - cx) * z / fx, (float(v) - cy) * z / fy, z]
    return joints, samples, radii


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    mediapipe_dir = resolve(args.mediapipe_dir, root)
    hamer_dir = resolve(args.hamer_dir, root)
    depth = np.load(resolve(args.depth, root)).astype(float)
    k = np.load(resolve(args.K, root)).astype(float)
    out_dir = resolve(args.out, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    mediapipe_3d = np.load(mediapipe_dir / "joints_3d_pointcloud.npy")[0].astype(float)
    mediapipe_2d = np.load(mediapipe_dir / "joints_2d_pixels.npy")[0].astype(float)
    mediapipe_valid = np.load(mediapipe_dir / "joints_valid.npy")[0].astype(bool)

    hamer_2d = np.load(hamer_dir / "hamer_joints_2d.npy").reshape(-1, 21, 2)[0].astype(float)
    hamer_mesh_exo = np.load(hamer_dir / "hamer_joints_exo.npy").reshape(-1, 21, 3)[0].astype(float)
    hamer_pointcloud, hamer_depth_samples, hamer_depth_radii = backproject(hamer_2d, depth, k)
    hamer_valid = np.isfinite(hamer_pointcloud).all(axis=1) & (hamer_pointcloud[:, 2] > 0)

    valid = mediapipe_valid & hamer_valid & np.isfinite(mediapipe_3d).all(axis=1)
    delta = hamer_pointcloud - mediapipe_3d
    distance = np.linalg.norm(delta, axis=1)

    mediapipe_rel = mediapipe_3d - mediapipe_3d[0]
    hamer_rel = hamer_pointcloud - hamer_pointcloud[0]
    wrist_aligned_delta = hamer_rel - mediapipe_rel
    wrist_aligned_distance = np.linalg.norm(wrist_aligned_delta, axis=1)

    stats = {
        "valid_joint_count": int(valid.sum()),
        "distance_mean": float(np.nanmean(distance[valid])),
        "distance_median": float(np.nanmedian(distance[valid])),
        "distance_min": float(np.nanmin(distance[valid])),
        "distance_max": float(np.nanmax(distance[valid])),
        "wrist_aligned_distance_mean": float(np.nanmean(wrist_aligned_distance[valid])),
        "wrist_aligned_distance_median": float(np.nanmedian(wrist_aligned_distance[valid])),
        "wrist_aligned_distance_max": float(np.nanmax(wrist_aligned_distance[valid])),
        "mean_abs_delta_xyz": np.nanmean(np.abs(delta[valid]), axis=0).round(6).tolist(),
        "median_abs_delta_xyz": np.nanmedian(np.abs(delta[valid]), axis=0).round(6).tolist(),
        "note": (
            "HaMeR projected point-cloud joints use HaMeR 2D joint projections, sample the provided "
            "depth map at those pixels, then backproject with K. hamer_mesh_exo_xyz_unscaled_reference "
            "is saved only as a reference."
        ),
    }

    rows = []
    for idx in range(21):
        rows.append(
            {
                "joint": idx,
                "valid": bool(valid[idx]),
                "mediapipe_xyz": mediapipe_3d[idx].round(6).tolist(),
                "hamer_projected_pointcloud_xyz": hamer_pointcloud[idx].round(6).tolist(),
                "delta_hamer_minus_mediapipe_xyz": delta[idx].round(6).tolist(),
                "distance": float(distance[idx]) if np.isfinite(distance[idx]) else None,
                "wrist_aligned_distance": (
                    float(wrist_aligned_distance[idx]) if np.isfinite(wrist_aligned_distance[idx]) else None
                ),
                "mediapipe_uv": mediapipe_2d[idx].round(3).tolist(),
                "hamer_projected_uv": hamer_2d[idx].round(3).tolist(),
                "hamer_mesh_exo_xyz_unscaled_reference": hamer_mesh_exo[idx].round(6).tolist(),
            }
        )

    np.save(out_dir / "hamer_projected_pointcloud_joints.npy", hamer_pointcloud)
    np.save(out_dir / "hamer_projected_pointcloud_joint_depth_samples.npy", hamer_depth_samples)
    np.save(out_dir / "hamer_projected_pointcloud_joint_depth_sample_radii.npy", hamer_depth_radii)
    np.save(out_dir / "hamer_projected_pointcloud_joints_valid.npy", hamer_valid)
    np.save(out_dir / "mediapipe_vs_hamer_projected_delta_xyz.npy", delta)
    np.save(out_dir / "mediapipe_vs_hamer_projected_distance.npy", distance)
    np.save(out_dir / "mediapipe_vs_hamer_projected_wrist_aligned_distance.npy", wrist_aligned_distance)
    (out_dir / "mediapipe_vs_hamer_projected_pointcloud_comparison.json").write_text(
        json.dumps({"stats": stats, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# MediaPipe vs HaMeR Projected Point-Cloud Joint Comparison",
        "",
        "Both compared coordinates are in the scaled depth point-cloud frame.",
        "",
        f"valid_joint_count: {stats['valid_joint_count']}/21",
        f"distance_mean: {stats['distance_mean']:.6f}",
        f"distance_median: {stats['distance_median']:.6f}",
        f"distance_min: {stats['distance_min']:.6f}",
        f"distance_max: {stats['distance_max']:.6f}",
        f"wrist_aligned_distance_mean: {stats['wrist_aligned_distance_mean']:.6f}",
        f"wrist_aligned_distance_median: {stats['wrist_aligned_distance_median']:.6f}",
        f"wrist_aligned_distance_max: {stats['wrist_aligned_distance_max']:.6f}",
        f"mean_abs_delta_xyz: {stats['mean_abs_delta_xyz']}",
        f"median_abs_delta_xyz: {stats['median_abs_delta_xyz']}",
        "",
        "| joint | dist | wrist_aligned_dist | dx | dy | dz | mp_u | mp_v | hamer_u | hamer_v |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx in range(21):
        lines.append(
            f"| {idx} | {distance[idx]:.6f} | {wrist_aligned_distance[idx]:.6f} | "
            f"{delta[idx, 0]:.6f} | {delta[idx, 1]:.6f} | {delta[idx, 2]:.6f} | "
            f"{mediapipe_2d[idx, 0]:.2f} | {mediapipe_2d[idx, 1]:.2f} | "
            f"{hamer_2d[idx, 0]:.2f} | {hamer_2d[idx, 1]:.2f} |"
        )
    (out_dir / "mediapipe_vs_hamer_projected_pointcloud_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
