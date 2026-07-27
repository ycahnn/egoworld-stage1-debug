from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_camera_pose_pipeline.geometry import (
    backproject_pointcloud,
    project_pointcloud_zbuffer,
    transform_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform exocentric point clouds into the estimated ego-camera frame."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/ego_camera_pose_exo"),
        help="Output directory produced by run_pipeline.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: RUN_DIR/ego_pointcloud).",
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Run containing frames/depth/pointcloud. Defaults to RUN_DIR.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--ego-intrinsics",
        type=Path,
        default=None,
        help="Optional 3x3 .npy intrinsics. Defaults to trajectory ego intrinsics, then exo K.",
    )
    parser.add_argument("--near", type=float, default=1e-4)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Rebuild stride-1 point clouds from full RGB/depth instead of using saved stride-4 clouds.",
    )
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Write RGB projections and video without large NPZ, PLY, or depth outputs.",
    )
    return parser.parse_args()


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertices = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return output
    values = depth[valid]
    low, high = np.percentile(values, [2.0, 98.0])
    if high <= low:
        high = low + 1e-6
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip((depth[valid] - low) / (high - low) * 255.0, 0, 255).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    output[valid] = colored[valid]
    return output


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    source_run = (args.source_run or run_dir).resolve()
    default_name = "ego_pointcloud_dense" if args.dense else "ego_pointcloud"
    out_dir = (args.out or run_dir / default_name).resolve()
    trajectory_path = run_dir / "camera_trajectory.npz"
    pointcloud_dir = source_run / "pointcloud"
    frame_dir = source_run / "frames"
    depth_dir = source_run / "depth"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {trajectory_path}")
    if not args.dense and not pointcloud_dir.is_dir():
        raise FileNotFoundError(f"Missing point-cloud directory: {pointcloud_dir}")
    if args.dense and (not frame_dir.is_dir() or not depth_dir.is_dir()):
        raise FileNotFoundError("Dense mode requires the run directory's frames/ and depth/ folders")

    trajectory = np.load(trajectory_path)
    frame_indices = trajectory["frame_index"].astype(np.int64)
    transforms = trajectory["transform_exo_to_camera"].astype(np.float64)
    camera_to_exo = trajectory["transform_camera_to_exo"].astype(np.float64)
    exo_intrinsics = trajectory["intrinsics"].astype(np.float64)
    ego_intrinsics = (
        np.load(args.ego_intrinsics).astype(np.float64)
        if args.ego_intrinsics is not None
        else trajectory["ego_intrinsics"].astype(np.float64)
        if "ego_intrinsics" in trajectory.files
        else exo_intrinsics.copy()
    )
    if ego_intrinsics.shape != (3, 3):
        raise ValueError("ego intrinsics must have shape (3, 3)")
    if transforms.shape != (len(frame_indices), 4, 4):
        raise ValueError("trajectory transform count does not match frame indices")

    directory_names = (
        ("projection_rgb",)
        if args.projection_only
        else (
            "frames",
            "ply_all",
            "ply_visible",
            "projection_rgb",
            "projection_depth",
            "projection_depth_vis",
        )
    )
    directories = {name: out_dir / name for name in directory_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "ego_intrinsics.npy", ego_intrinsics)
    (out_dir / "ego_intrinsics.json").write_text(
        json.dumps(
            {
                "width": args.width,
                "height": args.height,
                "matrix": ego_intrinsics.tolist(),
                "source": "user-provided"
                if args.ego_intrinsics is not None
                else "camera trajectory ego intrinsics"
                if "ego_intrinsics" in trajectory.files
                else "copied from exo intrinsics",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    fps = args.fps
    if fps is None:
        timestamps = trajectory["timestamp_seconds"]
        fps = float(1.0 / np.median(np.diff(timestamps))) if len(timestamps) > 1 else 30.0
    video_path = out_dir / "ego_pointcloud_projection.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    rows: list[dict[str, object]] = []
    roundtrip_max = 0.0
    try:
        for sequence_index, frame_index in enumerate(frame_indices):
            stem = f"frame_{int(frame_index):06d}"
            if args.dense:
                frame_path = frame_dir / f"{stem}.png"
                depth_path = depth_dir / f"{stem}.npz"
                bgr = cv2.imread(str(frame_path))
                if bgr is None:
                    raise FileNotFoundError(f"Missing RGB frame: {frame_path}")
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing depth: {depth_path}")
                depth_source = np.load(depth_path)
                depth_exo = depth_source["depth"].astype(np.float32)
                rgb_source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                points_exo, colors, source_pixels = backproject_pointcloud(
                    rgb_source, depth_exo, exo_intrinsics, stride=1
                )
            else:
                source_path = pointcloud_dir / f"{stem}.npz"
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing point cloud: {source_path}")
                source = np.load(source_path)
                points_exo = source["points_xyz"].astype(np.float32)
                colors = source["colors_rgb"].astype(np.uint8)
                source_pixels = source["pixels_uv"]
            points_ego = transform_points(points_exo, transforms[sequence_index]).astype(np.float32)
            frustum_mask, visible_mask, rgb, depth = project_pointcloud_zbuffer(
                points_ego,
                colors,
                ego_intrinsics,
                args.width,
                args.height,
                args.near,
            )

            recovered = transform_points(points_ego, camera_to_exo[sequence_index])
            frame_roundtrip = float(np.max(np.abs(recovered - points_exo)))
            roundtrip_max = max(roundtrip_max, frame_roundtrip)
            if not args.projection_only:
                np.savez_compressed(
                    directories["frames"] / f"{stem}.npz",
                    points_xyz_ego=points_ego,
                    colors_rgb=colors,
                    source_pixels_exo=source_pixels,
                    frustum_mask=frustum_mask,
                    visible_mask=visible_mask,
                    transform_exo_to_ego=transforms[sequence_index],
                    ego_intrinsics=ego_intrinsics,
                )
                write_binary_ply(directories["ply_all"] / f"{stem}.ply", points_ego, colors)
                write_binary_ply(
                    directories["ply_visible"] / f"{stem}.ply",
                    points_ego[visible_mask],
                    colors[visible_mask],
                )
            cv2.imwrite(
                str(directories["projection_rgb"] / f"{stem}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            if not args.projection_only:
                np.savez_compressed(directories["projection_depth"] / f"{stem}.npz", depth=depth)
                depth_vis = colorize_depth(depth)
                cv2.imwrite(str(directories["projection_depth_vis"] / f"{stem}.png"), depth_vis)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            rows.append(
                {
                    "frame_index": int(frame_index),
                    "total_points": int(len(points_ego)),
                    "points_in_front": int(np.count_nonzero(points_ego[:, 2] > args.near)),
                    "points_in_ego_frustum": int(frustum_mask.sum()),
                    "zbuffer_visible_points": int(visible_mask.sum()),
                    "roundtrip_max_abs_error": frame_roundtrip,
                }
            )
            print(
                f"[{sequence_index + 1:03d}/{len(frame_indices):03d}] {stem}: "
                f"frustum={int(frustum_mask.sum()):5d}, visible={int(visible_mask.sum()):5d}"
            )
    finally:
        writer.release()

    with (out_dir / "frame_statistics.csv").open("w", newline="", encoding="ascii") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    frustum_counts = np.array([row["points_in_ego_frustum"] for row in rows], dtype=np.int64)
    visible_counts = np.array([row["zbuffer_visible_points"] for row in rows], dtype=np.int64)
    summary = {
        "source_run_directory": str(run_dir),
        "source_data_directory": str(source_run),
        "source_mode": "full-resolution RGB/depth backprojection" if args.dense else "saved point cloud",
        "source_stride": 1 if args.dense else 4,
        "projection_only": args.projection_only,
        "frame_count": len(rows),
        "coordinate_convention": "OpenCV ego camera: +X right, +Y down, +Z gaze/forward",
        "backprojection_equation": "p_exo = depth * inv(K_exo) @ [u, v, 1]",
        "transform_equation": "p_ego = R_camera_to_exo.T @ (p_exo - C_ego)",
        "homogeneous_equation": "[p_ego;1] = T_exo_to_camera @ [p_exo;1]",
        "projection_equation": "pixel_ego ~ K_ego @ p_ego, valid when z_ego > near",
        "ego_intrinsics": ego_intrinsics.tolist(),
        "image_size": [args.width, args.height],
        "near": args.near,
        "frustum_points": {
            "min": int(frustum_counts.min()),
            "max": int(frustum_counts.max()),
            "mean": float(frustum_counts.mean()),
        },
        "zbuffer_visible_points": {
            "min": int(visible_counts.min()),
            "max": int(visible_counts.max()),
            "mean": float(visible_counts.mean()),
        },
        "roundtrip_max_abs_error": roundtrip_max,
        "scale_note": "Coordinates retain the monocular MoGe depth scale and are not calibrated metric units.",
    }
    (out_dir / "transform_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    output_kind = "ego projections" if args.projection_only else "ego point clouds"
    print(f"Wrote {len(rows)} {output_kind} to {out_dir}")


if __name__ == "__main__":
    main()
