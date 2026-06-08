#!/usr/bin/env python3
"""Create a geometric sparse egocentric RGB map from exocentric observations."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PLY_NUMPY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


DEFAULT_EGO_INTRINSICS = (636.6593017578125, 636.251953125, 635.283881879317, 366.8740353496978, 1280, 720)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sparse ego RGB map.")
    parser.add_argument("--exo_joints", default="outputs/hamer_pose_exo/hamer_joints_exo.npy", type=Path)
    parser.add_argument("--ego_joints", default="outputs/vit_handpose_exo_best/joints_2x21x3.npy", type=Path)
    parser.add_argument("--exo_ply", default="outputs/scaled_depth/exo_point_cloud_scaled.ply", type=Path)
    parser.add_argument("--out", default="outputs/ego_sparse_map", type=Path)
    parser.add_argument("--width", default=int(DEFAULT_EGO_INTRINSICS[4]), type=int)
    parser.add_argument("--height", default=int(DEFAULT_EGO_INTRINSICS[5]), type=int)
    parser.add_argument("--fov", default=90.0, type=float)
    parser.add_argument(
        "--intrinsics",
        nargs=6,
        type=float,
        metavar=("FX", "FY", "CX", "CY", "WIDTH", "HEIGHT"),
        default=DEFAULT_EGO_INTRINSICS,
        help="Ego camera intrinsics as fx fy cx cy width height.",
    )
    parser.add_argument(
        "--intrinsics_file",
        type=Path,
        help="Text file containing fx fy cx cy width height. Overrides --intrinsics.",
    )
    parser.add_argument(
        "--synthetic_intrinsics",
        action="store_true",
        help="Use --width/--height/--fov instead of calibrated intrinsics.",
    )
    parser.add_argument("--splat", default=1, type=int)
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--swap_ego_hands", action="store_true")
    parser.add_argument("--try_both_directions", action="store_true")
    return parser.parse_args()


def unwrap_npy_object(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    if isinstance(value, dict):
        for key in ("joints", "hand_joints", "joints_3d", "pred_joints", "data"):
            if key in value:
                return unwrap_npy_object(value[key])
        arrays = [item for item in value.values() if isinstance(item, (np.ndarray, list, tuple))]
        if len(arrays) == 1:
            return unwrap_npy_object(arrays[0])
        raise ValueError(f"Could not identify joints in dict keys: {sorted(value.keys())}")
    return value


def normalize_joints(path: Path) -> np.ndarray:
    value = unwrap_npy_object(np.load(path, allow_pickle=True))
    joints = np.asarray(value, dtype=np.float64)
    if joints.shape == (21, 3):
        return joints[None, ...]
    if joints.shape == (42, 3):
        return joints.reshape(2, 21, 3)
    if joints.ndim == 3 and joints.shape[1:] == (21, 3):
        return joints
    if joints.size in (63, 126):
        return joints.reshape(-1, 21, 3)
    raise ValueError(f"Unsupported joint shape {joints.shape} from {path}")


def paired_valid_correspondences(ego: np.ndarray, exo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand_count = min(len(ego), len(exo))
    if len(ego) != len(exo):
        print(f"WARNING: hand count differs; using first {hand_count} paired hand(s).")
    src = ego[:hand_count].reshape(-1, 3)
    dst = exo[:hand_count].reshape(-1, 3)
    valid = (
        np.isfinite(src).all(axis=1)
        & np.isfinite(dst).all(axis=1)
        & (np.linalg.norm(src, axis=1) > 1e-12)
        & (np.linalg.norm(dst, axis=1) > 1e-12)
    )
    return src[valid], dst[valid]


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    if len(src) < 6:
        raise ValueError(f"At least 6 valid correspondences are required, got {len(src)}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    if estimate_scale:
        src_variance = np.mean(np.sum(src_centered * src_centered, axis=1))
        if src_variance <= 1e-15:
            raise ValueError("Source joints have near-zero variance")
        scale = float(np.sum(singular_values * np.diag(correction)) / src_variance)
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return scale, rotation, translation, transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def read_ply_header(stream) -> tuple[str, int, list[tuple[str, str]]]:
    first = stream.readline().decode("ascii").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")
    ply_format = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected EOF in PLY header")
        text = line.decode("ascii").strip()
        parts = text.split()
        if not parts:
            continue
        if parts[0] == "format":
            ply_format = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List properties are not supported in the vertex element")
            vertex_properties.append((parts[2], parts[1]))
        elif parts[0] == "end_header":
            break
    return ply_format, vertex_count, vertex_properties


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData  # type: ignore

        vertex = PlyData.read(str(path))["vertex"].data
        names = set(vertex.dtype.names or ())
        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
        if {"red", "green", "blue"} <= names:
            colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 255, dtype=np.uint8)
        return points, colors
    except ImportError:
        pass

    with path.open("rb") as stream:
        ply_format, vertex_count, properties = read_ply_header(stream)
        names = [name for name, _ in properties]
        required = [names.index(axis) for axis in ("x", "y", "z")]
        color_indices = [names.index(channel) for channel in ("red", "green", "blue")] if all(
            channel in names for channel in ("red", "green", "blue")
        ) else None

        if ply_format == "ascii":
            values = np.loadtxt(stream, dtype=np.float64, max_rows=vertex_count)
            points = values[:, required].astype(np.float64, copy=False)
            colors = (
                np.clip(values[:, color_indices], 0, 255).astype(np.uint8)
                if color_indices is not None
                else np.full((vertex_count, 3), 255, dtype=np.uint8)
            )
            return points, colors

        if ply_format not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {ply_format}")
        endian = "<" if ply_format == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + PLY_NUMPY_TYPES[data_type]) for name, data_type in properties])
        vertex = np.fromfile(stream, dtype=dtype, count=vertex_count)
        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
        colors = (
            np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
            if color_indices is not None
            else np.full((vertex_count, 3), 255, dtype=np.uint8)
        )
        return points, colors


def save_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite].astype(np.float32, copy=False)
    colors = colors[finite].astype(np.uint8, copy=False)
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertex.tofile(stream)


def synthetic_intrinsics(width: int, height: int, fov_degrees: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if not 0.0 < fov_degrees < 180.0:
        raise ValueError("FOV must be between 0 and 180 degrees")
    focal = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def calibrated_intrinsics(values: tuple[float, ...] | list[float]) -> tuple[np.ndarray, int, int]:
    if len(values) != 6:
        raise ValueError(f"Expected 6 intrinsic values, got {len(values)}")
    fx, fy, cx, cy, width, height = values
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Intrinsic width and height must be positive")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Focal lengths must be positive")
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return intrinsic, width, height


def load_intrinsics_file(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8").strip()
    values = [float(item) for item in text.split()]
    if len(values) != 6:
        raise ValueError(f"Expected fx fy cx cy width height in {path}, got {len(values)} values")
    return values


def render_sparse(
    points: np.ndarray,
    colors: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    splat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    positive = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    camera_points = points[positive]
    camera_colors = colors[positive]
    positive_count = len(camera_points)
    depth = camera_points[:, 2]
    u = np.rint(intrinsic[0, 0] * camera_points[:, 0] / depth + intrinsic[0, 2]).astype(np.int64)
    v = np.rint(intrinsic[1, 1] * camera_points[:, 1] / depth + intrinsic[1, 2]).astype(np.int64)

    zbuffer = np.full(width * height, np.inf, dtype=np.float32)
    rgb_flat = np.zeros((width * height, 3), dtype=np.uint8)
    splat = max(0, int(splat))
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            uu = u + dx
            vv = v + dy
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not np.any(inside):
                continue
            indices = vv[inside] * width + uu[inside]
            candidate_depth = depth[inside].astype(np.float32, copy=False)
            candidate_colors = camera_colors[inside]
            np.minimum.at(zbuffer, indices, candidate_depth)
            winners = candidate_depth <= zbuffer[indices] + 1e-7
            rgb_flat[indices[winners]] = candidate_colors[winners]

    mask_flat = np.isfinite(zbuffer)
    depth_flat = zbuffer.copy()
    depth_flat[~mask_flat] = 0.0
    return (
        rgb_flat.reshape(height, width, 3),
        mask_flat.reshape(height, width),
        depth_flat.reshape(height, width),
        positive_count,
        int(mask_flat.sum()),
    )


def save_render(out_dir: Path, stem: str, rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, save_arrays: bool) -> None:
    Image.fromarray(rgb, mode="RGB").save(out_dir / f"{stem}.png")
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_dir / f"{stem.replace('rgb', 'mask')}.png")
    if save_arrays:
        np.save(out_dir / f"{stem}.npy", rgb)
        np.save(out_dir / f"{stem.replace('rgb', 'mask')}.npy", mask)
        np.save(out_dir / f"{stem.replace('rgb', 'depth')}.npy", depth)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exo_joints = normalize_joints(args.exo_joints)
    ego_joints = normalize_joints(args.ego_joints)
    print("Loaded exo joint shape:", exo_joints.shape)
    print("Loaded ego joint shape:", ego_joints.shape)
    if args.swap_ego_hands and len(ego_joints) == 2:
        ego_joints = ego_joints[::-1].copy()
        print("Swapped ego hand order.")

    src, dst = paired_valid_correspondences(ego_joints, exo_joints)
    print("Number of valid correspondences:", len(src))
    scale, rotation, translation, ego_to_exo = umeyama(src, dst, estimate_scale=not args.no_scale)
    exo_to_ego = np.linalg.inv(ego_to_exo)
    before_error = float(np.linalg.norm(dst - src, axis=1).mean())
    aligned = transform_points(src, ego_to_exo)
    after_error = float(np.linalg.norm(dst - aligned, axis=1).mean())
    print("Estimated scale:", scale)
    print("Rotation determinant:", float(np.linalg.det(rotation)))
    print("Translation:", translation.tolist())
    print("Mean alignment error before transform:", before_error)
    print("Mean alignment error after transform:", after_error)
    exo_back_to_ego = transform_points(dst, exo_to_ego)
    exo_to_ego_hand_error = float(np.linalg.norm(src - exo_back_to_ego, axis=1).mean())
    print("Mean exo hand error after T_exo_to_ego:", exo_to_ego_hand_error)
    if scale < 0.7 or scale > 1.3:
        print("WARNING: Scale is far from 1; units, root-relative coordinates, or left/right hand order may be wrong.")

    np.save(args.out / "T_ego_to_exo.npy", ego_to_exo)
    np.save(args.out / "T_exo_to_ego.npy", exo_to_ego)
    if args.synthetic_intrinsics:
        intrinsic = synthetic_intrinsics(args.width, args.height, args.fov)
        intrinsic_mode = "synthetic_fov"
    else:
        intrinsic_values = load_intrinsics_file(args.intrinsics_file) if args.intrinsics_file else args.intrinsics
        intrinsic, args.width, args.height = calibrated_intrinsics(intrinsic_values)
        intrinsic_mode = "calibrated"
    print("Ego intrinsics mode:", intrinsic_mode)
    print("Ego image size:", [args.width, args.height])
    print("K_ego:", intrinsic.tolist())
    np.save(args.out / "K_ego.npy", intrinsic)

    exo_points, colors = load_ply(args.exo_ply)
    print("Number of input point cloud points:", len(exo_points))
    ego_points = transform_points(exo_points, exo_to_ego)
    save_binary_ply(args.out / "ego_point_cloud_scaled.ply", ego_points, colors)

    rgb, mask, depth, positive_count, valid_pixels = render_sparse(
        ego_points, colors, intrinsic, args.width, args.height, args.splat
    )
    save_render(args.out, "ego_sparse_rgb", rgb, mask, depth, save_arrays=True)
    print("Number of ego points with positive z:", positive_count)
    print("Number of projected valid pixels:", valid_pixels)
    mask_ratio = valid_pixels / float(args.width * args.height)
    print("Sparse mask ratio:", mask_ratio)
    if positive_count < max(1, int(0.05 * len(ego_points))):
        print("WARNING: Almost all transformed ego points have negative z; transform direction or coordinate convention may be wrong.")
    if valid_pixels == 0:
        print("WARNING: No valid projected pixels. Try the inverse transform direction, check root-relative P_ego,")
        print("         check left/right ordering, or adjust FOV.")

    direction_debug = None
    if args.try_both_directions:
        diagnostic_points = transform_points(exo_points, ego_to_exo)
        debug_rgb, debug_mask, debug_depth, debug_positive, debug_pixels = render_sparse(
            diagnostic_points, colors, intrinsic, args.width, args.height, args.splat
        )
        Image.fromarray(debug_rgb, mode="RGB").save(args.out / "ego_sparse_rgb_wrong_direction_debug.png")
        direction_debug = {
            "T_exo_to_ego_valid_pixels": valid_pixels,
            "T_ego_to_exo_direct_valid_pixels": debug_pixels,
            "T_ego_to_exo_direct_positive_points": debug_positive,
        }
        winner = "T_exo_to_ego" if valid_pixels >= debug_pixels else "T_ego_to_exo direct diagnostic"
        print("Direction diagnostic valid pixels:", direction_debug)
        print("Direction with more valid projected pixels:", winner)

    report = {
        "exo_joints": str(args.exo_joints),
        "ego_joints": str(args.ego_joints),
        "exo_ply": str(args.exo_ply),
        "valid_correspondences": len(src),
        "scale": scale,
        "rotation_determinant": float(np.linalg.det(rotation)),
        "translation": translation.tolist(),
        "alignment_error_before": before_error,
        "alignment_error_after": after_error,
        "exo_to_ego_hand_error": exo_to_ego_hand_error,
        "intrinsic_mode": intrinsic_mode,
        "K_ego": intrinsic.tolist(),
        "input_point_count": len(exo_points),
        "positive_z_point_count": positive_count,
        "projected_valid_pixels": valid_pixels,
        "sparse_mask_ratio": mask_ratio,
        "width": args.width,
        "height": args.height,
        "fov": args.fov,
        "splat": args.splat,
        "no_scale": args.no_scale,
        "swap_ego_hands": args.swap_ego_hands,
        "direction_debug": direction_debug,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved outputs to:", args.out)


if __name__ == "__main__":
    main()
