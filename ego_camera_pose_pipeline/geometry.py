from __future__ import annotations

import math

import numpy as np


def synthetic_intrinsics(width: int, height: int, horizontal_fov_degrees: float = 60.0) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("horizontal FOV must be between 0 and 180 degrees")
    focal = (width * 0.5) / math.tan(math.radians(horizontal_fov_degrees) * 0.5)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def sample_depth_median(
    depth: np.ndarray,
    pixel: np.ndarray,
    radii: tuple[int, ...] = (2, 4, 8, 12),
) -> tuple[float, int]:
    height, width = depth.shape
    u, v = np.asarray(pixel, dtype=np.float64)
    x, y = int(round(float(u))), int(round(float(v)))
    for radius in radii:
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        values = depth[y0:y1, x0:x1]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            return float(np.median(values)), int(radius)
    return float("nan"), -1


def backproject_pixel(pixel: np.ndarray, depth: float, intrinsics: np.ndarray) -> np.ndarray:
    u, v = np.asarray(pixel, dtype=np.float64)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=np.float64)


def gaze3d_to_pointcloud(gaze: np.ndarray) -> np.ndarray:
    """Map Gaze3D screen axes to OpenCV point-cloud axes (X right, Y down, Z forward)."""
    gx, gy, gz = np.asarray(gaze, dtype=np.float64)
    mapped = np.array([-gx, -gy, gz], dtype=np.float64)
    norm = np.linalg.norm(mapped)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid gaze vector: {gaze}")
    return mapped / norm


def weighted_target_direction(
    camera_center: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average target rays so near and far targets contribute by angle, not depth."""
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    targets = np.asarray(target_points, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) == 0:
        raise ValueError("target_points must have shape (N, 3) with N > 0")
    rays = targets - center
    norms = np.linalg.norm(rays, axis=1)
    valid = np.isfinite(rays).all(axis=1) & (norms > 1e-8)
    if not valid.any():
        raise ValueError("target_points do not contain a valid direction")
    rays = rays[valid] / norms[valid, None]
    if weights is None:
        valid_weights = np.ones(len(rays), dtype=np.float64)
    else:
        all_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(all_weights) != len(targets):
            raise ValueError("weights must match target_points")
        valid_weights = all_weights[valid]
    if not np.isfinite(valid_weights).all() or np.any(valid_weights < 0):
        raise ValueError("weights must be finite and non-negative")
    direction = np.sum(rays * valid_weights[:, None], axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("weighted target rays cancel each other")
    return direction / norm


def blend_directions(primary: np.ndarray, secondary: np.ndarray, primary_weight: float) -> np.ndarray:
    """Blend two unit directions, retaining primary_weight of the first direction."""
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("primary_weight must be in [0, 1]")
    first = np.asarray(primary, dtype=np.float64).reshape(3)
    second = np.asarray(secondary, dtype=np.float64).reshape(3)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    blended = primary_weight * first + (1.0 - primary_weight) * second
    norm = np.linalg.norm(blended)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("directions cancel each other")
    return blended / norm


def stabilize_direction(
    previous: np.ndarray,
    current: np.ndarray,
    smoothing: float = 0.8,
    deadzone_degrees: float = 2.0,
    max_step_degrees: float = 3.0,
) -> np.ndarray:
    """Smooth a direction and clamp its angular change from the previous frame."""
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    if deadzone_degrees < 0.0 or max_step_degrees <= 0.0:
        raise ValueError("deadzone must be >= 0 and max step must be > 0")
    old = np.asarray(previous, dtype=np.float64).reshape(3)
    new = np.asarray(current, dtype=np.float64).reshape(3)
    old /= np.linalg.norm(old)
    new /= np.linalg.norm(new)
    raw_angle = np.arccos(np.clip(float(np.dot(old, new)), -1.0, 1.0))
    if raw_angle <= np.deg2rad(deadzone_degrees):
        return old

    smoothed = smoothing * old + (1.0 - smoothing) * new
    smoothed /= np.linalg.norm(smoothed)
    angle = np.arccos(np.clip(float(np.dot(old, smoothed)), -1.0, 1.0))
    maximum = np.deg2rad(max_step_degrees)
    if angle <= maximum:
        return smoothed
    fraction = maximum / angle
    sin_angle = np.sin(angle)
    limited = (
        np.sin((1.0 - fraction) * angle) / sin_angle * old
        + np.sin(fraction * angle) / sin_angle * smoothed
    )
    return limited / np.linalg.norm(limited)


def camera_extrinsics(
    camera_center: np.ndarray,
    forward: np.ndarray,
    camera_y_hint: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build OpenCV-style camera transforms with local +Z forward and local +Y down."""
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    z_axis = np.asarray(forward, dtype=np.float64).reshape(3)
    z_axis /= np.linalg.norm(z_axis)
    y_hint = np.asarray(camera_y_hint, dtype=np.float64).reshape(3)
    y_hint /= np.linalg.norm(y_hint)

    x_axis = np.cross(y_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fallback, z_axis))) > 0.95:
            fallback = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(fallback, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation_camera_to_exo = np.column_stack([x_axis, y_axis, z_axis])
    transform_camera_to_exo = np.eye(4, dtype=np.float64)
    transform_camera_to_exo[:3, :3] = rotation_camera_to_exo
    transform_camera_to_exo[:3, 3] = center

    transform_exo_to_camera = np.eye(4, dtype=np.float64)
    transform_exo_to_camera[:3, :3] = rotation_camera_to_exo.T
    transform_exo_to_camera[:3, 3] = -rotation_camera_to_exo.T @ center
    return rotation_camera_to_exo, transform_camera_to_exo, transform_exo_to_camera


def backproject_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("point-cloud stride must be >= 1")
    height, width = depth.shape
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride].astype(np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (uu.astype(np.float32) - cx) * z / fx
    y = (vv.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb[::stride, ::stride].reshape(-1, 3).astype(np.uint8)
    pixels = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.int32)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    return points[valid], colors[valid], pixels[valid]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to an Nx3 row-vector point array."""
    points = np.asarray(points)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_pointcloud_zbuffer(
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    near: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project an RGB point cloud and return frustum/visible masks, RGB, and depth."""
    points = np.asarray(points)
    colors = np.asarray(colors)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != (points.shape[0], 3):
        raise ValueError("colors must have shape (N, 3)")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    z = points[:, 2]
    candidate = np.isfinite(points).all(axis=1) & (z > near)
    candidate_indices = np.flatnonzero(candidate)
    frustum_mask = np.zeros(points.shape[0], dtype=bool)
    visible_mask = np.zeros(points.shape[0], dtype=bool)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), np.nan, dtype=np.float32)
    if candidate_indices.size == 0:
        return frustum_mask, visible_mask, rgb, depth

    candidate_points = points[candidate_indices]
    projected = candidate_points @ intrinsics.T
    uv = projected[:, :2] / projected[:, 2:3]
    pixels = np.rint(uv).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    frustum_indices = candidate_indices[inside]
    frustum_mask[frustum_indices] = True
    if frustum_indices.size == 0:
        return frustum_mask, visible_mask, rgb, depth

    pixels = pixels[inside]
    frustum_z = points[frustum_indices, 2]
    linear = pixels[:, 1] * width + pixels[:, 0]
    order = np.argsort(frustum_z, kind="stable")
    _, first = np.unique(linear[order], return_index=True)
    selected = order[first]
    visible_indices = frustum_indices[selected]
    visible_pixels = pixels[selected]
    visible_mask[visible_indices] = True
    rgb[visible_pixels[:, 1], visible_pixels[:, 0]] = colors[visible_indices]
    depth[visible_pixels[:, 1], visible_pixels[:, 0]] = points[visible_indices, 2]
    return frustum_mask, visible_mask, rgb, depth
