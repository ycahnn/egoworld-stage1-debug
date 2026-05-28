import numpy as np


def backproject_rgbd_to_pointcloud(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Backproject an RGB-D image into a 3D point cloud, filtering invalid depth pixels.

    Args:
        rgb: RGB image array with shape (H, W, 3).
        depth: Depth map array with shape (H, W).
        K: Camera intrinsic matrix shape (3, 3).

    Returns:
        points_xyz: Array of 3D points shape (N, 3), excluding invalid pixels.
        colors_rgb: Array of uint8 colors shape (N, 3), excluding invalid pixels.
    """
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB image and depth map must have the same height and width.")

    height, width = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    Z = depth.astype(np.float32)
    
    # Create valid mask: finite and positive depth
    valid = np.isfinite(Z) & (Z > 0)
    valid_count = np.sum(valid)
    print(f"Backprojection: {valid_count} / {Z.size} valid depth pixels")
    
    # Compute X, Y using all pixels
    X = (uu.astype(np.float32) - cx) * Z / fx
    Y = (vv.astype(np.float32) - cy) * Z / fy

    # Filter to valid pixels only
    points_xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors_rgb = rgb.reshape(-1, 3).astype(np.uint8)
    
    valid_flat = valid.reshape(-1)
    points_xyz = points_xyz[valid_flat]
    colors_rgb = colors_rgb[valid_flat]
    
    if points_xyz.shape[0] == 0:
        raise ValueError("Point cloud has zero valid points; depth must contain positive finite values.")

    print(f"Saved {points_xyz.shape[0]} points to point cloud")
    return points_xyz, colors_rgb


def save_pointcloud_ply(points_xyz: np.ndarray, colors_rgb: np.ndarray, output_path: str) -> None:
    """Save a point cloud to a simple ASCII PLY file."""
    if points_xyz.shape[0] != colors_rgb.shape[0]:
        raise ValueError("Number of points and colors must match.")

    num_points = points_xyz.shape[0]
    if num_points == 0:
        raise ValueError("Refusing to write an empty point cloud PLY.")
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for point, color in zip(points_xyz, colors_rgb):
            x, y, z = point.tolist()
            r, g, b = color.tolist()
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
