import os

import cv2
import numpy as np
from PIL import Image


def load_rgb_image(image_path: str) -> np.ndarray:
    """Load an RGB image and return a NumPy array."""
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """Return float32 depth with invalid values set to 0 for downstream files."""
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth after squeezing, got shape {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("Depth has no positive finite values.")
    sanitized = depth.copy()
    sanitized[~valid] = 0.0
    return sanitized


def save_depth_npy(depth: np.ndarray, output_path: str) -> None:
    """Save finite raw depth values to a NumPy file."""
    np.save(output_path, sanitize_depth(depth))


def save_depth_vis(depth: np.ndarray, output_path: str) -> None:
    """Save visual depth maps with inf/nan handling and percentile normalization.

    Saves three files:
    - depth_gray.png: grayscale depth
    - depth_vis.png: turbo colormap visualization
    - depth_mask.png: binary mask of valid pixels
    """
    depth = sanitize_depth(depth)

    # Create valid mask: finite and positive
    valid = depth > 0
    valid_count = np.sum(valid)

    print(f"Depth shape: {depth.shape}")
    print(f"Valid pixel count: {valid_count} / {depth.size}")

    # Compute statistics on valid pixels only
    valid_depths = depth[valid]
    depth_min_finite = float(np.min(valid_depths))
    depth_max_finite = float(np.max(valid_depths))
    p2 = float(np.percentile(valid_depths, 2))
    p50 = float(np.percentile(valid_depths, 50))
    p98 = float(np.percentile(valid_depths, 98))

    print(f"Finite min: {depth_min_finite:.4f}, max: {depth_max_finite:.4f}")
    print(f"Percentiles - 2%: {p2:.4f}, 50%: {p50:.4f}, 98%: {p98:.4f}")

    # Normalize using 2nd-98th percentile range
    lo = p2
    hi = p98
    if hi <= lo:
        print("Warning: percentile range [lo, hi] is invalid; using min/max.")
        lo = depth_min_finite
        hi = depth_max_finite
    if hi <= lo:
        raise ValueError("Depth normalization range is invalid.")

    # Clip and normalize to [0, 255]
    depth_clipped = np.clip(depth, lo, hi)
    normalized = ((depth_clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    # Set invalid pixels to black
    normalized[~valid] = 0

    # Create mask: valid=255, invalid=0
    mask = (valid.astype(np.uint8)) * 255

    # Save grayscale depth
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace(".png", "")
    gray_path = os.path.join(output_dir, f"{base_name.replace('_vis', '')}_gray.png")
    Image.fromarray(normalized, mode="L").save(gray_path)

    # Apply colormap and save
    depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    Image.fromarray(depth_color_rgb, mode="RGB").save(output_path)

    # Save mask
    mask_path = os.path.join(output_dir, f"{base_name.replace('_vis', '')}_mask.png")
    Image.fromarray(mask, mode="L").save(mask_path)
