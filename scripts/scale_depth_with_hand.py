#!/usr/bin/env python3
"""Compute depth scale alignment using hand depth and exocentric depth."""

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def backproject_rgbd_to_pointcloud(image_bgr, depth, fx=None, fy=None, cx=None, cy=None, stride=1):
    height, width = depth.shape
    if fx is None:
        fx = float(max(width, height))
    if fy is None:
        fy = float(max(width, height))
    if cx is None:
        cx = float(width) / 2.0
    if cy is None:
        cy = float(height) / 2.0

    u = np.arange(0, width, stride, dtype=np.float32)
    v = np.arange(0, height, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth.astype(np.float32)
    valid = np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = image_bgr.reshape(-1, 3).astype(np.uint8)
    valid_flat = valid.reshape(-1)
    points = points[valid_flat]
    colors = colors[valid_flat][:, ::-1] if colors.shape[1] == 3 else colors[valid_flat]
    return points, colors


def save_pointcloud_ply(points, colors, out_path):
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Points and colors must have the same number of rows.")
    num_points = points.shape[0]
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
    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for point, color in zip(points, colors):
            x, y, z = point.tolist()
            r, g, b = [int(c) for c in color]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Scale exocentric depth using hand depth alignment")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--depth", required=True, help="Path to exocentric raw depth npy")
    parser.add_argument("--hand_depth", required=True, help="Path to rendered hand depth npy")
    parser.add_argument("--hand_mask", required=True, help="Path to rendered hand mask PNG")
    parser.add_argument("--out", required=True, help="Output directory for scaled depth results")
    parser.add_argument("--K", default=None, help="Optional camera intrinsics .npy for scaled point cloud export")
    parser.add_argument("--erode_kernel", type=int, default=5, help="Erosion kernel size for hand mask cleanup")
    parser.add_argument(
        "--erode_min_ratio",
        type=float,
        default=0.2,
        help="Minimum ratio of eroded valid pixels required to use eroded mask",
    )
    return parser.parse_args()


def load_npy(path):
    return np.load(str(path))


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Hand mask not found: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def normalize_depth_vis(depth_map):
    vis = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid):
        return vis
    valid_depths = depth_map[valid]
    lo = float(np.percentile(valid_depths, 2))
    hi = float(np.percentile(valid_depths, 98))
    if hi <= lo:
        lo = float(np.min(valid_depths))
        hi = float(np.max(valid_depths))
    clipped = np.clip(depth_map, lo, hi)
    normalized = np.zeros_like(depth_map, dtype=np.uint8)
    normalized[valid] = ((clipped[valid] - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    vis[valid] = color[valid]
    return vis


def build_valid_mask(depth_exo, depth_hand, hand_mask):
    valid_exo = np.isfinite(depth_exo) & (depth_exo > 0)
    valid_hand = np.isfinite(depth_hand) & (depth_hand > 0)
    valid_mask = (hand_mask > 0) & valid_exo & valid_hand
    return valid_mask


def compute_ratio_stats(ratios):
    if ratios.size == 0:
        return {}
    p01, p05, p50, p95, p99 = np.percentile(ratios, [1, 5, 50, 95, 99])
    return {
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def save_histogram(ratios, output_path):
    if not MATPLOTLIB_AVAILABLE:
        return False
    if ratios.size == 0:
        return False
    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(ratios, bins=80, color="#4C72B0", edgecolor="black")
    plt.title("Hand-to-exo depth scale ratio")
    plt.xlabel("ratio")
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    return True


def make_pointcloud(image_bgr, depth_scaled, output_path, K_exo=None):
    height, width = depth_scaled.shape
    if K_exo is not None and K_exo.shape == (3, 3):
        fx = float(K_exo[0, 0])
        fy = float(K_exo[1, 1])
        cx = float(K_exo[0, 2])
        cy = float(K_exo[1, 2])
    else:
        # Fallback if K not provided
        fx = float(max(width, height))
        fy = float(max(width, height))
        cx = float(width) / 2.0
        cy = float(height) / 2.0
    points_xyz, colors_rgb = backproject_rgbd_to_pointcloud(
        image_bgr,
        depth_scaled,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    save_pointcloud_ply(points_xyz, colors_rgb, str(output_path))


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / args.image
    depth_path = project_root / args.depth
    hand_depth_path = project_root / args.hand_depth
    hand_mask_path = project_root / args.hand_mask
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    depth_exo = load_npy(depth_path)
    depth_hand = load_npy(hand_depth_path)
    hand_mask = load_mask(hand_mask_path)

    if depth_exo.ndim != 2 or depth_hand.ndim != 2:
        raise ValueError("Depth inputs must be 2D arrays.")
    if depth_exo.shape != depth_hand.shape:
        raise ValueError(f"D_exo shape {depth_exo.shape} does not match D_hand shape {depth_hand.shape}")
    if hand_mask.shape != depth_exo.shape:
        raise ValueError(f"hand_mask shape {hand_mask.shape} does not match depth shape {depth_exo.shape}")

    original_valid_mask = build_valid_mask(depth_exo, depth_hand, hand_mask)
    original_valid_count = int(np.sum(original_valid_mask))
    total_pixels = depth_exo.size

    if original_valid_count == 0:
        raise ValueError("No valid hand alignment pixels found after initial masking.")

    hand_mask_bin = (hand_mask > 0).astype(np.uint8)
    kernel_size = max(1, args.erode_kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded_mask = cv2.erode(hand_mask_bin, kernel, iterations=1) > 0
    eroded_valid_mask = build_valid_mask(depth_exo, depth_hand, eroded_mask)
    eroded_valid_count = int(np.sum(eroded_valid_mask))
    use_eroded = False
    if eroded_valid_count >= max(100, int(args.erode_min_ratio * original_valid_count)):
        use_eroded = True
        valid_mask = eroded_valid_mask
    else:
        valid_mask = original_valid_mask

    valid_indices = np.where(valid_mask)
    depth_exo_valid = depth_exo[valid_mask]
    depth_hand_valid = depth_hand[valid_mask]
    eps = 1e-6
    ratio = depth_hand_valid / (depth_exo_valid + eps)
    ratio = ratio[np.isfinite(ratio)]
    ratio = ratio[ratio > 0]
    ratio = ratio[(ratio >= 0.05) & (ratio <= 20.0)]

    if ratio.size == 0:
        raise ValueError("No valid ratio values remain after filtering.")

    stats = compute_ratio_stats(ratio)
    if not stats:
        raise ValueError("Unable to compute ratio statistics.")

    ratio_range_mask = (ratio >= stats["p05"]) & (ratio <= stats["p95"])
    ratio_filtered = ratio[ratio_range_mask]
    if ratio_filtered.size == 0:
        ratio_filtered = ratio
    s_star = float(np.median(ratio_filtered))

    depth_scaled = depth_exo.astype(np.float32) * s_star
    depth_scaled_path = out_dir / "depth_scaled.npy"
    np.save(str(depth_scaled_path), depth_scaled)

    depth_scaled_vis = normalize_depth_vis(depth_scaled)
    depth_scaled_vis_path = out_dir / "depth_scaled_vis.png"
    cv2.imwrite(str(depth_scaled_vis_path), depth_scaled_vis)

    scale_valid_mask = (valid_mask.astype(np.uint8) * 255)
    scale_valid_mask_path = out_dir / "scale_valid_mask.png"
    cv2.imwrite(str(scale_valid_mask_path), scale_valid_mask)

    hist_path = out_dir / "scale_ratio_hist.png"
    hist_saved = False
    if MATPLOTLIB_AVAILABLE:
        try:
            hist_saved = save_histogram(ratio, hist_path)
        except Exception:
            hist_saved = False

    # Load camera intrinsics from original depth estimation if available
    K_exo = None
    K_exo_path = project_root / args.K if args.K is not None else project_root / "outputs" / "K_exo.npy"
    if K_exo_path.exists():
        try:
            K_exo = np.load(str(K_exo_path))
        except Exception as e:
            print(f"Warning: could not load K_exo: {e}")

    ply_path = out_dir / "exo_point_cloud_scaled.ply"
    make_pointcloud(rgb_bgr, depth_scaled, ply_path, K_exo=K_exo)

    valid_ratio_count = ratio.size
    valid_ratio_filtered_count = ratio_filtered.size
    min_ratio = float(np.min(ratio))
    max_ratio = float(np.max(ratio))
    warnings = []
    if original_valid_count < 500:
        warnings.append("Low original valid pixel count: results may be noisy.")
    if valid_ratio_count < 500:
        warnings.append("Low final ratio count: scale estimate may be unstable.")
    if stats["p95"] / max(stats["p05"], 1e-6) > 10.0:
        warnings.append("Large ratio spread detected: consider checking alignment quality.")

    report_lines = [
        "# Depth Scale Alignment Report",
        "",
        f"Image: {image_path}",
        f"Depth raw: {depth_path}",
        f"Hand depth: {hand_depth_path}",
        f"Hand mask: {hand_mask_path}",
        "",
        f"D_exo shape: {depth_exo.shape}",
        f"D_hand shape: {depth_hand.shape}",
        f"Original valid pixels: {original_valid_count}",
        f"Eroded valid pixels: {eroded_valid_count}",
        f"Using eroded mask for scale computation: {use_eroded}",
        "",
        "# Ratio statistics",
        f"Valid ratio count: {valid_ratio_count}",
        f"Filtered ratio count (p05-p95): {valid_ratio_filtered_count}",
        f"ratio p01: {stats['p01']:.4f}",
        f"ratio p05: {stats['p05']:.4f}",
        f"ratio p50: {stats['p50']:.4f}",
        f"ratio p95: {stats['p95']:.4f}",
        f"ratio p99: {stats['p99']:.4f}",
        "",
        f"Final scale s_star: {s_star:.6f}",
    ]

    if ratio_filtered.size != ratio.size:
        report_lines.append(f"Used ratio count after p05/p95 filtering: {ratio_filtered.size}")
        report_lines.append("")

    depth_scaled_valid = depth_scaled[valid_mask]
    if np.any(np.isfinite(depth_scaled_valid)):
        report_lines.extend(
            [
                f"Scaled depth p05: {float(np.percentile(depth_scaled_valid, 5)):.4f}",
                f"Scaled depth p50: {float(np.percentile(depth_scaled_valid, 50)):.4f}",
                f"Scaled depth p95: {float(np.percentile(depth_scaled_valid, 95)):.4f}",
            ]
        )
    else:
        report_lines.extend(["Scaled depth p05: nan", "Scaled depth p50: nan", "Scaled depth p95: nan"])

    report_lines.append("")
    report_lines.append(f"depth_scaled.npy: {depth_scaled_path}")
    report_lines.append(f"depth_scaled_vis.png: {depth_scaled_vis_path}")
    report_lines.append(f"scale_valid_mask.png: {scale_valid_mask_path}")
    report_lines.append(f"exo_point_cloud_scaled.ply: {ply_path}")
    if hist_saved:
        report_lines.append(f"scale_ratio_hist.png: {hist_path}")
    else:
        report_lines.append("scale_ratio_hist.png: matplotlib unavailable or save failed")
    if warnings:
        report_lines.append("")
        report_lines.append("# Warnings")
        for warning in warnings:
            report_lines.append(f"- {warning}")

    report_path = out_dir / "scale_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved scaled depth: {depth_scaled_path}")
    print(f"Saved scaled depth visualization: {depth_scaled_vis_path}")
    print(f"Saved scale valid mask: {scale_valid_mask_path}")
    print(f"Saved scaled point cloud: {ply_path}")
    if hist_saved:
        print(f"Saved scale ratio histogram: {hist_path}")
    print(f"Saved scale report: {report_path}")


if __name__ == "__main__":
    main()
