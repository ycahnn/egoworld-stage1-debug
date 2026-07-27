import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_estimator import DepthEstimator
from src.io_utils import load_rgb_image, save_depth_npy, save_depth_vis
from src.pointcloud import backproject_rgbd_to_pointcloud, save_pointcloud_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer depth and save an RGB-D point cloud from a single exocentric image.")
    parser.add_argument("--image", required=True, help="Path to the input exocentric RGB image.")
    parser.add_argument("--out", required=True, help="Output directory for depth and point cloud files.")
    parser.add_argument(
        "--depth_model",
        choices=["dummy", "moge"],
        default="dummy",
        help="Depth model to use: dummy for the built-in gradient fallback, moge to use an installed MoGe model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    estimator = DepthEstimator(mode=args.depth_model)
    depth, K = estimator.predict(args.image)

    depth_raw_path = os.path.join(args.out, "depth_raw.npy")
    depth_vis_path = os.path.join(args.out, "depth_vis.png")
    ply_path = os.path.join(args.out, "exo_point_cloud.ply")
    k_npy_path = os.path.join(args.out, "K_exo.npy")
    k_json_path = os.path.join(args.out, "depth_intrinsics.json")

    save_depth_npy(depth, depth_raw_path)
    print(f"\nGenerating depth visualization...")
    save_depth_vis(depth, depth_vis_path)

    rgb = load_rgb_image(args.image)
    print(f"Backprojecting RGB-D to point cloud...")
    points_xyz, colors_rgb = backproject_rgbd_to_pointcloud(rgb, depth, K)
    save_pointcloud_ply(points_xyz, colors_rgb, ply_path)

    # Save intrinsics
    import numpy as np
    np.save(k_npy_path, K)
    k_dict = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }
    with open(k_json_path, "w", encoding="utf-8") as f:
        json.dump(k_dict, f, indent=2)

    print(f"Saved raw depth: {depth_raw_path}")
    print(f"Saved depth visualization: {depth_vis_path}")
    print(f"Saved point cloud: {ply_path}")
    print(f"Saved camera intrinsics: {k_npy_path}, {k_json_path}")


if __name__ == "__main__":
    main()
