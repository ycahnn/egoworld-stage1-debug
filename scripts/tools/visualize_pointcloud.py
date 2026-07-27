import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a point cloud from a PLY file with RGB colors.")
    parser.add_argument("--ply", required=True, help="Path to the PLY point cloud file.")
    return parser.parse_args()


def visualize_with_matplotlib(points, colors) -> None:
    """Fallback visualization using matplotlib."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if len(points) > 100000:
        indices = np.random.choice(len(points), 100000, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices]
    else:
        points_vis = points
        colors_vis = colors

    ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
               c=colors_vis, s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point Cloud ({len(points)} points)')

    print("Backend: matplotlib fallback")
    print("Displaying matplotlib visualization (close the window to exit)...")
    plt.show()


def main() -> None:
    args = parse_args()
    print(f"Loading point cloud from {args.ply}...")

    try:
        import open3d as o3d
        print("Backend: Open3D")
        pcd = o3d.io.read_point_cloud(args.ply)
        if pcd.is_empty():
            print(f"Warning: Loaded point cloud is empty from {args.ply}")
        print(f"Loaded points: {len(pcd.points)}")
        print(f"Colors available: {pcd.has_colors()}")
        o3d.visualization.draw_geometries([pcd])
        return
    except ImportError:
        print("Open3D import failed; using matplotlib fallback.")
    except Exception as exc:
        print(f"Open3D failed to load or visualize the PLY file: {exc}")
        print("Falling back to matplotlib visualization.")

    try:
        points, colors = load_ply(args.ply)
    except FileNotFoundError:
        print(f"Error: File not found: {args.ply}")
        return
    except Exception as e:
        print(f"Error loading PLY file for fallback visualization: {e}")
        return

    print(f"Loaded points: {len(points)}")
    print(f"Colors available: {len(colors) > 0}")
    visualize_with_matplotlib(points, colors)


def load_ply(file_path: str):
    """Load a PLY file and return (points, colors)."""
    import numpy as np

    points = []
    colors = []
    with open(file_path, 'r') as f:
        header_done = False
        for line in f:
            line = line.strip()
            if line == 'end_header':
                header_done = True
                break

        if not header_done:
            raise ValueError("Invalid PLY file: no end_header found")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


if __name__ == "__main__":
    main()

