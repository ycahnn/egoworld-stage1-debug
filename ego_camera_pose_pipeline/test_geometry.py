import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_camera_pose_pipeline.geometry import (
    blend_directions,
    camera_extrinsics,
    project_pointcloud_zbuffer,
    stabilize_direction,
    synthetic_intrinsics,
    transform_points,
    weighted_target_direction,
)


def main() -> None:
    intrinsics = synthetic_intrinsics(1280, 720, 60.0)
    assert np.allclose(intrinsics[0, 0], 1108.51251684)
    center = np.array([1.0, 2.0, 3.0])
    rotation, camera_to_exo, exo_to_camera = camera_extrinsics(center, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(exo_to_camera @ camera_to_exo, np.eye(4), atol=1e-8)
    assert np.allclose(camera_to_exo[:3, 3], center)
    exo_points = np.array([center + rotation[:, 2] * 2.0, center - rotation[:, 2]])
    ego_points = transform_points(exo_points, exo_to_camera)
    assert np.allclose(ego_points, [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])

    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    frustum, visible, rgb, depth = project_pointcloud_zbuffer(
        ego_points, colors, intrinsics, 1280, 720
    )
    assert np.array_equal(frustum, [True, False])
    assert np.array_equal(visible, [True, False])
    assert np.array_equal(rgb[360, 640], [255, 0, 0])
    assert np.isclose(depth[360, 640], 2.0)

    targets = np.array([[-1.0, 0.0, 2.0], [2.0, 0.0, 4.0]])
    task = weighted_target_direction(np.zeros(3), targets, np.array([0.5, 0.5]))
    assert np.allclose(task, [0.0, 0.0, 1.0])
    blended = blend_directions(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.5)
    assert np.allclose(blended, [np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    stable = stabilize_direction(
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        smoothing=0.0,
        deadzone_degrees=0.0,
        max_step_degrees=3.0,
    )
    assert np.isclose(np.rad2deg(np.arccos(stable[2])), 3.0)
    print("geometry tests passed")


if __name__ == "__main__":
    main()
