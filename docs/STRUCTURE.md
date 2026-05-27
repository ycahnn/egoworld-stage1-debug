# Repository Structure

This document explains the files and folders needed to run Stage 1.

## Top Level

```text
egoworld-stage1/
  README.md
  PROJECT_STATUS.md
  requirements.txt
  requirements-torch-cpu.txt
  requirements-torch-cu126.txt
  requirements-hamer.txt
  requirements-hamer-torch-cpu.txt
  requirements-hamer-torch-cu118.txt
  infer_depth_pointcloud.py
  infer_hand_bbox.py
  crop_hands_from_bboxes.py
  infer_hamer_from_bboxes.py
  visualize_pointcloud.py
  src/
  scripts/
  docs/
  external/
  inputs/
  outputs/
```

## Main Pipeline Files

```text
infer_depth_pointcloud.py
infer_hand_bbox.py
crop_hands_from_bboxes.py
infer_hamer_from_bboxes.py
```

- `infer_depth_pointcloud.py`: depth estimation and RGB-D point cloud export.
- `infer_hand_bbox.py`: MediaPipe hand bounding boxes.
- `crop_hands_from_bboxes.py`: crop hand images from detected boxes.
- `infer_hamer_from_bboxes.py`: HaMeR/MANO inference from MediaPipe boxes.

## Source Modules

```text
src/
  depth_estimator.py
  hand_bbox_detector.py
  hand_cropper.py
  hand_pose_estimator.py
  io_utils.py
  pointcloud.py
```

- `depth_estimator.py`: dummy and MoGe depth wrapper.
- `hand_bbox_detector.py`: MediaPipe detection logic.
- `hand_cropper.py`: crop and crop metadata logic.
- `io_utils.py`: image and depth save/load helpers.
- `pointcloud.py`: RGB-D backprojection and PLY writing.
- `hand_pose_estimator.py`: placeholder legacy hand pose helper.

## Diagnostic Scripts

```text
scripts/
  check_project_state.py
  check_workspace_integrity.py
  verify_hamer_projection.py
  render_hand_depth.py
  scale_depth_with_hand.py
  export_p_exo_from_depth.py
  render_p_exo_skeleton_on_black.py
```

- `check_project_state.py`: read-only environment and file checks by default.
- `check_workspace_integrity.py`: checks expected final outputs.
- `verify_hamer_projection.py`: projects HaMeR mesh outputs back onto the image.
- `render_hand_depth.py`: rasterizes hand depth from projected HaMeR mesh.
- `scale_depth_with_hand.py`: aligns exocentric depth scale to hand depth.
- `export_p_exo_from_depth.py`: lifts 2D hand landmarks into exocentric 3D points.
- `render_p_exo_skeleton_on_black.py`: visualizes final `P_exo`.

## External Dependency

```text
external/hamer/
```

This is a manually cloned HaMeR repository. It is ignored by Git and should not be committed.

Expected source markers:

```text
external/hamer/hamer/
external/hamer/demo.py
external/hamer/setup.py
```

Expected model assets:

```text
external/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
external/hamer/_DATA/data/mano/MANO_RIGHT.pkl
external/hamer/_DATA/data/mano/MANO_LEFT.pkl
external/hamer/_DATA/data/mano_mean_params.npz
```

## Inputs And Outputs

```text
inputs/
  exo.jpg

outputs/
  ...
```

`inputs/` and `outputs/` are ignored by Git.

Typical outputs:

```text
outputs/depth_raw.npy
outputs/K_exo.npy
outputs/exo_point_cloud.ply
outputs/hand_bboxes.json
outputs/hand_crops/
outputs/hamer/
outputs/hamer_projection/
```

## Files That Should Not Be Committed

```text
inputs/
outputs/
external/hamer/
*.ckpt
*.pth
*.pt
*.pkl
*.npy
*.ply
*.obj
```
