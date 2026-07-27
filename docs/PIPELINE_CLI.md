# Pipeline CLI Guide

This document organizes the project entrypoints by use case. Run commands from
the repository root unless noted otherwise.

## Environments

Use the matching conda environment for each stage:

```bash
# Main geometry, depth, MediaPipe, projection, scale fitting
conda activate egoworld-main

# HaMeR/MANO inference only
conda activate egoworld-hamer

# ViT ego hand-pose model
conda activate vitpose

# EgoWorld diffusion model
conda activate egoworld-model
```

The wrapper scripts call `conda run -n ...` internally, so they can be launched
from any shell as long as `conda` is available.

## Output Layout

Prefer one output directory per input image or video frame:

```text
outputs/<sample_name>/
  depth/
  bbox/
  hand_crops/
  hamer/
  hamer_projection/
  hamer_depth/
  scaled_depth/
  hamer_pose_exo/
  vit_handpose/
  ego_sparse_map_raw_pointcloud_mediapipe_exo/
```

This avoids mixing results from different inputs in the legacy flat `outputs/`
layout.

## Stage 1 Image: Depth, HaMeR, Projection

Use this path when you want exocentric depth, hand detection, HaMeR mesh, and
projection diagnostics for a single RGB image.

```bash
IMAGE=inputs/h2o_subject1_h1_0_cam0_000000.png
RUN=outputs/h2o_subject1_h1_0_cam0_000000
```

### 1. Depth and Point Cloud

Produces raw depth, camera intrinsics, and exocentric point cloud.

```bash
conda activate egoworld-main
python scripts/stage1/infer_depth_pointcloud.py \
  --image "$IMAGE" \
  --out "$RUN/depth" \
  --depth_model moge
```

Use `--depth_model dummy` only for smoke tests. It is not meaningful for final
scale fitting.

### 2. Hand Bounding Boxes

Runs MediaPipe hand detection and saves bboxes.

```bash
conda activate egoworld-main
python scripts/stage1/infer_hand_bbox.py \
  --image "$IMAGE" \
  --out "$RUN/bbox"
```

Outputs:

```text
$RUN/bbox/hand_bboxes.json
$RUN/bbox/hand_bbox_debug.png
```

### 3. Hand Crops

Creates crop images for inspection/debugging.

```bash
conda activate egoworld-main
python scripts/stage1/crop_hands_from_bboxes.py \
  --image "$IMAGE" \
  --bbox_json "$RUN/bbox/hand_bboxes.json" \
  --out "$RUN/hand_crops"
```

### 4. HaMeR Inference

Runs HaMeR from the detected bboxes and saves MANO vertices, joints, faces, and
OBJ meshes.

```bash
conda activate egoworld-hamer
python scripts/stage1/infer_hamer_from_bboxes.py \
  --image "$IMAGE" \
  --bbox_json "$RUN/bbox/hand_bboxes.json" \
  --out "$RUN/hamer"
```

Outputs:

```text
$RUN/hamer/hamer_metadata.json
$RUN/hamer/hand_00_<side>_vertices.npy
$RUN/hamer/hand_00_<side>_joints.npy
$RUN/hamer/hand_00_<side>_faces.npy
$RUN/hamer/hand_00_<side>_mesh.obj
```

### 5. Mesh Projection Check

Projects the HaMeR mesh back into the original exocentric image and reports
whether it overlaps the MediaPipe bbox.

```bash
conda activate egoworld-main
python scripts/verify_hamer_projection.py \
  --image "$IMAGE" \
  --hamer_dir "$RUN/hamer" \
  --bbox_json "$RUN/bbox/hand_bboxes.json" \
  --out "$RUN/hamer_projection"
```

Use these first when judging HaMeR quality:

```text
$RUN/hamer_projection/final_projection_debug.png
$RUN/hamer_projection/final_projection_report.md
```

## Scale Fitting Choices

There are two related but different modes. Pick one deliberately.

### A. Raw HaMeR Scale Fitting

Use this when you want to estimate the scale between MoGE depth and HaMeR mesh
depth. This is the diagnostic "what scale would align them?" mode.

Render HaMeR depth without pre-aligning it:

```bash
conda activate egoworld-main
python scripts/render_hand_depth.py \
  --image "$IMAGE" \
  --hamer_dir "$RUN/hamer" \
  --bbox_json "$RUN/bbox/hand_bboxes.json" \
  --out "$RUN/hamer_depth_rawscale" \
  --reference_depth "$RUN/depth/depth_raw.npy"
```

Then fit a depth scale:

```bash
conda activate egoworld-main
python scripts/scale_depth_with_hand.py \
  --image "$IMAGE" \
  --depth "$RUN/depth/depth_raw.npy" \
  --hand_depth "$RUN/hamer_depth_rawscale/hand_depth.npy" \
  --hand_mask "$RUN/hamer_depth_rawscale/hand_mask.png" \
  --K "$RUN/depth/K_exo.npy" \
  --out "$RUN/scaled_depth_rawhand"
```

Read:

```text
$RUN/scaled_depth_rawhand/scale_report.md
```

Important: if the report warns about low final ratio count, inspect the ratio
filtering. The current script filters ratios to `0.05 <= ratio <= 20.0`, which
can be too strict for some HaMeR/MoGE scale gaps.

### B. Reference-Aligned HaMeR Depth

Use this when you want the rendered HaMeR hand depth to already sit on the MoGE
depth scale before downstream export. This is useful for a stable overlay, but
it makes the later scale fitting report close to `s_star ~= 1.0` by construction.

```bash
conda activate egoworld-main
python scripts/render_hand_depth.py \
  --image "$IMAGE" \
  --hamer_dir "$RUN/hamer" \
  --bbox_json "$RUN/bbox/hand_bboxes.json" \
  --out "$RUN/hamer_depth" \
  --reference_depth "$RUN/depth/depth_raw.npy" \
  --align_depth_to_reference

python scripts/scale_depth_with_hand.py \
  --image "$IMAGE" \
  --depth "$RUN/depth/depth_raw.npy" \
  --hand_depth "$RUN/hamer_depth/hand_depth.npy" \
  --hand_mask "$RUN/hamer_depth/hand_mask.png" \
  --K "$RUN/depth/K_exo.npy" \
  --out "$RUN/scaled_depth"
```

This path is not the best way to diagnose the original HaMeR/MoGE scale gap,
because the hand depth has already been median-aligned to the reference depth.

### Export HaMeR Joints in Exo Coordinates

After scale fitting, export HaMeR joints in the projected exo camera coordinate
frame:

```bash
conda activate egoworld-main
python scripts/export_hamer_joints_exo.py \
  --image "$IMAGE" \
  --hamer_dir "$RUN/hamer" \
  --scaled_depth "$RUN/scaled_depth/depth_scaled.npy" \
  --out "$RUN/hamer_pose_exo"
```

Main output:

```text
$RUN/hamer_pose_exo/hamer_joints_exo.npy
```

## Ego Sparse Map Pipeline

Use this path when the goal is to create an egocentric sparse RGB map from an
exocentric observation. It combines:

1. Exo depth and point cloud.
2. Exo hand joints.
3. Ego hand-pose prediction.
4. Exo-to-ego sparse projection.

### Exo Joints From MediaPipe + Depth

This is the current default path used by `run_egoworld_image_pipeline.py`.

```bash
conda activate egoworld-main
python scripts/export_mediapipe_joints_from_depth.py \
  --image "$IMAGE" \
  --depth "$RUN/depth/depth_raw.npy" \
  --K "$RUN/depth/K_exo.npy" \
  --out "$RUN/mediapipe_pointcloud_joints"
```

Output:

```text
$RUN/mediapipe_pointcloud_joints/joints_3d_pointcloud.npy
```

### Ego Hand Pose From ViT

Predict ego-space hand joints from the exocentric RGB image using the fine-tuned
ViT hand-pose model.

```bash
conda activate vitpose
python scripts/infer_vit_handpose_single.py \
  --image "$IMAGE" \
  --checkpoint checkpoints/vit_handpose_subjects1234_pretrained/best.pt \
  --out "$RUN/vit_handpose"
```

Render the predicted ego pose:

```bash
conda activate egoworld-main
python scripts/render_egoview_hand_pose.py \
  --input "$RUN/vit_handpose/joints_2x21x3.npy" \
  --out "$RUN/vit_handpose/predicted_hand_pose_egoview.png"
```

### Build Ego Sparse Map

Use MediaPipe exo joints:

```bash
conda activate egoworld-main
python scripts/make_ego_sparse_map.py \
  --exo_joints "$RUN/mediapipe_pointcloud_joints/joints_3d_pointcloud.npy" \
  --ego_joints "$RUN/vit_handpose/joints_2x21x3.npy" \
  --exo_ply "$RUN/depth/exo_point_cloud.ply" \
  --out "$RUN/ego_sparse_map_raw_pointcloud_mediapipe_exo" \
  --try_both_directions
```

Use HaMeR exo joints instead:

```bash
conda activate egoworld-main
python scripts/make_ego_sparse_map.py \
  --exo_joints "$RUN/hamer_pose_exo/hamer_joints_exo.npy" \
  --ego_joints "$RUN/vit_handpose/joints_2x21x3.npy" \
  --exo_ply "$RUN/scaled_depth/exo_point_cloud_scaled.ply" \
  --out "$RUN/ego_sparse_map_hamer_exo" \
  --try_both_directions
```

Main outputs:

```text
$RUN/ego_sparse_map_*/ego_sparse_rgb.png
$RUN/ego_sparse_map_*/ego_sparse_depth.npy
$RUN/ego_sparse_map_*/report.json
```

## Image End-to-End Wrappers

### Ego Sparse + Optional EgoWorld Diffusion

This is the existing image-level wrapper. It does not run HaMeR mesh projection
or hand-depth scale fitting. It uses MediaPipe depth-lifted exo joints plus ViT
ego joints to create sparse ego inputs for EgoWorld.

```bash
conda activate egoworld-main
python scripts/run_egoworld_image_pipeline.py \
  --image "$IMAGE" \
  --run_dir "$RUN" \
  --depth_model moge \
  --skip_egoworld
```

Remove `--skip_egoworld` to also run `external/EgoWorld/test.py` for one image.

Use this when:

- You want sparse ego map generation.
- You want to prepare EgoWorld diffusion inputs.
- You do not need HaMeR mesh projection or scale fitting.

### Full HaMeR + Scale Fitting Image Pipeline

There is no dedicated wrapper script yet for the full HaMeR path. For now, run
the Stage 1 commands above in order:

```text
depth -> bbox -> crop -> hamer -> projection -> hand_depth -> scale_fit -> export_hamer_joints_exo
```

Use this when:

- You need HaMeR mesh projection diagnostics.
- You need hand-depth scale fitting.
- You want `hamer_joints_exo.npy` or scaled point clouds based on HaMeR depth.

## Video Pipelines

### Prepare Sparse Ego Preview From Video

This extracts video frames and runs the image wrapper on each frame. By default
it also runs EgoWorld video diffusion after preparation; add `--skip_diffusion`
to only prepare sparse/pose frames and assemble the sparse preview video.

```bash
conda activate egoworld-main
python scripts/run_egoworld_video_pipeline.py \
  --video inputs/exo.mp4 \
  --run_dir outputs/video_pipeline \
  --frame_stride 1 \
  --depth_model moge \
  --skip_diffusion
```

Useful quick test:

```bash
python scripts/run_egoworld_video_pipeline.py \
  --video inputs/exo.mp4 \
  --run_dir outputs/video_smoke \
  --max_frames 3 \
  --depth_model dummy \
  --skip_diffusion
```

### Video Diffusion After Sparse Preparation

Run full preparation plus EgoWorld video diffusion:

```bash
conda activate egoworld-main
python scripts/run_egoworld_video_pipeline.py \
  --video inputs/exo.mp4 \
  --run_dir outputs/video_pipeline \
  --frame_stride 1 \
  --depth_model moge \
  --ddim_steps 50 \
  --scale 7.5 \
  --temporal_alpha 0.1
```

Reuse an existing prepared `run_dir` and only rerun diffusion:

```bash
python scripts/run_egoworld_video_pipeline.py \
  --run_dir outputs/video_pipeline \
  --diffusion_only \
  --ddim_steps 50 \
  --scale 7.5
```

Current video wrapper uses the image-level Ego sparse path. It does not run the
HaMeR mesh/scale-fitting path per frame.

## Validation and Debugging

Workspace checks:

```bash
conda activate egoworld-main
python scripts/check_project_state.py --write-report

conda activate egoworld-hamer
python scripts/check_stage1_env.py
```

Validate Stage 1 outputs in a directory:

```bash
conda activate egoworld-main
python scripts/verify_stage1_outputs.py \
  --outputs "$RUN"
```

Inspect depth values interactively:

```bash
python scripts/click_depth_viewer.py \
  --image "$IMAGE" \
  --raw "$RUN/depth/depth_raw.npy" \
  --scaled "$RUN/scaled_depth/depth_scaled.npy" \
  --hand "$RUN/hamer_depth/hand_depth.npy"
```

## Eye-Centered Camera Trajectory

This path is separate from the image-level sparse-map wrapper. It uses MoGe,
MediaPipe Face Mesh, and Gaze3D to estimate an eye-centered camera for every
video frame. Exo and virtual ego cameras both use OpenCV axes: `+X` right,
`+Y` down, and `+Z` forward.

Gaze-only:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n egoworld-main \
  python ego_camera_pose_pipeline/run_pipeline.py \
  --video inputs/exo.mp4 \
  --out outputs/ego_camera_pose_gaze
```

Reuse an existing Gaze3D CSV and enable task-aware orientation:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n egoworld-main \
  python ego_camera_pose_pipeline/run_pipeline.py \
  --video inputs/exo.mp4 \
  --gaze_csv external/gaze3d/output/exo_video_predicted_gaze.csv \
  --out outputs/ego_camera_pose_task_aware \
  --task-aware \
  --ego_horizontal_fov 80
```

Create a task-only trajectory from a task-aware run:

```bash
conda run -n egoworld-main \
  python scripts/make_task_only_trajectory.py \
  --source-run outputs/ego_camera_pose_task_aware \
  --out outputs/ego_camera_pose_task_only
```

Transform stored exocentric point clouds into the virtual ego camera:

```bash
conda run -n egoworld-main \
  python ego_camera_pose_pipeline/transform_exo_to_ego.py \
  --run-dir outputs/ego_camera_pose_task_aware \
  --dense
```

The run directory contains trajectory arrays/CSV, per-frame camera transforms,
intrinsics, depth, landmark observations, overlays, and a pipeline summary.
Monocular MoGe translation units are not guaranteed to be metric.

## Root/Pose-Decoupled ViT Hand Pose

`vit_handpose_v2/` is independent of the legacy `vit_handpose/` model. It uses
separate heads for wrist/root and wrist-relative pose. H2O labels are treated
as meters by default and metrics are reported in millimeters.

Train:

```bash
conda run -n vitpose python vit_handpose_v2/train.py \
  --train_csv data/preprocessed_224_png_nvme/combined_subjects1234/h2o_subjects1234_train_pre224.csv \
  --val_csv data/preprocessed_224_png_nvme/combined_subjects1234/h2o_subjects1234_val_pre224.csv \
  --image_root data/preprocessed_224_png_nvme \
  --already_resized \
  --batch_size 64 \
  --save_dir checkpoints/vit_handpose_v2_subjects1234
```

Evaluate:

```bash
conda run -n vitpose python vit_handpose_v2/evaluate.py \
  --test_csv data/preprocessed_224_png_nvme/combined_subjects1234/h2o_subjects1234_test_pre224.csv \
  --image_root data/preprocessed_224_png_nvme \
  --already_resized \
  --checkpoint checkpoints/vit_handpose_v2_subjects1234/best_pose.pt
```

## Which Command Should I Use?

Use `scripts/run_egoworld_image_pipeline.py` when you want the current image
EgoWorld-prep path: depth, MediaPipe exo joints, ViT ego joints, sparse ego map,
and optional single-image EgoWorld diffusion.

Use the manual Stage 1 sequence when you need HaMeR mesh outputs, projection
debug images, hand-depth rendering, scale fitting, or HaMeR joints in exo camera
coordinates.

Use `scripts/run_egoworld_video_pipeline.py` for videos. It runs the image
EgoWorld-prep path per frame and can optionally run video diffusion. It does not
currently include the HaMeR scale-fitting branch per frame.

Use raw HaMeR scale fitting when measuring the original MoGE/HaMeR scale gap.
Use reference-aligned hand depth when you want a stable hand-depth overlay and
downstream export on the reference depth scale.

Use `ego_camera_pose_pipeline/run_pipeline.py` when the output should be an
eye-centered gaze/task camera trajectory rather than a hand-aligned sparse map.
