# EgoWorld Stage 1

Linux server setup and run guide for the Stage 1 EgoWorld-style observation pipeline.

This repository uses conda only. Do not create `.venv` or `external/hamer/.hamer`.

## What This Runs

Input:

```text
inputs/exo.jpg
```

Pipeline:

1. Estimate exocentric depth with MoGe, or use dummy depth for a lightweight smoke test.
2. Backproject RGB-D into an exocentric point cloud.
3. Detect hands with MediaPipe.
4. Crop hands from detected boxes.
5. Run HaMeR/MANO inference from MediaPipe boxes.
6. Verify HaMeR projection on the original image.

## Constraints

- Do not install `detectron2`.
- Do not use the HaMeR renderer.
- Do not use `pyrender`, `PyOpenGL`, or OpenGL for Stage 1 inference.
- Do not run `pip install -e .[all]` inside HaMeR.
- Treat HaMeR as a pure-inference backend cloned under `external/hamer`.
- Do not commit inputs, outputs, checkpoints, MANO files, conda envs, or caches.

## Environments

Use two conda environments:

```text
egoworld-main   # Python 3.11, depth / point cloud / MediaPipe / verification
egoworld-hamer  # Python 3.10, HaMeR inference only
```

GPU is not assumed. The commands below use CPU wheels by default. CPU is enough for setup checks and lightweight smoke tests, but MoGe and HaMeR inference can be slow on CPU.

## 1. Clone

```bash
git clone <repo-url> egoworld-stage1
cd egoworld-stage1
```

Clone HaMeR source as an external dependency:

```bash
mkdir -p external
git clone --recursive https://github.com/geopavlakos/hamer.git external/hamer
```

Check the clone:

```bash
test -d external/hamer/hamer && echo "HaMeR package ok"
test -f external/hamer/demo.py && echo "HaMeR demo.py ok"
test -f external/hamer/setup.py && echo "HaMeR setup.py ok"
```

## 2. Create Conda Environments

```bash
conda create -n egoworld-main python=3.11 pip -y
conda create -n egoworld-hamer python=3.10 pip -y
```

If `conda activate` is unavailable:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Adjust the path if your conda installation is elsewhere.

## 3. Install Main Environment

```bash
conda activate egoworld-main
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip install -r requirements-torch-cpu.txt
python -m pip check
```

Verify:

```bash
python -c "import cv2, mediapipe, numpy, torch; print('main deps ok', torch.__version__, torch.cuda.is_available())"
```

## 4. Install HaMeR Environment

```bash
conda activate egoworld-hamer
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-hamer-torch-cpu.txt
python -m pip install chumpy==0.70 --no-build-isolation --index-url https://pypi.org/simple
python -m pip install -r requirements-hamer.txt --index-url https://pypi.org/simple
python -m pip check
```

Verify:

```bash
python -c "import torch; print('hamer torch', torch.__version__, torch.cuda.is_available())"
python -c "import smplx, timm, cv2, numpy, scipy; print('hamer deps ok')"
cd external/hamer
python -c "import hamer; print('hamer import ok')"
cd ../..
```

## 5. Add Model Assets

Place HaMeR assets under `external/hamer/_DATA`.

The wrapper checks for:

```text
external/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
external/hamer/_DATA/data/mano/MANO_RIGHT.pkl
external/hamer/_DATA/data/mano/MANO_LEFT.pkl
external/hamer/_DATA/data/mano_mean_params.npz
```

These files are not included in Git.

## 6. Add Input

```bash
mkdir -p inputs
```

Place your image here:

```text
inputs/exo.jpg
```

## 7. Smoke Test Without Heavy Models

This checks repo paths and the main pipeline without MoGe or HaMeR inference:

```bash
conda activate egoworld-main
python infer_depth_pointcloud.py --image inputs/exo.jpg --out outputs --depth_model dummy
python infer_hand_bbox.py --image inputs/exo.jpg --out outputs
python crop_hands_from_bboxes.py --image inputs/exo.jpg --bbox_json outputs/hand_bboxes.json --out outputs/hand_crops
python scripts/check_project_state.py
```

`check_project_state.py` is read-only by default. To write `outputs/environment_report.md`:

```bash
python scripts/check_project_state.py --write-report
```

## 8. Run Stage 1

Depth and point cloud:

```bash
conda activate egoworld-main
python infer_depth_pointcloud.py --image inputs/exo.jpg --out outputs --depth_model moge
```

Hand detection and crops:

```bash
conda activate egoworld-main
python infer_hand_bbox.py --image inputs/exo.jpg --out outputs
python crop_hands_from_bboxes.py --image inputs/exo.jpg --bbox_json outputs/hand_bboxes.json --out outputs/hand_crops
```

HaMeR inference:

```bash
conda activate egoworld-hamer
python infer_hamer_from_bboxes.py --image inputs/exo.jpg --bbox_json outputs/hand_bboxes.json --out outputs/hamer
```

Projection verification:

```bash
conda activate egoworld-main
python scripts/verify_hamer_projection.py --image inputs/exo.jpg --hamer_dir outputs/hamer --bbox_json outputs/hand_bboxes.json --out outputs/hamer_projection
```

Optional diagnostics:

```bash
conda activate egoworld-main
python scripts/render_hand_depth.py --image inputs/exo.jpg --hamer_dir outputs/hamer --bbox_json outputs/hand_bboxes.json --out outputs/hamer_depth
python scripts/scale_depth_with_hand.py --image inputs/exo.jpg --depth outputs/depth_raw.npy --hand_depth outputs/hamer_depth/hand_depth.npy --hand_mask outputs/hamer_depth/hand_mask.png --out outputs/scaled_depth
python scripts/export_p_exo_from_depth.py --image inputs/exo.jpg --depth outputs/scaled_depth/depth_scaled.npy --K outputs/K_exo.npy --out outputs/pose_depth
```

## Expected Outputs

```text
outputs/depth_raw.npy
outputs/depth_vis.png
outputs/depth_gray.png
outputs/depth_mask.png
outputs/K_exo.npy
outputs/exo_point_cloud.ply
outputs/hand_bboxes.json
outputs/hand_bbox_debug.png
outputs/hand_crops/
outputs/hamer/hamer_metadata.json
outputs/hamer/*_vertices.npy
outputs/hamer/*_joints.npy
outputs/hamer/*_faces.npy
outputs/hamer_projection/final_projection_report.md
outputs/hamer_projection/final_projection_debug.png
```

## More Info

- `PROJECT_STATUS.md`: current validation status and known limitations
- `docs/STRUCTURE.md`: file and folder layout
