# Project Status

## Setup Direction

The project is documented for Linux server execution with conda only.

Supported environments:

```text
egoworld-main   # Python 3.11
egoworld-hamer  # Python 3.10
```

The old `.venv` / `external/hamer/.hamer` workflow is intentionally not used.

## Current Validation Level

Validated in this repo pass:

- `external/hamer` is treated as a local pure-inference backend.
- Stage 1 no longer requires HaMeR renderer, `pyrender`, OpenGL, EGL, GUI rendering, or `detectron2`.
- `hamer.utils.renderer` has a Stage 1 compatibility shim that exports `Renderer` and `cam_crop_to_full`; renderer calls raise `NotImplementedError`.
- `infer_hamer_from_bboxes.py` puts local `external/hamer` first on `sys.path` and installs renderer compatibility before importing HaMeR internals.
- HaMeR inference now fails clearly for missing input files, missing model/MANO files, bad bboxes, missing output keys, or non-finite arrays.
- Requested MoGe inference no longer silently falls back to dummy depth. Use `--depth_model dummy` explicitly for smoke tests.
- Empty point clouds and placeholder invalid crops are rejected.
- `scripts/check_stage1_env.py` was added for HaMeR import/path/model preflight checks.
- `scripts/verify_stage1_outputs.py` was added for generated output validation.
- `scripts/check_project_state.py` was updated for the current conda workflow.

Still environment-dependent and should be run on the target machine:

- dependency installation and `pip check`
- CUDA availability in both conda envs
- MoGe model download/inference
- HaMeR checkpoint loading/inference
- full end-to-end Stage 1 run on `inputs/exo.jpg`

## CPU/GPU Status

The current README assumes GPU use and installs CUDA PyTorch wheels:

```text
requirements-torch-cu126.txt
requirements-hamer-torch-cu118.txt
```

If CUDA is unavailable, `scripts/check_stage1_env.py` reports it. HaMeR may run on CPU but can be slow. MoGe and HaMeR GPU/runtime failures should be fixed directly; the pipeline should not silently produce dummy or fake successful outputs for a requested full Stage 1 run.

## Required Model Assets

Required paths:

```text
external/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
external/hamer/_DATA/data/mano/MANO_RIGHT.pkl
external/hamer/_DATA/data/mano/MANO_LEFT.pkl
external/hamer/_DATA/data/mano_mean_params.npz
```

`MANO_RIGHT.pkl` and `MANO_LEFT.pkl` cannot be auto-downloaded because of MANO license restrictions. The user must manually download them from MANO and place them at the paths above.

## Known Constraints

- Do not install `detectron2` for this Stage 1 path.
- Do not use the HaMeR renderer.
- Do not use `pyrender`, PyOpenGL, OpenGL, EGL, offscreen rendering, or GUI rendering for Stage 1 inference.
- Do not run `pip install -e .[all]` inside HaMeR.
- Do not commit inputs, outputs, checkpoints, MANO files, conda envs, or caches.

## Next Validation Steps

1. Activate `egoworld-hamer` and run `python scripts/check_stage1_env.py`.
2. Activate `egoworld-main` and run dummy-depth smoke steps from `README.md`.
3. Place all required HaMeR/MANO assets under `external/hamer/_DATA`.
4. Run HaMeR inference from `outputs/hand_bboxes.json`.
5. Run projection diagnostics.
6. Run `python scripts/verify_stage1_outputs.py`.
