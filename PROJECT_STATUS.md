# Project Status

## Setup Direction

The project is now documented for Linux server execution with conda only.

Supported environments:

```text
egoworld-main   # Python 3.11
egoworld-hamer  # Python 3.10
```

The old `.venv` / `external/hamer/.hamer` workflow is intentionally not used.

## Current Validation Level

Validated without installing packages or running heavy inference:

- broken `external/hamer` gitlink removed from tracking
- `external/hamer/` ignored as an external clone
- `requirements-hamer-torch-cpu.txt` filename fixed
- `setup_hamer.ps1` parses and handles empty/broken HaMeR folders more clearly
- `scripts/check_project_state.py` parses and is read-only by default
- active docs no longer use the stale HaMeR `--bboxes` command
- active docs are consolidated around Linux conda execution

Not yet validated:

- conda environment creation
- dependency installation
- `pip check` after install
- imports inside conda environments
- MoGe model download or inference
- HaMeR model loading or inference
- end-to-end Linux server run

## CPU/GPU Status

The current README assumes no GPU and installs CPU PyTorch wheels.

CPU is appropriate for:

- environment validation
- dummy depth smoke test
- MediaPipe detection
- hand crop extraction
- path and metadata checks

CPU may be slow or impractical for:

- MoGe inference
- HaMeR inference

If a GPU server becomes available, replace:

```text
requirements-torch-cpu.txt
requirements-hamer-torch-cpu.txt
```

with:

```text
requirements-torch-cu126.txt
requirements-hamer-torch-cu118.txt
```

## Known Constraints

- Do not install detectron2.
- Do not use the HaMeR renderer.
- Do not use pyrender, PyOpenGL, or OpenGL for Stage 1 inference.
- Do not run `pip install -e .[all]` inside HaMeR.
- Do not commit inputs, outputs, checkpoints, MANO files, conda envs, or caches.

## Next Validation Steps

1. On the Linux server, create `egoworld-main` and `egoworld-hamer`.
2. Install CPU requirements from `README.md`.
3. Run the import verification commands.
4. Add `inputs/exo.jpg`.
5. Run the dummy-depth smoke test.
6. Add HaMeR assets.
7. Attempt HaMeR import and then HaMeR inference if CPU runtime is acceptable.
