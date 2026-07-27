# Environment Setup

이 프로젝트는 충돌이 많은 vision 모델을 한 환경에 합치지 않고 기능별 Conda 환경으로 분리합니다. 아래 명령은 저장소 루트에서 실행합니다.

## 1. 시스템 요구사항

- Linux
- Conda/Miniconda
- Git과 Git LFS
- NVIDIA GPU 및 설치할 PyTorch wheel과 호환되는 driver
- 모델과 dataset을 위한 충분한 로컬 디스크

현재 확인된 환경은 다음과 같습니다.

| 환경 | Python | 핵심 패키지 |
|---|---:|---|
| `egoworld-main` | 3.11.15 | torch 2.12.0, MediaPipe 0.10.14, Open3D 0.19.0, NumPy 1.26.4 |
| `egoworld-hamer` | 3.10.20 | torch 2.7.1+cu118, torchvision 0.22.1+cu118, NumPy 1.23.5 |
| `vitpose` | 3.10.20 | torch 2.12.0, torchvision 0.27.0, timm 1.0.27 |
| `gazeCVPR` | 3.10.14 | torch 2.1.1, torchvision 0.16.1, NumPy 1.24.4 |
| `egoworld-model` | 3.10.20 | torch 2.4.1+cu121, torchvision 0.19.1+cu121 |

버전 표는 현재 서버에서 검증된 기준이며, GPU driver와 wheel 제공 상황에 따라 CUDA variant는 조정할 수 있습니다.

## 2. 저장소와 외부 코드

```bash
git clone --recurse-submodules <repo-url>
cd egoworld-stage1-debug
git submodule update --init --recursive
```

HaMeR는 대용량 asset과 별도 라이선스 파일을 포함하므로 submodule이 아니라 로컬 의존성으로 둡니다.

```bash
git clone --recursive https://github.com/geopavlakos/hamer.git external/hamer
```

확인:

```bash
test -f external/hamer/demo.py
test -d external/hamer/hamer
test -f external/EgoWorld/test.py
test -f external/gaze3d/demo.py
```

## 3. Main 환경

MoGe, MediaPipe, point cloud, projection과 camera geometry에 사용합니다.

```bash
conda create -n egoworld-main python=3.11 pip -y
conda activate egoworld-main
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip install -r requirements-torch-cu126.txt
python -m pip check
```

CPU만 사용할 경우 마지막 설치 명령을 다음으로 바꿉니다.

```bash
python -m pip install -r requirements-torch-cpu.txt
```

검증:

```bash
python -c "import cv2, mediapipe, numpy, open3d, torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/check_project_state.py
python ego_camera_pose_pipeline/test_geometry.py
```

## 4. HaMeR 환경

HaMeR/MANO inference만 담당합니다.

```bash
conda create -n egoworld-hamer python=3.10 pip -y
conda activate egoworld-hamer
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-hamer-torch-cu118.txt
python -m pip install chumpy==0.70 --no-build-isolation
python -m pip install -r requirements-hamer.txt --index-url https://pypi.org/simple
python -m pip check
python scripts/check_stage1_env.py
```

CPU 환경은 `requirements-hamer-torch-cpu.txt`를 사용합니다. CPU HaMeR inference는 매우 느릴 수 있습니다.

다음 asset은 Git에 포함되지 않습니다.

```text
external/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
external/hamer/_DATA/data/mano/MANO_RIGHT.pkl
external/hamer/_DATA/data/mano/MANO_LEFT.pkl
external/hamer/_DATA/data/mano_mean_params.npz
```

`MANO_RIGHT.pkl`과 `MANO_LEFT.pkl`은 MANO 라이선스에 동의한 뒤 직접 내려받아야 합니다.

## 5. ViT hand-pose 환경

```bash
conda create -n vitpose python=3.10 pip -y
conda activate vitpose
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r vit_handpose/requirements.txt
python -m pip check
```

학습 데이터는 `data/`, checkpoint는 `checkpoints/` 아래에 두며 둘 다 Git에서 제외됩니다.

## 6. Gaze3D 환경

`external/gaze3d/environment.yaml`이 Gaze3D용 Python, FFmpeg/PyAV와 package version을 정의합니다.

```bash
conda env create -f external/gaze3d/environment.yaml
conda activate gazeCVPR
python -m pip check
```

Gaze3D checkpoint와 detector weight는 외부 저장소 안내에 따라 `external/gaze3d/checkpoints/`와 `external/gaze3d/weights/`에 둡니다.

## 7. EgoWorld diffusion 환경

```bash
conda create -n egoworld-model python=3.10 pip -y
conda activate egoworld-model
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r external/EgoWorld/requirements.txt
python -m pip check
```

EgoWorld checkpoint는 `external/EgoWorld/logs/` 아래에 두며 submodule 저장소에 커밋하지 않습니다.

## 8. 입력과 사전 점검

```bash
mkdir -p inputs outputs
```

입력 이미지 또는 비디오를 `inputs/`에 배치한 뒤:

```bash
conda run -n egoworld-main python scripts/check_project_state.py
conda run -n egoworld-hamer python scripts/check_stage1_env.py
```

환경별 실행 명령은 [PIPELINE_CLI.md](PIPELINE_CLI.md)를 참고합니다.

## 9. Submodule 수정 시 주의

부모 저장소는 submodule 내부 파일이 아니라 submodule commit hash만 기록합니다.

```bash
git -C external/EgoWorld status
git -C external/gaze3d status
```

내부 수정이 필요하면 각 submodule을 쓰기 가능한 fork로 연결해 먼저 커밋·push하고, 부모 저장소에서 변경된 submodule hash를 커밋해야 다른 서버에서도 동일하게 재현됩니다.
