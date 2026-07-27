# EgoWorld Stage 1 Debug

Exocentric RGB 이미지·비디오에서 depth, hand pose, eye-centered ego camera pose를 추정하고 EgoWorld 입력을 만드는 연구용 파이프라인입니다.

## 주요 기능

- MoGe depth 추정 및 RGB-D point cloud 생성
- MediaPipe hand bbox/crop과 HaMeR/MANO 추론
- HaMeR projection 검증, hand depth 렌더링, depth scale 정합
- ViT 기반 ego hand-pose 학습·평가·단일 이미지 추론
- exo point cloud를 ego sparse map으로 변환
- Gaze3D와 눈 위치를 이용한 gaze-only/task-aware/task-only camera trajectory 생성
- 이미지·비디오용 EgoWorld diffusion 입력 준비

대용량 데이터, 입력 영상, 출력 결과, 체크포인트, MANO 파일은 저장소에 포함하지 않습니다.

## 빠른 시작

```bash
git clone --recurse-submodules <repo-url>
cd egoworld-stage1-debug
```

환경은 기능별로 분리합니다.

```text
egoworld-main   Python 3.11  depth, MediaPipe, point cloud, geometry
egoworld-hamer  Python 3.10  HaMeR/MANO inference
vitpose         Python 3.10  ViT hand-pose training/inference
gazeCVPR        Python 3.10  Gaze3D
egoworld-model  Python 3.10  EgoWorld diffusion
```

설치 순서, CUDA/CPU 선택, 외부 모델과 MANO asset 배치는 [환경 설정 문서](docs/ENVIRONMENT.md)를 따릅니다. 전체 명령과 파이프라인별 선택 기준은 [CLI 가이드](docs/PIPELINE_CLI.md)에 있습니다.

설치 후 기본 점검:

```bash
conda run -n egoworld-main python scripts/check_project_state.py
conda run -n egoworld-hamer python scripts/check_stage1_env.py
conda run -n egoworld-main python ego_camera_pose_pipeline/test_geometry.py
```

가벼운 Stage 1 smoke test:

```bash
mkdir -p inputs
# inputs/exo.jpg 배치

conda run -n egoworld-main python scripts/stage1/infer_depth_pointcloud.py \
  --image inputs/exo.jpg --out outputs/smoke/depth --depth_model dummy
conda run -n egoworld-main python scripts/stage1/infer_hand_bbox.py \
  --image inputs/exo.jpg --out outputs/smoke/bbox
conda run -n egoworld-main python scripts/stage1/crop_hands_from_bboxes.py \
  --image inputs/exo.jpg \
  --bbox_json outputs/smoke/bbox/hand_bboxes.json \
  --out outputs/smoke/hand_crops
```

`dummy` depth는 경로와 입출력 확인용입니다. 실제 결과에는 `--depth_model moge`를 사용합니다.

## 프로젝트 구조

```text
.
├── datasets/                    H2O/hand-pose dataset loader
├── docs/
│   ├── ENVIRONMENT.md           환경 구축 및 외부 asset 배치
│   └── PIPELINE_CLI.md          파이프라인별 실행 명령
├── ego_camera_pose_pipeline/    eye-centered camera trajectory와 좌표 변환
├── external/
│   ├── EgoWorld/                Git submodule
│   ├── gaze3d/                  Git submodule
│   └── hamer/                   로컬 clone, Git 제외
├── scripts/
│   ├── stage1/                  depth, bbox, crop, HaMeR 기본 단계
│   ├── tools/                   시각화 도구
│   └── *.py                     정합, 평가, wrapper, 데이터 준비
├── src/                         공용 depth/hand/point-cloud 모듈
├── vit_handpose/                기존 ViT hand-pose 구현
├── vit_handpose_v2/             root/pose 분리형 ViT 구현
├── requirements*.txt            main/HaMeR 및 CPU/CUDA 의존성
├── inputs/                      로컬 입력, Git 제외
├── outputs/                     실행 결과, Git 제외
├── data/                        dataset, Git 제외
└── checkpoints/                 모델 checkpoint, Git 제외
```

## 대표 실행 경로

### Stage 1 이미지

```text
depth → hand bbox/crop → HaMeR → projection check
      → hand depth → scale fitting → exo joints
```

진입점은 `scripts/stage1/`와 `scripts/verify_hamer_projection.py`입니다.

### Ego sparse map

```text
exo depth/point cloud + exo hand joints + ViT ego hand joints
→ scripts/make_ego_sparse_map.py
→ EgoWorld image/video input
```

이미지 wrapper는 `scripts/run_egoworld_image_pipeline.py`, 비디오 wrapper는 `scripts/run_egoworld_video_pipeline.py`입니다.

### Eye-centered camera pose

```text
video + Gaze3D + MoGe + MediaPipe Face/Hands
→ ego_camera_pose_pipeline/run_pipeline.py
→ camera trajectory + exo-to-ego point cloud
```

기본 좌표계는 OpenCV convention(`+X` right, `+Y` down, `+Z` forward)이며 monocular depth 단위는 metric scale을 보장하지 않습니다.

## Git에 포함하지 않는 항목

`.gitignore`는 다음을 제외합니다.

- `inputs/`, `outputs/`, `data/`, `checkpoints/`, `logs/`
- Python/pytest/editor cache와 가상환경
- checkpoint, NumPy array, point cloud, mesh, 영상, 압축 파일
- HaMeR source/model asset(`external/hamer/`, `external/hamer/_DATA/`)
- `.env`, cookie 파일과 로컬 credential

`external/EgoWorld`와 `external/gaze3d`는 submodule입니다. 두 폴더 안에서 수정한 코드는 부모 저장소 커밋에 자동 포함되지 않으므로, 각 저장소에서 별도로 커밋한 뒤 부모 저장소의 submodule commit을 갱신해야 합니다.

## 주의사항

- HaMeR의 MANO 파일은 라이선스상 직접 내려받아야 합니다.
- Stage 1에서는 `detectron2`, HaMeR renderer, `pyrender`, OpenGL/EGL이 필요하지 않습니다.
- `external/hamer`에서 `pip install -e .[all]`을 실행하지 않습니다.
- 요청한 MoGe 실행이 실패해도 자동으로 dummy depth로 대체하지 않습니다.
- raw HaMeR scale fitting과 reference-aligned hand depth는 목적이 다릅니다. 선택 기준은 CLI 가이드를 확인하세요.
