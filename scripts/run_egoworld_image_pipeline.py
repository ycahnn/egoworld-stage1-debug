#!/usr/bin/env python3
"""Run the single-image EgoWorld preparation and inference pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROMPT = (
    "The image shows a man outdoors, standing under a large tree at a table. "
    "In front of him is a wooden cutting board holding chopped vegetables, including green stalks, red pieces, and sliced onions. "
    "He grips the vegetables with his left hand while using a chef's knife in his right hand to cut them into smaller pieces. "
    "Nearby objects, such as a glass container, towel, oil bottle, and extra knives, suggest he is preparing food in an outdoor cooking setup."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="inputs/exo.jpg", type=Path)
    parser.add_argument("--run_dir", default="outputs/pipeline_raw_mediapipe_egoworld", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt_file", type=Path)
    parser.add_argument("--depth_model", default="moge", choices=["moge", "dummy"])
    parser.add_argument("--main_env", default="egoworld-main")
    parser.add_argument("--vit_env", default="vitpose")
    parser.add_argument("--egoworld_env", default="egoworld-model")
    parser.add_argument("--vit_checkpoint", default="checkpoints/vit_handpose_subjects1234_pretrained/best.pt", type=Path)
    parser.add_argument("--egoworld_dir", default="external/EgoWorld", type=Path)
    parser.add_argument(
        "--egoworld_checkpoint",
        default="logs/h2o_action/inpainting/lightning_logs/version_0/checkpoints/step=7000.ckpt",
        type=Path,
    )
    parser.add_argument("--dataset_name", default="custom_pipeline_raw_mediapipe")
    parser.add_argument("--project_name", default="inpainting")
    parser.add_argument("--json_name", default="test")
    parser.add_argument("--sample_id", default="000000")
    parser.add_argument("--skip_egoworld", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def conda(env: str, args: list[str]) -> list[str]:
    return ["conda", "run", "-n", env, *args]


def write_egoworld_inputs(args: argparse.Namespace, sparse: Path, pose: Path, run_dir: Path) -> tuple[Path, Path]:
    egoworld_dir = resolve(args.egoworld_dir)
    dataset_root = egoworld_dir / "database" / args.dataset_name
    data_dir = egoworld_dir / "data" / args.dataset_name / args.project_name
    sample = args.sample_id

    for subdir in ("dense", "sparse", "pose", "text"):
        (dataset_root / subdir).mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sparse, dataset_root / "sparse" / f"{sample}.png")
    shutil.copy2(sparse, dataset_root / "dense" / f"{sample}.png")
    shutil.copy2(pose, dataset_root / "pose" / f"{sample}.png")

    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt
    (dataset_root / "text" / f"{sample}.txt").write_text(prompt + "\n", encoding="utf-8")

    row = {
        "dense": f"dense/{sample}.png",
        "sparse": f"sparse/{sample}.png",
        "pose": f"pose/{sample}.png",
        "text": f"text/{sample}.txt",
    }
    json_path = data_dir / f"{args.json_name}.json"
    json_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = [
        "# EgoWorld Input Package",
        "",
        f"source_image: {rel(resolve(args.image))}",
        f"sparse_source: {rel(sparse)}",
        f"pose_source: {rel(pose)}",
        "dense_source: placeholder copied from sparse because no ground-truth dense ego image is available",
        "exo_pose_source: MediaPipe 2D landmarks lifted with raw depth and K",
        "point_cloud_source: unscaled raw exocentric point cloud",
    ]
    (dataset_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    provenance = {
        "dataset_root": rel(dataset_root),
        "json": str(json_path.relative_to(egoworld_dir)),
        "root_path_for_test_py": str(Path("database") / args.dataset_name),
    }
    (run_dir / "egoworld_input_package.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return dataset_root, json_path


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image = resolve(args.image)

    depth_dir = run_dir / "exo_depth_pointcloud"
    mp_dir = run_dir / "mediapipe_pointcloud_joints"
    vit_dir = run_dir / "vit_handpose"
    sparse_dir = run_dir / "ego_sparse_map_raw_pointcloud_mediapipe_exo"

    run(conda(args.main_env, [
        "python", "scripts/stage1/infer_depth_pointcloud.py",
        "--image", rel(image),
        "--out", rel(depth_dir),
        "--depth_model", args.depth_model,
    ]))

    run(conda(args.main_env, [
        "python", "scripts/export_mediapipe_joints_from_depth.py",
        "--image", rel(image),
        "--depth", rel(depth_dir / "depth_raw.npy"),
        "--K", rel(depth_dir / "K_exo.npy"),
        "--out", rel(mp_dir),
    ]))

    run(conda(args.vit_env, [
        "python", "scripts/infer_vit_handpose_single.py",
        "--image", rel(image),
        "--checkpoint", rel(resolve(args.vit_checkpoint)),
        "--out", rel(vit_dir),
    ]))

    run(conda(args.main_env, [
        "python", "scripts/render_egoview_hand_pose.py",
        "--input", rel(vit_dir / "joints_2x21x3.npy"),
        "--out", rel(vit_dir / "predicted_hand_pose_egoview.png"),
    ]))

    run(conda(args.main_env, [
        "python", "scripts/make_ego_sparse_map.py",
        "--exo_joints", rel(mp_dir / "joints_3d_pointcloud.npy"),
        "--ego_joints", rel(vit_dir / "joints_2x21x3.npy"),
        "--exo_ply", rel(depth_dir / "exo_point_cloud.ply"),
        "--out", rel(sparse_dir),
        "--try_both_directions",
    ]))

    dataset_root, json_path = write_egoworld_inputs(
        args=args,
        sparse=sparse_dir / "ego_sparse_rgb.png",
        pose=vit_dir / "predicted_hand_pose_egoview.png",
        run_dir=run_dir,
    )

    summary = {
        "image": rel(image),
        "depth_dir": rel(depth_dir),
        "mediapipe_exo_pose": rel(mp_dir / "joints_3d_pointcloud.npy"),
        "vit_ego_pose": rel(vit_dir / "joints_2x21x3.npy"),
        "vit_ego_pose_image": rel(vit_dir / "predicted_hand_pose_egoview.png"),
        "raw_exo_point_cloud": rel(depth_dir / "exo_point_cloud.ply"),
        "ego_sparse_rgb": rel(sparse_dir / "ego_sparse_rgb.png"),
        "egoworld_dataset_root": rel(dataset_root),
        "egoworld_json": str(json_path),
    }
    (run_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.skip_egoworld:
        egoworld_dir = resolve(args.egoworld_dir)
        run(conda(args.egoworld_env, [
            "python", "test.py",
            "--dataset_name", args.dataset_name,
            "--project_name", args.project_name,
            "--json_name", args.json_name,
            "--root_path", str(Path("database") / args.dataset_name),
            "--pretrained_path", str(args.egoworld_checkpoint),
        ]), cwd=egoworld_dir)

    print("\nPipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
