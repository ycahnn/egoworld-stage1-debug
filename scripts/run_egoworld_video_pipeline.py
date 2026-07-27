#!/usr/bin/env python3
# Run the single-image EgoWorld pipeline over a video, then run prepared-frame EgoWorld diffusion.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EGG_PROMPT = (
    "A chef in a white uniform is standing at a counter, cooking eggs in a small frying pan on a portable electric stove. "
    "He grips the pan with tongs or a handle while using a red silicone spatula to stir and fold the eggs, carefully "
    "controlling their texture as he prepares a simple scrambled egg dish."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the video preparation pipeline, then optional EgoWorld diffusion.")
    parser.add_argument("--video", type=Path, help="Input exocentric video. Required unless --diffusion_only is used.")
    parser.add_argument("--run_dir", default="outputs/video_pipeline", type=Path)
    parser.add_argument("--output_video", type=Path, help="Sparse preview video. Defaults to <run_dir>/ego_sequence.mp4.")
    parser.add_argument("--frame_stride", default=1, type=int, help="Process every Nth source frame.")
    parser.add_argument("--start_frame", default=0, type=int, help="First source frame index to consider.")
    parser.add_argument("--max_frames", type=int, help="Maximum number of extracted frames to process.")
    parser.add_argument("--fps", type=float, help="Output FPS. Defaults to source_fps / frame_stride.")
    parser.add_argument("--image_ext", default="jpg", choices=["jpg", "png"])
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
    parser.add_argument("--dataset_name", default="custom_pipeline_video")
    parser.add_argument("--project_name", default="inpainting")
    parser.add_argument("--json_name_prefix", default="test")
    parser.add_argument("--prompt", default=DEFAULT_EGG_PROMPT, help="Prompt passed to each frame and diffusion video inference.")
    parser.add_argument("--prompt_file", type=Path, help="Prompt file passed to each frame and diffusion video inference.")
    parser.add_argument(
        "--frame_output",
        default="ego_sparse_map_raw_pointcloud_mediapipe_exo/ego_sparse_rgb.png",
        help="Per-frame sparse preview image path relative to each frame run directory.",
    )
    parser.add_argument(
        "--run_egoworld",
        action="store_true",
        help="Also run external/EgoWorld/test.py for each frame. Usually leave this off; video diffusion runs once after preparation.",
    )
    parser.add_argument("--skip_diffusion", action="store_true", help="Only prepare sparse/pose frames and skip EgoWorld video diffusion.")
    parser.add_argument("--diffusion_only", action="store_true", help="Reuse an existing run_dir and run only prepared-input creation plus EgoWorld video diffusion.")
    parser.add_argument("--diffusion_outdir", type=Path, help="Diffusion frame output directory. Defaults to <run_dir>/egoworld_video.")
    parser.add_argument("--diffusion_video_out", type=Path, help="Diffusion video path. Defaults to <run_dir>/egoworld_diffusion_sequence.mp4.")
    parser.add_argument("--ddim_steps", default=50, type=int)
    parser.add_argument("--scale", default=7.5, type=float)
    parser.add_argument("--strength", default=1.0, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temporal_alpha", default=0.1, type=float)
    parser.add_argument("--temporal_normalize_start_code", action="store_true")
    parser.add_argument("--no_save_diffusion_video", action="store_false", dest="save_diffusion_video")
    parser.set_defaults(save_diffusion_video=True)
    parser.add_argument("--overwrite_frames", action="store_true", help="Overwrite already extracted source frames.")
    parser.add_argument("--continue_on_error", action="store_true", help="Skip failed frames and keep processing.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def conda(env: str, args: list[str]) -> list[str]:
    return ["conda", "run", "-n", env, *args]


def prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return resolve(args.prompt_file).read_text(encoding="utf-8").strip()
    return args.prompt


def extract_frames(args: argparse.Namespace, run_dir: Path) -> tuple[list[dict[str, Any]], float]:
    video_path = resolve(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start_frame must be >= 0")

    frames_dir = run_dir / "exo_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_index = 0
    processed_index = 0
    records: list[dict[str, Any]] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        should_take = source_index >= args.start_frame and (source_index - args.start_frame) % args.frame_stride == 0
        if should_take:
            stem = f"frame_{processed_index:06d}"
            frame_path = frames_dir / f"{stem}.{args.image_ext}"
            if args.overwrite_frames or not frame_path.exists():
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
            records.append({"source_frame_index": source_index, "frame_index": processed_index, "image": rel(frame_path)})
            processed_index += 1
            if args.max_frames is not None and processed_index >= args.max_frames:
                break

        source_index += 1

    capture.release()
    if not records:
        raise RuntimeError("No frames were extracted from the input video.")
    return records, source_fps


def image_pipeline_command(args: argparse.Namespace, record: dict[str, Any], frame_run_dir: Path) -> list[str]:
    frame_index = int(record["frame_index"])
    cmd = conda(
        args.main_env,
        [
            "python", "scripts/run_egoworld_image_pipeline.py",
            "--image", record["image"],
            "--run_dir", rel(frame_run_dir),
            "--depth_model", args.depth_model,
            "--main_env", args.main_env,
            "--vit_env", args.vit_env,
            "--egoworld_env", args.egoworld_env,
            "--vit_checkpoint", rel(resolve(args.vit_checkpoint)),
            "--egoworld_dir", rel(resolve(args.egoworld_dir)),
            "--egoworld_checkpoint", str(args.egoworld_checkpoint),
            "--dataset_name", args.dataset_name,
            "--project_name", args.project_name,
            "--json_name", f"{args.json_name_prefix}_{frame_index:06d}",
            "--sample_id", f"{frame_index:06d}",
            "--prompt", prompt_text(args),
        ],
    )
    if not args.run_egoworld:
        cmd.append("--skip_egoworld")
    return cmd


def process_frames(args: argparse.Namespace, run_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for record in records:
        frame_index = int(record["frame_index"])
        frame_run_dir = run_dir / "frames" / f"frame_{frame_index:06d}"
        frame_run_dir.mkdir(parents=True, exist_ok=True)
        record["run_dir"] = rel(frame_run_dir)
        record["status"] = "pending"
        try:
            run(image_pipeline_command(args, record, frame_run_dir))
            output_image = frame_run_dir / args.frame_output
            if not output_image.exists():
                raise FileNotFoundError(f"Expected per-frame sparse preview output not found: {output_image}")
            record["ego_image"] = rel(output_image)
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            if not args.continue_on_error:
                raise
        processed.append(record)
    return processed


def prepare_diffusion_inputs(args: argparse.Namespace, run_dir: Path, records: list[dict[str, Any]]) -> Path:
    input_root = run_dir / "prepared_egoworld_video_inputs"
    if input_root.exists():
        shutil.rmtree(input_root)
    sparse_dir = input_root / "sparse"
    pose_dir = input_root / "pose"
    text_dir = input_root / "text"
    for directory in (sparse_dir, pose_dir, text_dir):
        directory.mkdir(parents=True, exist_ok=True)

    text = prompt_text(args)
    prepared = 0
    for record in records:
        if record.get("status") != "ok":
            continue
        frame_index = int(record["frame_index"])
        frame_name = f"{frame_index:06d}"
        frame_run_dir = resolve(Path(record["run_dir"]))
        sparse_path = frame_run_dir / "ego_sparse_map_raw_pointcloud_mediapipe_exo" / "ego_sparse_rgb.png"
        pose_path = frame_run_dir / "vit_handpose" / "predicted_hand_pose_egoview.png"
        if not sparse_path.exists():
            raise FileNotFoundError(f"Missing sparse map for diffusion input: {sparse_path}")
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose map for diffusion input: {pose_path}")
        shutil.copy2(sparse_path, sparse_dir / f"{frame_name}.png")
        shutil.copy2(pose_path, pose_dir / f"{frame_name}.png")
        (text_dir / f"{frame_name}.txt").write_text(text + "\n", encoding="utf-8")
        record["diffusion_sparse"] = rel(sparse_dir / f"{frame_name}.png")
        record["diffusion_pose"] = rel(pose_dir / f"{frame_name}.png")
        prepared += 1

    if prepared == 0:
        raise RuntimeError("No successful frames are available to prepare diffusion inputs.")
    return input_root


def run_diffusion_video(args: argparse.Namespace, run_dir: Path, input_root: Path, fps: float) -> dict[str, Any]:
    egoworld_dir = resolve(args.egoworld_dir)
    diffusion_outdir = resolve(args.diffusion_outdir) if args.diffusion_outdir else run_dir / "egoworld_video"
    diffusion_video_out = resolve(args.diffusion_video_out) if args.diffusion_video_out else run_dir / "egoworld_diffusion_sequence.mp4"

    cmd = conda(
        args.egoworld_env,
        [
            "python", "video_test.py",
            "--config", "models/inpainting.yaml",
            "--pretrained_path", str(args.egoworld_checkpoint),
            "--input_root", str(input_root),
            "--outdir", str(diffusion_outdir),
            "--ddim_steps", str(args.ddim_steps),
            "--scale", str(args.scale),
            "--strength", str(args.strength),
            "--eta", str(args.eta),
            "--seed", str(args.seed),
            "--temporal_alpha", str(args.temporal_alpha),
            "--fps", str(fps),
            "--prompt", prompt_text(args),
        ],
    )
    if args.temporal_normalize_start_code:
        cmd.append("--temporal_normalize_start_code")
    if args.save_diffusion_video:
        cmd.extend(["--save_video", "--video_out", str(diffusion_video_out)])

    run(cmd, cwd=egoworld_dir)
    return {
        "input_root": rel(input_root),
        "outdir": rel(diffusion_outdir),
        "frames_dir": rel(diffusion_outdir / "frames"),
        "video": rel(diffusion_video_out) if args.save_diffusion_video else None,
        "ddim_steps": args.ddim_steps,
        "scale": args.scale,
        "temporal_alpha": args.temporal_alpha,
        "temporal_normalize_start_code": args.temporal_normalize_start_code,
        "prompt": prompt_text(args),
    }


def assemble_video(records: list[dict[str, Any]], output_path: Path, fps: float) -> int:
    ok_records = [record for record in records if record.get("status") == "ok" and record.get("ego_image")]
    if not ok_records:
        raise RuntimeError("No successful sparse preview frames are available for video assembly.")
    if fps <= 0.0:
        fps = 30.0

    first = cv2.imread(str(resolve(Path(ok_records[0]["ego_image"]))))
    if first is None:
        raise FileNotFoundError(f"Unable to read first sparse preview frame: {ok_records[0]['ego_image']}")
    height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")

    written = 0
    for record in ok_records:
        image_path = resolve(Path(record["ego_image"]))
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Unable to read sparse preview frame: {image_path}")
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    writer.release()
    return written


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "video_pipeline_summary.json"

    if args.diffusion_only:
        if not summary_path.exists():
            raise FileNotFoundError(f"--diffusion_only requires an existing summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        processed = summary.get("frames", [])
        if not processed:
            raise RuntimeError(f"No frame records found in {summary_path}")
        output_fps = float(args.fps) if args.fps else float(summary.get("output_fps") or 30.0)
        diffusion_input_root = prepare_diffusion_inputs(args, run_dir, processed)
        diffusion = run_diffusion_video(args, run_dir, diffusion_input_root, output_fps)
        summary["diffusion"] = diffusion
        summary["diffusion_only_rerun"] = True
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nDiffusion-only video pipeline complete.")
        print(json.dumps(summary, indent=2))
        return

    if args.video is None:
        raise ValueError("--video is required unless --diffusion_only is used")

    records, source_fps = extract_frames(args, run_dir)
    output_fps = float(args.fps) if args.fps else source_fps / float(max(args.frame_stride, 1))
    if output_fps <= 0.0:
        output_fps = 30.0

    processed = process_frames(args, run_dir, records)
    output_video = resolve(args.output_video) if args.output_video else run_dir / "ego_sequence.mp4"
    written = assemble_video(processed, output_video, output_fps)

    diffusion = None
    if not args.skip_diffusion:
        diffusion_input_root = prepare_diffusion_inputs(args, run_dir, processed)
        diffusion = run_diffusion_video(args, run_dir, diffusion_input_root, output_fps)

    summary = {
        "video": rel(resolve(args.video)),
        "run_dir": rel(run_dir),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "extracted_frame_count": len(records),
        "written_frame_count": written,
        "frame_output": args.frame_output,
        "output_video": rel(output_video),
        "output_video_kind": "sparse_preview",
        "diffusion": diffusion,
        "frames": processed,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nVideo pipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
