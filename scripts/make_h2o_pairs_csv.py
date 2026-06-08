"""Create H2O RGB-to-target CSV pairs.

This script pairs RGB images from input cameras with labels from a fixed target
camera. For the common subject1 setup, images from cam0, cam1, cam2, and cam3
are paired with labels from cam4/hand_pose using the same frame id.

Example:

python scripts/make_h2o_pairs_csv.py \
  --data-root /data/H2O/subject1_v1_1 \
  --subject subject1 \
  --envs h1 h2 o1 o2 k1 k2 \
  --input-cams cam0 cam1 cam2 cam3 \
  --target-cam cam4 \
  --target-type hand_pose \
  --output-csv data/h2o_subject1_cam0123_to_cam4_hand_pose.csv \
  --relative
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path


CSV_COLUMNS = [
    "image_path",
    "target_path",
    "subject",
    "env",
    "clip",
    "input_cam",
    "target_cam",
    "frame_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV pairs from H2O fixed-camera RGB frames to target camera labels."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to the root that contains the subject folder, e.g. /path/to/subject1_v1_1.",
    )
    parser.add_argument("--subject", default="subject1", help="Subject folder name.")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["h1", "h2", "o1", "o2", "k1", "k2"],
        help="Environment folders to scan, in output order.",
    )
    parser.add_argument(
        "--input-cams",
        nargs="+",
        default=["cam0", "cam1", "cam2", "cam3"],
        help="Input camera folders to scan, in output order.",
    )
    parser.add_argument(
        "--target-cam",
        default="cam4",
        help="Camera folder to use for target labels.",
    )
    parser.add_argument(
        "--target-type",
        default="hand_pose",
        help="Target label folder under target camera, e.g. hand_pose.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Write image_path and target_path relative to --data-root.",
    )
    return parser.parse_args()


def clip_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def display_path(path: Path, data_root: Path, relative: bool) -> str:
    if relative:
        return path.relative_to(data_root).as_posix()
    return str(path.resolve())


def validate_roots(data_root: Path, subject_dir: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")
    if not subject_dir.exists():
        raise FileNotFoundError(f"subject folder does not exist: {subject_dir}")
    if not subject_dir.is_dir():
        raise NotADirectoryError(f"subject path is not a directory: {subject_dir}")


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], int, int]:
    data_root = args.data_root.resolve()
    subject_dir = data_root / args.subject
    validate_roots(data_root, subject_dir)

    rows: list[dict[str, str]] = []
    missing_target = 0
    missing_rgb_folders = 0

    for env in args.envs:
        env_dir = subject_dir / env
        if not env_dir.exists():
            logging.warning("Environment folder missing, skipping: %s", env_dir)
            continue
        if not env_dir.is_dir():
            logging.warning("Environment path is not a directory, skipping: %s", env_dir)
            continue

        clip_dirs = sorted(
            (path for path in env_dir.iterdir() if path.is_dir()),
            key=clip_sort_key,
        )
        if not clip_dirs:
            logging.warning("No clip folders found under environment: %s", env_dir)

        for clip_dir in clip_dirs:
            clip = clip_dir.name
            for input_cam in args.input_cams:
                rgb_dir = clip_dir / input_cam / "rgb"
                if not rgb_dir.exists() or not rgb_dir.is_dir():
                    logging.warning("RGB folder missing, skipping: %s", rgb_dir)
                    missing_rgb_folders += 1
                    continue

                image_paths = sorted(rgb_dir.glob("*.png"), key=lambda path: path.name)
                for image_path in image_paths:
                    frame_id = image_path.stem
                    target_path = (
                        clip_dir
                        / args.target_cam
                        / args.target_type
                        / f"{frame_id}.txt"
                    )
                    if not target_path.exists():
                        missing_target += 1
                        continue

                    rows.append(
                        {
                            "image_path": display_path(image_path, data_root, args.relative),
                            "target_path": display_path(target_path, data_root, args.relative),
                            "subject": args.subject,
                            "env": env,
                            "clip": clip,
                            "input_cam": input_cam,
                            "target_cam": args.target_cam,
                            "frame_id": frame_id,
                        }
                    )

    return rows, missing_target, missing_rgb_folders


def write_csv(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    args = parse_args()

    try:
        rows, missing_target, missing_rgb_folders = build_rows(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    write_csv(args.output_csv, rows)

    logging.info("Rows written: %d", len(rows))
    logging.info("Missing target txt files: %d", missing_target)
    logging.info("Missing RGB folders: %d", missing_rgb_folders)
    logging.info("Output CSV: %s", args.output_csv)


if __name__ == "__main__":
    main()
