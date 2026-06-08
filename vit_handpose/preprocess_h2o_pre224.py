"""Preprocess H2O RGB/hand-pose pairs into 224x224 PNGs and direct target CSV.

The output directory is separate from the original H2O files. The saved PNGs are
plain RGB uint8 images; ImageNet Normalize is intentionally left for Dataset
transforms during training.

Quick test:
python vit_handpose/preprocess_h2o_pre224.py \
  --src_csv data/h2o_subject1_cam0123_to_cam4_hand_pose.csv \
  --out_root data/h2o_dataset/preprocessed_224_png \
  --image_root data/h2o_dataset \
  --label_root data/h2o_dataset \
  --limit 1000

Full conversion, preferably inside tmux:
python vit_handpose/preprocess_h2o_pre224.py \
  --src_csv data/h2o_subject1_cam0123_to_cam4_hand_pose.csv \
  --out_root data/h2o_dataset/preprocessed_224_png \
  --image_root data/h2o_dataset \
  --label_root data/h2o_dataset

Train from preprocessed outputs:
python vit_handpose/train.py \
  --train_csv data/h2o_dataset/preprocessed_224_png/h2o_subject1_cam0123_to_cam4_hand_pose_pre224.csv \
  --image_root data/h2o_dataset/preprocessed_224_png \
  --already_resized \
  --num_workers 2 \
  --pin_memory
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


Y_COLUMNS = [f"y{i}" for i in range(126)]
REQUIRED_COLUMNS = ["image_path", "left_valid", "right_valid", *Y_COLUMNS]

try:
    DEFAULT_RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9.1
    DEFAULT_RESAMPLE = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess H2O image/label pairs into 224x224 PNGs and direct target CSV."
    )
    parser.add_argument("--src_csv", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--image_root", default=".", type=Path)
    parser.add_argument("--label_root", default=None, type=Path)
    parser.add_argument("--output_csv", default=None, type=Path)
    parser.add_argument(
        "--image_subdir",
        default="",
        help="Optional subdirectory under images/, e.g. train or val, to avoid name collisions.",
    )
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument(
        "--interpolation",
        choices=["bicubic", "lanczos"],
        default="bicubic",
    )
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_sanity_check",
        action="store_true",
        help="Skip final output CSV/image/Dataset-like sanity checks.",
    )
    return parser.parse_args()


def resolve_path(path_value: str, root: Path | None, fallback_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if root is not None:
        return root / path
    return fallback_root / path


def load_h2o_label(label_path: Path) -> tuple[float, float, np.ndarray]:
    values = np.loadtxt(label_path, dtype=np.float32).reshape(-1)

    if values.shape[0] == 128:
        left_valid = float(values[0])
        right_valid = float(values[64])
        target = np.concatenate([values[1:64], values[65:128]], axis=0).astype(np.float32)
    elif values.shape[0] == 126:
        left_valid = 1.0
        right_valid = 1.0
        target = values.astype(np.float32)
    else:
        raise ValueError(
            f"Expected label length 126 or 128, got {values.shape[0]} at {label_path}"
        )

    if target.shape[0] != 126:
        raise ValueError(f"Expected target length 126, got {target.shape[0]} at {label_path}")

    return left_valid, right_valid, target


def row_to_task(
    idx: int,
    row: dict[str, Any],
    src_csv_parent: Path,
    image_root: Path,
    label_root: Path | None,
    images_dir: Path,
    image_rel_dir: Path,
    image_size: int,
    interpolation: str,
    overwrite: bool,
) -> dict[str, Any]:
    label_col = "label_path" if "label_path" in row and pd.notna(row["label_path"]) else "target_path"
    if label_col not in row or pd.isna(row[label_col]):
        raise ValueError("Input CSV must contain label_path or target_path for preprocessing.")

    out_rel = image_rel_dir / f"{idx:08d}.png"
    out_path = images_dir / out_rel.name

    return {
        "idx": idx,
        "image_path": resolve_path(str(row["image_path"]), image_root, src_csv_parent),
        "label_path": resolve_path(str(row[label_col]), label_root, src_csv_parent),
        "out_rel": out_rel.as_posix(),
        "out_path": out_path,
        "image_size": image_size,
        "interpolation": interpolation,
        "overwrite": overwrite,
        "original_image_path": str(row["image_path"]),
        "original_label_path": str(row[label_col]),
    }


def process_task(task: dict[str, Any]) -> dict[str, Any]:
    out_path = Path(task["out_path"])
    image_path = Path(task["image_path"])
    label_path = Path(task["label_path"])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    if task["overwrite"] or not out_path.exists():
        if task["interpolation"] == "lanczos":
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        else:
            resample = DEFAULT_RESAMPLE

        image = Image.open(image_path).convert("RGB")
        image = image.resize((task["image_size"], task["image_size"]), resample=resample)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")

    left_valid, right_valid, target = load_h2o_label(label_path)

    output = {
        "image_path": task["out_rel"],
        "left_valid": left_valid,
        "right_valid": right_valid,
    }
    output.update({f"y{i}": float(target[i]) for i in range(126)})
    output["original_image_path"] = task["original_image_path"]
    output["original_label_path"] = task["original_label_path"]
    return output


def output_csv_path(src_csv: Path, out_root: Path, output_csv: Path | None) -> Path:
    if output_csv is not None:
        return output_csv
    return out_root / f"{src_csv.stem}_pre224.csv"


def run_sanity_check(csv_path: Path, out_root: Path, image_size: int) -> None:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Preprocessed CSV is missing columns: {missing}")
    if len(df) == 0:
        raise ValueError(f"Preprocessed CSV is empty: {csv_path}")

    first_image = out_root / df.iloc[0]["image_path"]
    if not first_image.exists():
        raise FileNotFoundError(f"First preprocessed image not found: {first_image}")

    with Image.open(first_image) as image:
        if image.size != (image_size, image_size):
            raise ValueError(f"Expected image size {(image_size, image_size)}, got {image.size}")
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB PNG, got mode {image.mode}: {first_image}")

    print("Sanity check OK")
    print(f"  csv: {csv_path}")
    print(f"  rows: {len(df)}")
    print(f"  first_image: {first_image}")
    print(f"  columns: image_path,left_valid,right_valid,y0..y125")


def main() -> None:
    args = parse_args()
    src_csv = args.src_csv
    out_root = args.out_root
    image_rel_dir = Path("images") / args.image_subdir if args.image_subdir else Path("images")
    images_dir = out_root / image_rel_dir
    output_csv = output_csv_path(src_csv, out_root, args.output_csv)

    out_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv, dtype=str)
    if args.limit is not None:
        df = df.head(args.limit)

    if "image_path" not in df.columns:
        raise ValueError("Input CSV must contain image_path.")
    if "label_path" not in df.columns and "target_path" not in df.columns:
        raise ValueError("Input CSV must contain label_path or target_path.")

    tasks = [
        row_to_task(
            idx=idx,
            row=row.to_dict(),
            src_csv_parent=src_csv.parent,
            image_root=args.image_root,
            label_root=args.label_root,
            images_dir=images_dir,
            image_rel_dir=image_rel_dir,
            image_size=args.image_size,
            interpolation=args.interpolation,
            overwrite=args.overwrite,
        )
        for idx, row in df.reset_index(drop=True).iterrows()
    ]

    fieldnames = [*REQUIRED_COLUMNS, "original_image_path", "original_label_path"]
    workers = max(1, int(args.workers))

    rows_written = 0
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        if workers == 1:
            iterator = (process_task(task) for task in tasks)
            progress = tqdm(iterator, total=len(tasks), desc="preprocess")
            for output_row in progress:
                writer.writerow(output_row)
                rows_written += 1
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                progress = tqdm(executor.map(process_task, tasks), total=len(tasks), desc="preprocess")
                for output_row in progress:
                    writer.writerow(output_row)
                    rows_written += 1

    print(f"Rows written: {rows_written}")
    print(f"Images dir: {images_dir}")
    print(f"Output CSV: {output_csv}")

    if not args.no_sanity_check:
        run_sanity_check(output_csv, out_root, args.image_size)


if __name__ == "__main__":
    main()
