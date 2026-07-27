"""Split an H2O subject pair CSV using template env/clip splits.

Example:
python scripts/split_h2o_subject_csv.py \
  --src_csv data/h2o_subject4_cam0123_to_cam4_hand_pose.csv \
  --split_dir data/splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")
KEY_COLUMNS = ["env", "clip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test CSVs for one H2O subject from env/clip split templates."
    )
    parser.add_argument("--src_csv", required=True, type=Path)
    parser.add_argument(
        "--split_dir",
        default=Path("data/splits"),
        type=Path,
        help="Directory containing h2o_train.csv, h2o_val.csv, and h2o_test.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=Path,
        help="Output directory. Defaults to the source CSV directory.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Output prefix. Defaults to the source CSV stem.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def split_csv(src_csv: Path, split_dir: Path, output_dir: Path, output_prefix: str) -> None:
    base = pd.read_csv(src_csv, dtype=str)
    require_columns(base, KEY_COLUMNS, src_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_csv_path = split_dir / f"h2o_{split}.csv"
        template = pd.read_csv(split_csv_path, dtype=str)
        require_columns(template, KEY_COLUMNS, split_csv_path)

        keys = set(map(tuple, template[KEY_COLUMNS].drop_duplicates().to_numpy()))
        mask = base[KEY_COLUMNS].apply(tuple, axis=1).isin(keys)
        output = base[mask].copy()

        output_csv = output_dir / f"{output_prefix}_{split}.csv"
        output.to_csv(output_csv, index=False)
        print(f"{split}: {len(output)} rows -> {output_csv}")


def main() -> None:
    args = parse_args()
    src_csv = args.src_csv
    output_dir = args.output_dir if args.output_dir is not None else src_csv.parent
    output_prefix = args.output_prefix if args.output_prefix is not None else src_csv.stem

    split_csv(src_csv, args.split_dir, output_dir, output_prefix)


if __name__ == "__main__":
    main()
