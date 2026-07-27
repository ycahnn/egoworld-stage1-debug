import argparse
from pathlib import Path

import numpy as np
import pandas as pd


Y_COLS = [f"y{i}" for i in range(126)]


def parse_source_path(path: str) -> tuple[str, str, str, str, str]:
    parts = Path(path).parts
    subject = parts[0] if len(parts) > 0 else ""
    env = parts[1] if len(parts) > 1 else ""
    clip = parts[2] if len(parts) > 2 else ""
    cam = parts[3] if len(parts) > 3 else ""
    frame = Path(parts[-1]).stem if parts else ""
    return subject, env, clip, cam, frame


def load_split(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["split"] = split
    source_col = "original_image_path" if "original_image_path" in df.columns else "image_path"
    parsed = df[source_col].map(parse_source_path)
    df[["subject", "env", "clip", "cam", "frame"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["seq"] = df["subject"] + "/" + df["env"] + "/" + df["clip"] + "/" + df["cam"]
    df["clip_key"] = df["subject"] + "/" + df["env"] + "/" + df["clip"]
    df["frame_int"] = pd.to_numeric(df["frame"], errors="coerce")
    df["valid_hands"] = df.get("left_valid", 1.0).astype(float) + df.get("right_valid", 1.0).astype(float)
    y = df[Y_COLS].to_numpy(dtype=np.float32)
    df["target_l2"] = np.linalg.norm(y.reshape(len(df), 2, 21, 3), axis=-1).mean(axis=(1, 2))
    return df


def split_summary(df: pd.DataFrame) -> dict:
    seq_sizes = df.groupby("seq").size()
    clip_sizes = df.groupby("clip_key").size()
    return {
        "rows": len(df),
        "subjects": df["subject"].nunique(),
        "envs": df["env"].nunique(),
        "clips": df["clip_key"].nunique(),
        "seqs": df["seq"].nunique(),
        "left_valid_rate": float(df["left_valid"].astype(float).mean()) if "left_valid" in df else 1.0,
        "right_valid_rate": float(df["right_valid"].astype(float).mean()) if "right_valid" in df else 1.0,
        "both_valid_rate": float(((df.get("left_valid", 1.0).astype(float) > 0) & (df.get("right_valid", 1.0).astype(float) > 0)).mean()),
        "seq_min_rows": int(seq_sizes.min()) if len(seq_sizes) else 0,
        "seq_median_rows": float(seq_sizes.median()) if len(seq_sizes) else 0.0,
        "seq_max_rows": int(seq_sizes.max()) if len(seq_sizes) else 0,
        "clip_min_rows": int(clip_sizes.min()) if len(clip_sizes) else 0,
        "clip_median_rows": float(clip_sizes.median()) if len(clip_sizes) else 0.0,
        "clip_max_rows": int(clip_sizes.max()) if len(clip_sizes) else 0,
        "target_l2_mean": float(df["target_l2"].mean()),
        "target_l2_std": float(df["target_l2"].std()),
    }


def print_table(title: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    print(f"\n## {title}")
    if len(df) == 0:
        print("(empty)")
    else:
        print(df.head(max_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/hdd2/youngchan/h2o_dataset/preprocessed_224_png"))
    parser.add_argument("--prefix", default="h2o")
    args = parser.parse_args()

    paths = {
        "train": args.root / f"{args.prefix}_train_pre224.csv",
        "val": args.root / f"{args.prefix}_val_pre224.csv",
        "test": args.root / f"{args.prefix}_test_pre224.csv",
    }
    frames = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frames[split] = load_split(path, split)
        print(f"{split}: {path}")

    summary = pd.DataFrame([{"split": k, **split_summary(v)} for k, v in frames.items()])
    print_table("split summary", summary)

    counts = pd.concat(frames.values()).groupby(["split", "subject", "env"]).size().reset_index(name="rows")
    print_table("rows by subject/env", counts.sort_values(["split", "subject", "env"]))

    clip_counts = pd.concat(frames.values()).groupby(["split", "clip_key"]).size().reset_index(name="rows")
    print_table("clip counts", clip_counts.sort_values(["split", "clip_key"]), max_rows=120)

    train_clip = set(frames["train"]["clip_key"])
    train_seq = set(frames["train"]["seq"])
    overlap_rows = []
    for split in ("val", "test"):
        clip = set(frames[split]["clip_key"])
        seq = set(frames[split]["seq"])
        overlap_rows.append({
            "split": split,
            "clips": len(clip),
            "clip_overlap_with_train": len(clip & train_clip),
            "seqs": len(seq),
            "seq_overlap_with_train": len(seq & train_seq),
        })
    print_table("train overlap", pd.DataFrame(overlap_rows))

    val_clip = set(frames["val"]["clip_key"])
    test_clip = set(frames["test"]["clip_key"])
    print("\n## val/test overlap")
    print(f"clip_overlap: {len(val_clip & test_clip)}")
    print(f"seq_overlap: {len(set(frames['val']['seq']) & set(frames['test']['seq']))}")

    combined = pd.concat(frames.values())
    y = combined[Y_COLS].to_numpy(dtype=np.float32).reshape(len(combined), 2, 21, 3)
    for split, df in frames.items():
        idx = combined.index[combined["split"] == split].to_numpy()
        vals = y[idx]
        flat = vals.reshape(-1, 3)
        print(f"\n## target coordinate distribution: {split}")
        print(f"x mean/std/min/max: {flat[:,0].mean():.6f} {flat[:,0].std():.6f} {flat[:,0].min():.6f} {flat[:,0].max():.6f}")
        print(f"y mean/std/min/max: {flat[:,1].mean():.6f} {flat[:,1].std():.6f} {flat[:,1].min():.6f} {flat[:,1].max():.6f}")
        print(f"z mean/std/min/max: {flat[:,2].mean():.6f} {flat[:,2].std():.6f} {flat[:,2].min():.6f} {flat[:,2].max():.6f}")


if __name__ == "__main__":
    main()
