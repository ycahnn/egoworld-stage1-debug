import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.handpose_dataset import HandPoseDataset
from vit_handpose.models.vit_handpose import EgoHandPoseViT


def parse_group(path: str, level: str) -> str:
    parts = Path(path).parts
    if len(parts) < 4:
        return path
    subject, env, clip, cam = parts[:4]
    if level == "env":
        return f"{subject}/{env}"
    if level == "clip":
        return f"{subject}/{env}/{clip}"
    return f"{subject}/{env}/{clip}/{cam}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--already_resized", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--group_level", choices=("env", "clip", "seq"), default="clip")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    source_col = "original_image_path" if "original_image_path" in df.columns else "image_path"
    groups_by_index = df[source_col].map(lambda p: parse_group(p, args.group_level)).tolist()

    dataset = HandPoseDataset(
        csv_path=args.csv,
        image_root=args.image_root,
        train=False,
        already_resized=args.already_resized,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("args", {}).get("model_name", "vit_base_patch16_224.orig_in21k")
    model = EgoHandPoseViT(model_name=model_name, pretrained=False).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = defaultdict(lambda: {"sq": 0.0, "coords": 0, "dist": 0.0, "joints": 0, "rows": 0})
    row_offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Groups"):
            batch_size = batch["image"].shape[0]
            batch_groups = groups_by_index[row_offset:row_offset + batch_size]
            row_offset += batch_size

            images = batch["image"].cuda(non_blocking=True)
            targets = batch["target"].cuda(non_blocking=True).view(-1, 2, 21, 3)
            valid = batch["valid"].cuda(non_blocking=True).view(-1, 2, 1, 1)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                preds = model(images).view(-1, 2, 21, 3)

            errors = (preds - targets).float()
            sq = errors.square()
            dist = torch.linalg.norm(errors, dim=-1)
            valid_coord = valid.expand_as(errors)
            valid_hand = valid.squeeze(-1).squeeze(-1)

            for i, group in enumerate(batch_groups):
                coord_count = int(valid_coord[i].sum().item())
                joint_count = int(valid_hand[i].sum().item() * 21)
                stats[group]["sq"] += float((sq[i] * valid_coord[i]).sum().item())
                stats[group]["coords"] += coord_count
                stats[group]["dist"] += float((dist[i] * valid_hand[i].unsqueeze(-1).expand_as(dist[i])).sum().item())
                stats[group]["joints"] += joint_count
                stats[group]["rows"] += 1

    rows = []
    for group, item in stats.items():
        mse = item["sq"] / max(item["coords"], 1)
        rows.append({
            "group": group,
            "rows": item["rows"],
            "coord_mse": mse,
            "coord_rmse": float(np.sqrt(mse)),
            "mpjpe": item["dist"] / max(item["joints"], 1),
        })
    out = pd.DataFrame(rows).sort_values("coord_mse", ascending=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.8f}"))


if __name__ == "__main__":
    main()
