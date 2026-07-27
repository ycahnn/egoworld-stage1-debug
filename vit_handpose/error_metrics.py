import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.handpose_dataset import HandPoseDataset
from vit_handpose.models.vit_handpose import EgoHandPoseViT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--already_resized", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dataset = HandPoseDataset(
        csv_path=args.test_csv,
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

    coord_sq_errors = []
    coord_abs_errors = []
    joint_distances = []
    hand_joint_distances = [[], []]
    samples = 0
    valid_hands = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Errors"):
            images = batch["image"].cuda(non_blocking=True)
            targets = batch["target"].cuda(non_blocking=True).view(-1, 2, 21, 3)
            valid = batch["valid"].cuda(non_blocking=True).view(-1, 2, 1, 1)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                preds = model(images).view(-1, 2, 21, 3)

            errors = (preds - targets).float()
            coord_mask = valid.expand_as(errors).bool()
            valid_hand_mask = valid.squeeze(-1).squeeze(-1).bool()
            distances = torch.linalg.norm(errors, dim=-1)

            coord_sq_errors.append((errors[coord_mask] ** 2).cpu().numpy())
            coord_abs_errors.append(errors[coord_mask].abs().cpu().numpy())
            joint_distances.append(distances[valid_hand_mask].cpu().numpy())

            for hand_idx in range(2):
                hand_joint_distances[hand_idx].append(
                    distances[:, hand_idx, :][valid_hand_mask[:, hand_idx]].cpu().numpy()
                )

            samples += images.shape[0]
            valid_hands += int(valid_hand_mask.sum().item())

    coord_sq_errors = np.concatenate(coord_sq_errors)
    coord_abs_errors = np.concatenate(coord_abs_errors)
    joint_distances = np.concatenate(joint_distances)
    left_joint_distances = np.concatenate(hand_joint_distances[0])
    right_joint_distances = np.concatenate(hand_joint_distances[1])

    coord_mse = float(coord_sq_errors.mean())
    coord_rmse = float(np.sqrt(coord_mse))
    coord_mae = float(coord_abs_errors.mean())
    mpjpe = float(joint_distances.mean())

    metrics = {
        "samples": samples,
        "valid_hands": valid_hands,
        "coord_mse": coord_mse,
        "coord_rmse": coord_rmse,
        "coord_mae": coord_mae,
        "mpjpe_mean": mpjpe,
        "mpjpe_median": float(np.median(joint_distances)),
        "mpjpe_p90": float(np.percentile(joint_distances, 90)),
        "mpjpe_p95": float(np.percentile(joint_distances, 95)),
        "left_mpjpe_mean": float(left_joint_distances.mean()),
        "right_mpjpe_mean": float(right_joint_distances.mean()),
        "coord_rmse_cm_if_meter": coord_rmse * 100,
        "coord_mae_cm_if_meter": coord_mae * 100,
        "mpjpe_mean_cm_if_meter": mpjpe * 100,
        "mpjpe_median_cm_if_meter": float(np.median(joint_distances)) * 100,
        "mpjpe_p90_cm_if_meter": float(np.percentile(joint_distances, 90)) * 100,
    }
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
