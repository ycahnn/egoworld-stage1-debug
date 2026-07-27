import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.handpose_dataset import HandPoseDataset
from vit_handpose.losses.handpose_loss import masked_hand_mse_loss
from vit_handpose.models.vit_handpose import EgoHandPoseViT


def dataloader_kwargs(args) -> dict:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()

    total_sq_error = 0.0
    total_values = 0.0
    total_loss_weighted_by_values = 0.0
    total_loss_weighted_by_samples = 0.0
    total_samples = 0
    skipped_batches = 0

    for batch in tqdm(loader, desc="Test"):
        valid_hands = batch["valid"].sum().item()
        if valid_hands == 0:
            skipped_batches += 1
            continue

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            preds = model(images)
            loss = masked_hand_mse_loss(preds, targets, valid)

        batch_size = images.size(0)
        values = valid.sum().item() * 21 * 3
        total_loss_weighted_by_values += loss.item() * values
        total_loss_weighted_by_samples += loss.item() * batch_size
        total_sq_error += loss.item() * values
        total_values += values
        total_samples += batch_size

    mse = total_sq_error / max(total_values, 1.0)
    return {
        "test_loss_train_style": total_loss_weighted_by_samples / max(total_samples, 1),
        "test_loss_coord_weighted": total_loss_weighted_by_values / max(total_values, 1.0),
        "test_rmse": math.sqrt(mse),
        "evaluated_samples": total_samples,
        "skipped_batches": skipped_batches,
        "valid_values": int(total_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--label_root", default=None)
    parser.add_argument("--already_resized", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_name", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device == "cuda")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model_name = args.model_name or ckpt_args.get("model_name") or "vit_base_patch16_224.orig_in21k"

    print("device:", device)
    print("amp:", use_amp)
    print("checkpoint:", args.checkpoint)
    print("checkpoint_epoch:", checkpoint.get("epoch"))
    print("checkpoint_val_loss:", checkpoint.get("val_loss"))
    print("model_name:", model_name)
    print("test_csv:", args.test_csv)
    print("image_root:", args.image_root)
    print("already_resized:", args.already_resized)

    dataset = HandPoseDataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        label_root=args.label_root,
        train=False,
        already_resized=args.already_resized,
    )
    loader = DataLoader(dataset, **dataloader_kwargs(args))

    model = EgoHandPoseViT(model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, loader, device, use_amp)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
