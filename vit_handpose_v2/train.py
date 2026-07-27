from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_handpose_v2.data import RootPoseDataset, compute_target_normalizer
from vit_handpose_v2.losses import metric_sums, root_pose_loss
from vit_handpose_v2.model import RootPoseViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train root/pose-decoupled ViT hand pose.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv")
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--already_resized", action="store_true")
    parser.add_argument("--model_name", default="vit_base_patch16_224.orig_in21k")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--root_weight", type=float, default=0.1)
    parser.add_argument("--pose_weight", type=float, default=1.0)
    parser.add_argument("--label_unit", choices=["m", "mm"], default="m")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_dir", default="checkpoints/vit_handpose_v2")
    parser.add_argument("--early_stop_patience", type=int, default=15)
    return parser.parse_args()


def loader_for(dataset, args, shuffle: bool) -> DataLoader:
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    if args.num_workers > 0:
        kwargs.update(
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    return DataLoader(dataset, **kwargs)


def lr_factor(epoch: int, warmup_epochs: int, total_epochs: int, min_ratio: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def run_epoch(model, loader, optimizer, scaler, normalizer, args, device, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "root_loss": 0.0, "pose_loss": 0.0, "samples": 0.0,
              "root_distance_sum": 0.0, "pose_distance_sum": 0.0,
              "absolute_distance_sum": 0.0, "pa_distance_sum": 0.0, "valid_hands": 0.0}
    description = "Train" if train else "Validation"
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=description):
            if batch["valid"].sum().item() == 0:
                continue
            images = batch["image"].to(device, non_blocking=True)
            target_root = batch["root_target"].to(device, non_blocking=True)
            target_pose = batch["pose_target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred_root, pred_pose = model(images)
                losses = root_pose_loss(
                    pred_root, pred_pose, target_root, target_pose, valid,
                    args.root_weight, args.pose_weight,
                )
            if train:
                scaler.scale(losses.total).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = images.shape[0]
            totals["loss"] += losses.total.item() * batch_size
            totals["root_loss"] += losses.root.item() * batch_size
            totals["pose_loss"] += losses.pose.item() * batch_size
            totals["samples"] += batch_size
            raw_pred_root, raw_pred_pose = normalizer.denormalize_torch(pred_root.float(), pred_pose.float())
            raw_target_root, raw_target_pose = normalizer.denormalize_torch(target_root.float(), target_pose.float())
            distances = metric_sums(
                raw_pred_root, raw_pred_pose, raw_target_root, raw_target_pose, valid,
                compute_pa=not train,
            )
            for key, value in distances.items():
                totals[key] += value

    samples = max(totals["samples"], 1.0)
    valid_hands = max(totals["valid_hands"], 1.0)
    scale = 1000.0 if args.label_unit == "m" else 1.0
    metrics = {
        "loss": totals["loss"] / samples,
        "root_loss": totals["root_loss"] / samples,
        "pose_loss": totals["pose_loss"] / samples,
        "root_error_mm": totals["root_distance_sum"] / valid_hands * scale,
        "pose_mpjpe_mm": totals["pose_distance_sum"] / (valid_hands * 20) * scale,
        "absolute_mpjpe_mm": totals["absolute_distance_sum"] / (valid_hands * 21) * scale,
    }
    if not train:
        metrics["pa_mpjpe_mm"] = totals["pa_distance_sum"] / (valid_hands * 21) * scale
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Computing valid-hand target statistics from train split...")
    normalizer = compute_target_normalizer(args.train_csv)
    stats = normalizer.to_dict()
    (save_dir / "target_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    probe = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    config = timm.data.resolve_model_data_config(probe)
    del probe
    image_mean, image_std = tuple(config["mean"]), tuple(config["std"])
    print(f"device={device}, pretrained={not args.no_pretrained}, image_mean={image_mean}, image_std={image_std}")

    dataset_kwargs = dict(
        image_root=args.image_root,
        normalizer=normalizer,
        image_mean=image_mean,
        image_std=image_std,
        already_resized=args.already_resized,
    )
    train_loader = loader_for(RootPoseDataset(args.train_csv, **dataset_kwargs), args, True)
    val_loader = loader_for(RootPoseDataset(args.val_csv, **dataset_kwargs), args, False) if args.val_csv else None

    model = RootPoseViT(args.model_name, not args.no_pretrained, args.freeze_backbone).to(device)
    parameter_groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
        {"params": list(model.root_head.parameters()) + list(model.pose_head.parameters()), "lr": args.head_lr},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: lr_factor(epoch, args.warmup_epochs, args.epochs, args.min_lr_ratio),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    best_pose = float("inf")
    best_total = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, normalizer, args, device, True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, normalizer, args, device, False) if val_loader else None
        lrs = [group["lr"] for group in optimizer.param_groups]
        print(f"epoch={epoch} lr={lrs} train={train_metrics} val={val_metrics}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "target_stats": stats,
            "image_config": {"mean": image_mean, "std": image_std, "input_size": config["input_size"]},
            "args": vars(args),
        }
        torch.save(checkpoint, save_dir / "last.pt")
        monitored = val_metrics or train_metrics
        if monitored["loss"] < best_total:
            best_total = monitored["loss"]
            torch.save(checkpoint, save_dir / "best_total.pt")
        if monitored["pose_mpjpe_mm"] < best_pose:
            best_pose = monitored["pose_mpjpe_mm"]
            stale_epochs = 0
            torch.save(checkpoint, save_dir / "best_pose.pt")
        else:
            stale_epochs += 1
        scheduler.step()
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(f"Early stopping: pose MPJPE did not improve for {stale_epochs} epochs.")
            break


if __name__ == "__main__":
    main()
