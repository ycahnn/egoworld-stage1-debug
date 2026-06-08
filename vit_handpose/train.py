import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vit_handpose import EgoHandPoseViT
from datasets.handpose_dataset import HandPoseDataset
from losses.handpose_loss import masked_hand_mse_loss


def _cuda_sync_if_needed(device: str, enabled: bool) -> None:
    if enabled and device == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, use_amp, log_data_time):
    model.train()

    total_loss = 0.0
    total_samples = 0
    skipped_batches = 0
    data_time_total = 0.0
    compute_time_total = 0.0
    timed_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    end_time = time.perf_counter()

    for batch in pbar:
        data_time = time.perf_counter() - end_time

        if batch["valid"].sum().item() == 0:
            skipped_batches += 1
            end_time = time.perf_counter()
            continue

        compute_start = time.perf_counter()

        images = batch["image"].to(device, non_blocking=True)      # [B, 3, 224, 224]
        targets = batch["target"].to(device, non_blocking=True)    # [B, 126]
        valid = batch["valid"].to(device, non_blocking=True)       # [B, 2]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            preds = model(images)                   # [B, 126]
            loss = masked_hand_mse_loss(
                pred=preds,
                target=targets,
                valid=valid,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _cuda_sync_if_needed(device, log_data_time)
        compute_time = time.perf_counter() - compute_start

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        postfix = {
            "loss": f"{loss.item():.6f}",
            "valid_hands": int(valid.sum().item()),
        }
        if log_data_time:
            data_time_total += data_time
            compute_time_total += compute_time
            timed_batches += 1
            postfix.update({
                "data_s": f"{data_time:.3f}",
                "compute_s": f"{compute_time:.3f}",
            })
        pbar.set_postfix(postfix)

        end_time = time.perf_counter()

    avg_loss = total_loss / max(total_samples, 1)
    avg_data_time = data_time_total / max(timed_batches, 1)
    avg_compute_time = compute_time_total / max(timed_batches, 1)

    return avg_loss, skipped_batches, avg_data_time, avg_compute_time


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Validation"):
        if batch["valid"].sum().item() == 0:
            continue

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            preds = model(images)
            loss = masked_hand_mse_loss(preds, targets, valid)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    return avg_loss


def dataloader_kwargs(args, shuffle: bool) -> dict:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", default=None)

    parser.add_argument("--image_root", default=".")
    parser.add_argument("--label_root", default=None)
    parser.add_argument(
        "--already_resized",
        action="store_true",
        help="Use for preprocessed 224x224 images; skips Resize but keeps ToTensor+Normalize.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--log_data_time", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_dir", default="checkpoints")

    parser.add_argument(
        "--model_name",
        default="vit_base_patch16_224.orig_in21k",
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")
    print("device:", device)
    print("amp:", use_amp)
    print("already_resized:", args.already_resized)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = HandPoseDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        label_root=args.label_root,
        train=True,
        already_resized=args.already_resized,
    )

    train_loader = DataLoader(
        train_dataset,
        **dataloader_kwargs(args, shuffle=True),
    )

    val_loader = None
    if args.val_csv is not None:
        val_dataset = HandPoseDataset(
            csv_path=args.val_csv,
            image_root=args.image_root,
            label_root=args.label_root,
            train=False,
            already_resized=args.already_resized,
        )

        val_loader = DataLoader(
            val_dataset,
            **dataloader_kwargs(args, shuffle=False),
        )

    model = EgoHandPoseViT(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, skipped, avg_data_time, avg_compute_time = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            log_data_time=args.log_data_time,
        )

        msg = f"[Epoch {epoch}] train_loss={train_loss:.6f}, skipped_batches={skipped}"
        if args.log_data_time:
            msg += f", avg_data_time={avg_data_time:.4f}s, avg_compute_time={avg_compute_time:.4f}s"
        print(msg)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "args": vars(args),
        }

        torch.save(ckpt, save_dir / "last.pt")

        if val_loader is not None:
            val_loss = validate(model, val_loader, device, use_amp)
            print(f"[Epoch {epoch}] val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt["val_loss"] = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print(f"Saved best checkpoint: val_loss={val_loss:.6f}")


if __name__ == "__main__":
    main()
