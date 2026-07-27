from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from vit_handpose_v2.data import RootPoseDataset, TargetNormalizer
from vit_handpose_v2.model import RootPoseViT
from vit_handpose_v2.train import loader_for, run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a root/pose ViT checkpoint in millimeters.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--already_resized", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    args.root_weight = train_args["root_weight"]
    args.pose_weight = train_args["pose_weight"]
    args.label_unit = train_args["label_unit"]
    normalizer = TargetNormalizer.from_dict(checkpoint["target_stats"])
    image_config = checkpoint["image_config"]
    dataset = RootPoseDataset(
        args.test_csv, args.image_root, normalizer,
        image_config["mean"], image_config["std"], args.already_resized,
    )
    loader = loader_for(dataset, args, False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RootPoseViT(train_args["model_name"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    metrics = run_epoch(model, loader, None, scaler, normalizer, args, device, False)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
