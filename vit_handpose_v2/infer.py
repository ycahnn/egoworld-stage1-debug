from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch

from vit_handpose_v2.data import TargetNormalizer, build_image_transform
from vit_handpose_v2.model import RootPoseViT


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer absolute 2x21x3 joints with a root/pose ViT.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", default="outputs/vit_handpose_v2", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    normalizer = TargetNormalizer.from_dict(checkpoint["target_stats"])
    image_config = checkpoint["image_config"]
    transform = build_image_transform(image_config["mean"], image_config["std"], already_resized=False)
    with Image.open(args.image) as image:
        batch = transform(image.convert("RGB")).unsqueeze(0).to(args.device)

    model = RootPoseViT(train_args["model_name"], pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        norm_root, norm_pose = model(batch)
        root, pose = normalizer.denormalize_torch(norm_root, norm_pose)
        joints = torch.cat([root[:, :, None], pose + root[:, :, None]], dim=2)[0].cpu().numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "joints_2x21x3.npy", joints.astype(np.float32))
    np.save(args.out / "roots_2x3.npy", joints[:, 0].astype(np.float32))
    np.save(args.out / "root_relative_2x20x3.npy", (joints[:, 1:] - joints[:, :1]).astype(np.float32))
    metadata = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_name": train_args["model_name"],
        "label_unit": train_args["label_unit"],
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.out / 'joints_2x21x3.npy'}")


if __name__ == "__main__":
    main()
