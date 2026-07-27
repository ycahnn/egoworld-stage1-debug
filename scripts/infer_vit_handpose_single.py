#!/usr/bin/env python3
"""Run the fine-tuned ViT hand-pose model on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.handpose_dataset import build_image_transform
from vit_handpose.models.vit_handpose import EgoHandPoseViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="inputs/exo.jpg", type=Path)
    parser.add_argument("--checkpoint", default="checkpoints/vit_handpose_subjects1234_pretrained/best.pt", type=Path)
    parser.add_argument("--out", default="outputs/vit_handpose_single", type=Path)
    parser.add_argument("--model_name", default="vit_base_patch16_224.orig_in21k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_state(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def main() -> None:
    args = parse_args()
    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image.resize((224, 224), Image.BILINEAR).save(out_dir / "input_resized_224.png")

    transform = build_image_transform(train=False, already_resized=False)
    x = transform(image).unsqueeze(0).to(args.device)

    model = EgoHandPoseViT(model_name=args.model_name, pretrained=False).to(args.device)
    model.load_state_dict(load_state(checkpoint_path), strict=True)
    model.eval()

    with torch.no_grad():
        pred = model(x).detach().cpu().numpy()[0].astype(np.float32)

    joints = pred.reshape(2, 21, 3)
    np.save(out_dir / "prediction_126.npy", pred)
    np.savetxt(out_dir / "prediction_126.txt", pred.reshape(1, -1), fmt="%.8f")
    np.save(out_dir / "joints_2x21x3.npy", joints)
    np.save(out_dir / "left_joints_21x3.npy", joints[0])
    np.save(out_dir / "right_joints_21x3.npy", joints[1])
    np.savetxt(out_dir / "left_joints_21x3.txt", joints[0], fmt="%.8f")
    np.savetxt(out_dir / "right_joints_21x3.txt", joints[1], fmt="%.8f")

    metadata = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "model_name": args.model_name,
        "input_resize": [224, 224],
        "outputs": {
            "joints": "joints_2x21x3.npy",
            "left": "left_joints_21x3.npy",
            "right": "right_joints_21x3.npy",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved ViT hand pose to {out_dir / 'joints_2x21x3.npy'}")


if __name__ == "__main__":
    main()
