from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


Y_COLUMNS = [f"y{i}" for i in range(126)]


def split_root_pose(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split [..., 2, 21, 3] joints into wrists and 20 wrist-relative joints."""
    roots = joints[..., :, 0, :]
    relative = joints[..., :, 1:, :] - roots[..., :, None, :]
    return roots, relative


def reconstruct_joints(roots: np.ndarray, relative: np.ndarray) -> np.ndarray:
    wrists = roots[..., :, None, :]
    return np.concatenate([wrists, relative + wrists], axis=-2)


class TargetNormalizer:
    def __init__(
        self,
        root_mean: np.ndarray,
        root_std: np.ndarray,
        pose_mean: np.ndarray,
        pose_std: np.ndarray,
    ) -> None:
        self.root_mean = np.asarray(root_mean, dtype=np.float32).reshape(2, 3)
        self.root_std = np.asarray(root_std, dtype=np.float32).reshape(2, 3)
        self.pose_mean = np.asarray(pose_mean, dtype=np.float32).reshape(2, 20, 3)
        self.pose_std = np.asarray(pose_std, dtype=np.float32).reshape(2, 20, 3)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TargetNormalizer":
        return cls(
            values["root_mean"],
            values["root_std"],
            values["pose_mean"],
            values["pose_std"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TargetNormalizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_mean": self.root_mean.tolist(),
            "root_std": self.root_std.tolist(),
            "pose_mean": self.pose_mean.tolist(),
            "pose_std": self.pose_std.tolist(),
        }

    def normalize(self, roots: np.ndarray, relative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        roots = (roots - self.root_mean) / self.root_std
        relative = (relative - self.pose_mean) / self.pose_std
        return roots.astype(np.float32), relative.astype(np.float32)

    def denormalize_torch(
        self, roots: torch.Tensor, relative: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_mean = torch.as_tensor(self.root_mean, device=roots.device, dtype=roots.dtype)
        root_std = torch.as_tensor(self.root_std, device=roots.device, dtype=roots.dtype)
        pose_mean = torch.as_tensor(self.pose_mean, device=relative.device, dtype=relative.dtype)
        pose_std = torch.as_tensor(self.pose_std, device=relative.device, dtype=relative.dtype)
        return roots * root_std + root_mean, relative * pose_std + pose_mean


def compute_target_normalizer(
    csv_path: str | Path,
    chunk_size: int = 8192,
    min_std: float = 1e-6,
) -> TargetNormalizer:
    """Compute valid-hand train statistics without loading all 126 targets at once."""
    usecols = ["left_valid", "right_valid", *Y_COLUMNS]
    root_sum = np.zeros((2, 3), dtype=np.float64)
    root_sq_sum = np.zeros((2, 3), dtype=np.float64)
    pose_sum = np.zeros((2, 20, 3), dtype=np.float64)
    pose_sq_sum = np.zeros((2, 20, 3), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)

    for frame in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        joints = frame[Y_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 2, 21, 3)
        valid = frame[["left_valid", "right_valid"]].to_numpy(dtype=np.float32) > 0.5
        roots, relative = split_root_pose(joints)
        for hand in range(2):
            selected = valid[:, hand]
            if not selected.any():
                continue
            hand_roots = roots[selected, hand].astype(np.float64)
            hand_pose = relative[selected, hand].astype(np.float64)
            counts[hand] += selected.sum()
            root_sum[hand] += hand_roots.sum(axis=0)
            root_sq_sum[hand] += np.square(hand_roots).sum(axis=0)
            pose_sum[hand] += hand_pose.sum(axis=0)
            pose_sq_sum[hand] += np.square(hand_pose).sum(axis=0)

    if np.any(counts == 0):
        raise ValueError(f"Cannot normalize a hand with zero valid samples: counts={counts.tolist()}")

    root_mean = root_sum / counts[:, None]
    pose_mean = pose_sum / counts[:, None, None]
    root_var = root_sq_sum / counts[:, None] - np.square(root_mean)
    pose_var = pose_sq_sum / counts[:, None, None] - np.square(pose_mean)
    root_std = np.sqrt(np.maximum(root_var, min_std**2))
    pose_std = np.sqrt(np.maximum(pose_var, min_std**2))
    return TargetNormalizer(root_mean, root_std, pose_mean, pose_std)


def build_image_transform(
    mean: Sequence[float],
    std: Sequence[float],
    already_resized: bool = False,
):
    steps = []
    if not already_resized:
        steps.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


class RootPoseDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        normalizer: TargetNormalizer,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        already_resized: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.df = pd.read_csv(self.csv_path)
        missing = [column for column in ["image_path", "left_valid", "right_valid", *Y_COLUMNS] if column not in self.df]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        self.normalizer = normalizer
        self.transform = build_image_transform(image_mean, image_std, already_resized)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        joints = row[Y_COLUMNS].to_numpy(dtype=np.float32).reshape(2, 21, 3)
        roots, relative = split_root_pose(joints)
        norm_roots, norm_relative = self.normalizer.normalize(roots, relative)
        valid = row[["left_valid", "right_valid"]].to_numpy(dtype=np.float32)
        return {
            "image": image_tensor,
            "root_target": torch.from_numpy(norm_roots),
            "pose_target": torch.from_numpy(norm_relative),
            "valid": torch.from_numpy(valid),
            "image_path": str(row["image_path"]),
        }
