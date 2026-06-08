from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(train: bool = True, already_resized: bool = False):
    """
    ViT-B/16 pretrained model input transform.
    Input image -> [3, 224, 224]

    already_resized=True is for preprocessed 224x224 PNGs. The saved PNGs are
    plain RGB uint8 images, so Normalize must still run during training.
    """
    steps = []
    if not already_resized:
        steps.append(transforms.Resize((224, 224)))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(steps)


class HandPoseDataset(Dataset):
    """
    Dataset for:
        Exocentric image -> 3D egocentric hand pose

    Supported CSV formats:

    Format A:
        image_path,label_path
        images/000001.jpg,labels/000001.txt

        H2O pair CSVs may also use target_path instead of label_path.

        label txt can contain:
            128 numbers: left_flag + left_63_xyz + right_flag + right_63_xyz
            126 numbers: left_63_xyz + right_63_xyz

    Format B:
        image_path,left_valid,right_valid,y0,y1,...,y125
        images/000001.png,1,1,0.1,0.2,...,0.3

        If y0~y125 are present, label_path/target_path txt files are never read.
        left_valid/right_valid are optional; when missing, both hands are valid.

    Return:
        image:  [3, 224, 224]
        target: [126]
        valid:  [2]  # [left_valid, right_valid]
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str = ".",
        label_root: Optional[str] = None,
        train: bool = True,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        already_resized: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.label_root = Path(label_root) if label_root is not None else None

        self.df = pd.read_csv(self.csv_path)
        self.transform = build_image_transform(
            train=train,
            already_resized=already_resized,
        )

        self.target_mean = target_mean
        self.target_std = target_std

        if "image_path" not in self.df.columns:
            raise ValueError("CSV must contain an 'image_path' column.")

        self.y_cols = [f"y{i}" for i in range(126)]
        self.has_direct_targets = all(col in self.df.columns for col in self.y_cols)
        self.label_path_col = None
        if "label_path" in self.df.columns:
            self.label_path_col = "label_path"
        elif "target_path" in self.df.columns:
            self.label_path_col = "target_path"
        self.has_label_path = self.label_path_col is not None

        if not self.has_direct_targets and not self.has_label_path:
            raise ValueError(
                "CSV must contain y0~y125 columns, a 'label_path' column, or a 'target_path' column."
            )

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)

        if not path.is_absolute():
            path = self.image_root / path

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image

    def _load_label_from_txt(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(label_path)

        if not path.is_absolute():
            if self.label_root is not None:
                path = self.label_root / path
            else:
                path = self.csv_path.parent / path

        if not path.exists():
            raise FileNotFoundError(f"Label not found: {path}")

        values = np.loadtxt(path, dtype=np.float32).reshape(-1)

        if values.shape[0] == 128:
            left_valid = float(values[0])
            left_xyz = values[1:64]

            right_valid = float(values[64])
            right_xyz = values[65:128]

            target = np.concatenate([left_xyz, right_xyz], axis=0).astype(np.float32)
            valid = np.array([left_valid, right_valid], dtype=np.float32)

        elif values.shape[0] == 126:
            target = values.astype(np.float32)
            valid = np.array([1.0, 1.0], dtype=np.float32)

        else:
            raise ValueError(
                f"Expected label length 126 or 128, but got {values.shape[0]} at {path}"
            )

        return target, valid

    def _load_label_from_csv_row(self, row) -> Tuple[np.ndarray, np.ndarray]:
        target = row[self.y_cols].values.astype(np.float32)

        if "left_valid" in row.index and "right_valid" in row.index:
            valid = np.array(
                [float(row["left_valid"]), float(row["right_valid"])],
                dtype=np.float32,
            )
        else:
            valid = np.array([1.0, 1.0], dtype=np.float32)

        return target, valid

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / (self.target_std + 1e-8)
        return target

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image = self._load_image(row["image_path"])

        if self.has_direct_targets:
            target, valid = self._load_label_from_csv_row(row)
        else:
            target, valid = self._load_label_from_txt(row[self.label_path_col])

        target = self._normalize_target(target)

        target = torch.tensor(target, dtype=torch.float32)  # [126]
        valid = torch.tensor(valid, dtype=torch.float32)    # [2]

        return {
            "image": image,
            "target": target,
            "valid": valid,
            "image_path": row["image_path"],
        }
