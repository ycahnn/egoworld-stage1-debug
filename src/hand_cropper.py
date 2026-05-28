from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class HandCropMetadata:
    crop_path: str
    original_image_path: str
    x1: int
    y1: int
    x2: int
    y2: int
    handedness: str | None
    hand_side_for_hamer: str | None
    raw_handedness: str | None
    confidence: float | None
    crop_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_path": self.crop_path,
            "original_image_path": self.original_image_path,
            "bbox": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "handedness": self.handedness,
            "hand_side_for_hamer": self.hand_side_for_hamer,
            "raw_handedness": self.raw_handedness,
            "confidence": self.confidence,
        }


class HandCropper:
    """Extracts cropped hand images from bounding boxes."""

    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")
        self.image_path = image_path
        self.height, self.width = self.image.shape[:2]

    def crop_hand(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crop a hand region from the image."""
        x1_clamped = max(0, min(self.width - 1, x1))
        x2_clamped = max(0, min(self.width - 1, x2))
        y1_clamped = max(0, min(self.height - 1, y1))
        y2_clamped = max(0, min(self.height - 1, y2))

        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            raise ValueError(f"Invalid hand crop bbox: {[x1, y1, x2, y2]}")

        return self.image[y1_clamped : y2_clamped + 1, x1_clamped : x2_clamped + 1]

    def crop_hands_from_json(
        self, json_path: str, output_dir: str
    ) -> tuple[list[HandCropMetadata], np.ndarray]:
        """Extract cropped hands from bounding box JSON and save them."""
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            bbox_data = json.load(f)

        hands_data = bbox_data.get("hands", [])
        metadata_list: list[HandCropMetadata] = []
        debug_image = self.image.copy()

        for crop_index, hand in enumerate(hands_data):
            x1 = hand.get("x1")
            y1 = hand.get("y1")
            x2 = hand.get("x2")
            y2 = hand.get("y2")
            handedness = hand.get("handedness")
            hand_side_for_hamer = hand.get("hand_side_for_hamer")
            raw_handedness = hand.get("raw_handedness")
            confidence = hand.get("confidence")

            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue

            crop = self.crop_hand(x1, y1, x2, y2)

            crop_filename = f"hand_{crop_index:02d}_{hand_side_for_hamer or 'unknown'}.png"
            crop_path = output_dir_path / crop_filename
            cv2.imwrite(str(crop_path), crop)

            metadata = HandCropMetadata(
                crop_path=str(crop_path),
                original_image_path=self.image_path,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                handedness=handedness,
                hand_side_for_hamer=hand_side_for_hamer,
                raw_handedness=raw_handedness,
                confidence=confidence,
                crop_index=crop_index,
            )
            metadata_list.append(metadata)

            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Crop {crop_index}: {handedness or 'unknown'}"
            cv2.putText(
                debug_image,
                label_text,
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return metadata_list, debug_image


def save_crop_metadata(metadata_list: list[HandCropMetadata], json_path: str) -> None:
    """Save crop metadata to JSON."""
    data = {
        "num_crops": len(metadata_list),
        "crops": [m.to_dict() for m in metadata_list],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
