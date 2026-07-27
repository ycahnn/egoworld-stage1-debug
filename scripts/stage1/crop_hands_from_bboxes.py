import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.hand_cropper import HandCropper, save_crop_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cropped hand images from bounding box JSON."
    )
    parser.add_argument(
        "--image", required=True, help="Path to the input RGB image."
    )
    parser.add_argument(
        "--bbox_json",
        required=True,
        help="Path to the hand_bboxes.json file from scripts/stage1/infer_hand_bbox.py.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for cropped hand images and metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    cropper = HandCropper(args.image)
    metadata_list, debug_image = cropper.crop_hands_from_json(
        args.bbox_json, args.out
    )

    metadata_path = os.path.join(args.out, "hand_crops_metadata.json")
    save_crop_metadata(metadata_list, metadata_path)

    debug_path = os.path.join(args.out, "crop_debug.png")
    cv2.imwrite(debug_path, debug_image)

    print(f"Extracted {len(metadata_list)} hand crop(s).")
    print(f"Saved crops to: {args.out}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved debug image to: {debug_path}")


if __name__ == "__main__":
    main()
