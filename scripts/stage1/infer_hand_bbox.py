import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.hand_bbox_detector import HandBBoxDetector, save_hand_bboxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect hand bounding boxes using MediaPipe Hands.")
    parser.add_argument("--image", required=True, help="Path to the input exocentric RGB image.")
    parser.add_argument("--out", required=True, help="Output directory for hand bbox outputs.")
    parser.add_argument(
        "--no_flip_handedness",
        action="store_false",
        dest="flip_handedness",
        help="Do not flip MediaPipe handedness. By default handedness is flipped for normal exocentric images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {args.image}")
    image_height, image_width = image.shape[:2]

    detector = HandBBoxDetector(flip_handedness=args.flip_handedness)
    bboxes, debug_image = detector.predict(args.image)

    json_path = os.path.join(args.out, "hand_bboxes.json")
    save_hand_bboxes(
        args.image,
        image_width,
        image_height,
        args.flip_handedness,
        bboxes,
        json_path,
    )

    debug_path = os.path.join(args.out, "hand_bbox_debug.png")
    cv2.imwrite(debug_path, debug_image)

    print(f"Detected {len(bboxes)} hand(s).")
    for idx, bbox in enumerate(bboxes):
        print(json.dumps(bbox.to_dict(), indent=2))
    print(f"Saved hand bounding boxes JSON: {json_path}")
    print(f"Saved debug image: {debug_path}")


if __name__ == "__main__":
    main()
