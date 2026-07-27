import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_pose_estimator import HandPoseEstimator, save_hand_outputs, save_mesh_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer hand pose and save placeholder MANO outputs.")
    parser.add_argument("--image", required=True, help="Path to the input exocentric RGB image.")
    parser.add_argument("--out", required=True, help="Output directory for hand pose outputs.")
    parser.add_argument(
        "--hand_model",
        choices=["placeholder"],
        default="placeholder",
        help="Hand model to use. Currently only the placeholder backend is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    estimator = HandPoseEstimator(model_name=args.hand_model)

    try:
        result = estimator.predict(args.image)
    except NotImplementedError as exc:
        print("Hand pose inference is not available yet.")
        print(str(exc))
        print(
            "A real MANO-based estimator requires HaMeR, ACR, or another compatible model implementation."
        )
        return

    save_hand_outputs(result, args.out)

    mesh_path = os.path.join(args.out, "exo_hand_mesh.obj")
    save_mesh_obj(result["vertices"], result["faces"], mesh_path)

    print(f"Saved hand joints: {os.path.join(args.out, 'exo_hand_joints.npy')}")
    print(f"Saved hand vertices: {os.path.join(args.out, 'exo_hand_vertices.npy')}")
    print(f"Saved hand faces: {os.path.join(args.out, 'exo_hand_faces.npy')}")
    print(f"Saved hand metadata: {os.path.join(args.out, 'exo_hand_metadata.json')}")
    print(f"Saved hand mesh OBJ: {mesh_path}")


if __name__ == "__main__":
    main()
