#!/usr/bin/env python3
"""Infer HaMeR MANO outputs from MediaPipe hand bboxes.

This script runs HaMeR core inference (no rendering) for each bbox in a
MediaPipe-generated JSON. It saves vertices, joints, faces and an OBJ
mesh per detected hand.

Usage (example):
conda activate egoworld-hamer
python infer_hamer_from_bboxes.py \
    --image inputs/exo.jpg \
    --bbox_json outputs/hand_bboxes.json \
    --out outputs/hamer

Stage 1 uses HaMeR as a pure inference backend. Renderer/OpenGL paths
are intentionally disabled.
"""
from pathlib import Path
import os
import sys
import json
import math
import argparse
from typing import List, Dict, Any

import cv2
import numpy as np
import torch

from src.hamer_stage1_compat import cam_crop_to_full, install_hamer_renderer_stub


PROJECT_ROOT = Path(__file__).resolve().parent
HAMER_ROOT = PROJECT_ROOT / "external" / "hamer"

# Make the local external HaMeR clone win over any installed hamer package.
hamer_root_str = str(HAMER_ROOT)
if hamer_root_str in sys.path:
    sys.path.remove(hamer_root_str)
sys.path.insert(0, hamer_root_str)
install_hamer_renderer_stub()

def save_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# OBJ exported by infer_hamer_from_bboxes.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ is 1-based
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _safe_key(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").replace(" ", "_").replace("-", "_")


def _extract_array(value: Any) -> np.ndarray | Any:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return arr
    if isinstance(value, np.ndarray):
        return value
    return value


def _save_debug_value(name: str, value: Any, debug_dir: Path, prefix: str) -> Any:
    value = _extract_array(value)
    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return value.tolist()
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{_safe_key(name)}.npy"
        path = debug_dir / filename
        np.save(path, value)
        return str(path.resolve())
    if isinstance(value, (list, tuple)):
        if len(value) <= 10 and all(not isinstance(x, (np.ndarray, torch.Tensor, dict, list, tuple)) for x in value):
            return list(value)
        arr = np.asarray(value)
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{_safe_key(name)}.npy"
        path = debug_dir / filename
        np.save(path, arr)
        return str(path.resolve())
    return value


def _shallow_value_to_json(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) <= 10 and all(isinstance(v, (bool, int, float, str)) for v in value):
            return list(value)
        return [str(v) for v in value]
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        if arr.size <= 10:
            return arr.tolist()
        return f"<tensor shape={arr.shape}>"
    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return value.tolist()
        return f"<ndarray shape={value.shape}>"
    if isinstance(value, dict):
        return {k: _shallow_value_to_json(v) for k, v in value.items()}
    return str(value)


def _field_matches(name: str) -> bool:
    substrings = ["cam", "camera", "focal", "scale", "center", "bbox", "trans", "keypoint", "vertices", "joints"]
    key = name.lower()
    return any(sub in key for sub in substrings)


def _compute_scaled_focal_length(model_cfg: Any, img_size: torch.Tensor) -> float:
    model_image_size = model_cfg.MODEL.IMAGE_SIZE
    if isinstance(model_image_size, (tuple, list)):
        model_image_size = max(model_image_size)
    return float(model_cfg.EXTRA.FOCAL_LENGTH) / float(model_image_size) * float(img_size.max())


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        cast = float(value)
        return None if math.isnan(cast) or math.isinf(cast) else cast
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)



def _summarize_output_dict(out: Dict[str, Any], debug_dir: Path, prefix: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in out.items():
        if _field_matches(key):
            summary[key] = _save_debug_value(key, _extract_output_value(value), debug_dir, prefix)
        elif isinstance(value, dict):
            nested = {}
            for sub_key, sub_value in value.items():
                full_key = f"{key}.{sub_key}"
                if _field_matches(full_key):
                    nested[sub_key] = _save_debug_value(full_key, _extract_output_value(sub_value), debug_dir, prefix)
            if nested:
                summary[key] = nested
    return summary


def _extract_output_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        return arr
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == 1:
            return value[0]
        return value
    return value


def _dump_raw_output_summary(raw_keys: List[str], output_path: Path, hand_summaries: List[Dict[str, Any]]) -> None:
    payload = {
        "raw_output_keys": raw_keys,
        "hand_summaries": hand_summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(payload)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HaMeR inference from hand bboxes JSON")
    p.add_argument("--image", required=True, help="Path to input RGB image")
    p.add_argument("--bbox_json", required=True, help="Path to hand_bboxes.json")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--rescale", type=float, default=2.0, help="Rescale factor for bbox crops")
    return p.parse_args()


def check_required_files() -> bool:
    ckpt = HAMER_ROOT / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    mano_right = HAMER_ROOT / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"
    mano_left = HAMER_ROOT / "_DATA" / "data" / "mano" / "MANO_LEFT.pkl"
    mano_mean = HAMER_ROOT / "_DATA" / "data" / "mano_mean_params.npz"
    missing = []
    if not ckpt.exists():
        missing.append(str(ckpt))
    if not mano_right.exists():
        missing.append(str(mano_right))
    if not mano_left.exists():
        missing.append(str(mano_left))
    if not mano_mean.exists():
        missing.append(str(mano_mean))
    if missing:
        print("Missing required HaMeR data files:")
        for m in missing:
            print(" - ", m)
        print("MANO_RIGHT.pkl and MANO_LEFT.pkl must be downloaded manually from MANO and placed at the paths above; they cannot be auto-downloaded because of the MANO license.")
        return False
    return True


def _require_finite_array(name: str, array: np.ndarray, ndim: int | None = None) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} is not a numpy array")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} has shape {array.shape}; expected {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values")


def _validate_bbox(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int, index: int) -> None:
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Hand bbox {index} is invalid: {[x1, y1, x2, y2]}")
    if x1 < 0 or y1 < 0 or x2 >= image_width or y2 >= image_height:
        raise ValueError(
            f"Hand bbox {index} is outside image bounds {image_width}x{image_height}: "
            f"{[x1, y1, x2, y2]}"
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input paths relative to project root if needed
    image_path = (PROJECT_ROOT / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
    bbox_json_path = (PROJECT_ROOT / args.bbox_json).resolve() if not Path(args.bbox_json).is_absolute() else Path(args.bbox_json)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not bbox_json_path.exists():
        raise FileNotFoundError(f"BBox JSON not found: {bbox_json_path}")

    if not check_required_files():
        raise SystemExit(1)

    ckpt = HAMER_ROOT / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    mano_mean = HAMER_ROOT / "_DATA" / "data" / "mano_mean_params.npz"

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"HAMER_ROOT = {HAMER_ROOT}")
    print(f"Checkpoint path = {ckpt}")
    print(f"MANO mean params path = {mano_mean}")
    original_cwd = Path.cwd()
    print(f"Current working directory before load_hamer: {original_cwd}")

    try:
        os.chdir(HAMER_ROOT)
        print(f"Changed working directory to HAMER_ROOT: {Path.cwd()}")

        install_hamer_renderer_stub()

        # Import HaMeR loader after switching to HAMER_ROOT so relative paths in
        # the HaMeR config resolve correctly.
        try:
            from hamer.models import load_hamer
        except Exception as exc:
            raise RuntimeError("Failed to import HaMeR from external/hamer. Check sys.path and hamer env dependencies.") from exc

        model, model_cfg = load_hamer(str(ckpt))
    finally:
        try:
            os.chdir(original_cwd)
        except Exception as restore_exc:
            print(f"Warning: failed to restore working directory to {original_cwd}: {restore_exc}")
        print(f"Current working directory after load_hamer: {Path.cwd()}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load bbox JSON
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    hands = bbox_data.get("hands", [])
    if len(hands) == 0:
        raise SystemExit("No hands found in bbox JSON; HaMeR inference was not run.")

    # Read image once
    img_cv2 = cv2.imread(str(image_path))
    if img_cv2 is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # Build boxes and handedness arrays
    boxes = []
    rights = []
    entries: List[Dict[str, Any]] = []
    for h in hands:
        x1 = int(h.get("x1", 0))
        y1 = int(h.get("y1", 0))
        x2 = int(h.get("x2", 0))
        y2 = int(h.get("y2", 0))
        side = h.get("hand_side_for_hamer", None)
        raw = h.get("raw_handedness", None)
        conf = float(h.get("confidence", 0.0) or 0.0)
        _validate_bbox(x1, y1, x2, y2, img_cv2.shape[1], img_cv2.shape[0], len(entries))
        boxes.append([x1, y1, x2, y2])
        # HaMeR expects right==1, left==0
        rights.append(1 if (side and side.lower() == "right") else 0)
        entries.append({"bbox": [x1, y1, x2, y2], "hand_side_for_hamer": side, "raw_handedness": raw, "confidence": conf})

    boxes_np = np.array(boxes, dtype=np.float32)
    rights_np = np.array(rights, dtype=np.float32)

    # Use the ViTDetDataset preprocessing to generate model inputs (no detectron2/ViTPose involved)
    try:
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
    except Exception as exc:
        raise RuntimeError("Failed to import HaMeR dataset utilities from external/hamer") from exc

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes_np, rights_np, rescale_factor=args.rescale)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    debug_dir = out_dir / "debug"
    raw_summary_path = out_dir / "hamer_raw_output_summary.json"
    all_metadata = {
        "image": str(image_path),
        "raw_output_summary": str(raw_summary_path),
        "outputs": [],
    }
    raw_output_keys = []
    raw_hand_summaries: List[Dict[str, Any]] = []

    for idx, batch in enumerate(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        out_keys = sorted(list(out.keys()))
        print("Available HaMeR output keys:", out_keys)
        raw_output_keys.extend(k for k in out_keys if k not in raw_output_keys)

        hand_summary = _summarize_output_dict(out, debug_dir, f"hand_{idx:02d}")

        bbox_center = _shallow_value_to_json(_extract_output_value(batch.get("box_center"))) if "box_center" in batch else None
        bbox_size = _shallow_value_to_json(_extract_output_value(batch.get("box_size"))) if "box_size" in batch else None
        img_size_json = _shallow_value_to_json(_extract_output_value(batch.get("img_size"))) if "img_size" in batch else None

        scaled_focal_length = None
        pred_cam_t_full = None
        if "pred_cam" in out and "box_center" in batch and "box_size" in batch and "img_size" in batch:
            try:
                scaled_focal_length = _compute_scaled_focal_length(model_cfg, batch["img_size"].float())
                pred_cam = out["pred_cam"]
                full_cam_t = cam_crop_to_full(
                    pred_cam,
                    batch["box_center"].float(),
                    batch["box_size"].float(),
                    batch["img_size"].float(),
                    scaled_focal_length,
                )
                pred_cam_t_full = _extract_output_value(full_cam_t[0])
                hand_summary["pred_cam_t_full"] = _save_debug_value("pred_cam_t_full", pred_cam_t_full, debug_dir, f"hand_{idx:02d}")
                hand_summary["scaled_focal_length"] = float(scaled_focal_length)
            except Exception as exc:
                print(f"Warning: failed to compute pred_cam_t_full for hand {idx}: {exc}")

        # Preprocessing information from the crop batch
        preprocessing: Dict[str, Any] = {
            "original_image_width": int(img_cv2.shape[1]),
            "original_image_height": int(img_cv2.shape[0]),
            "bbox_xyxy": entries[idx]["bbox"],
            "hand_side_for_hamer": entries[idx].get("hand_side_for_hamer"),
            "raw_handedness": entries[idx].get("raw_handedness"),
            "confidence": entries[idx].get("confidence"),
            "crop_resolution": tuple(batch["img"].shape[-2:]) if "img" in batch else None,
            "crop_center": _shallow_value_to_json(_extract_output_value(batch.get("box_center"))) if "box_center" in batch else None,
            "crop_size": _shallow_value_to_json(_extract_output_value(batch.get("box_size"))) if "box_size" in batch else None,
            "bbox_center": bbox_center,
            "bbox_size": bbox_size,
            "img_size": img_size_json,
            "scaled_focal_length": float(scaled_focal_length) if scaled_focal_length is not None else None,
        }
        hand_summary["preprocessing"] = preprocessing

        # Extract arrays (first / only element in batch) and verify that real outputs exist.
        for required_key in ("pred_vertices", "pred_keypoints_3d"):
            if required_key not in out:
                raise KeyError(f"HaMeR output missing required key: {required_key}")
        if not hasattr(model, "mano") or not hasattr(model.mano, "faces"):
            raise AttributeError("Loaded HaMeR model does not expose model.mano.faces")

        verts = out["pred_vertices"][0].detach().cpu().numpy()
        joints = out["pred_keypoints_3d"][0].detach().cpu().numpy()
        faces = np.asarray(model.mano.faces).copy()
        _require_finite_array("pred_vertices", verts, ndim=2)
        _require_finite_array("pred_keypoints_3d", joints, ndim=2)
        _require_finite_array("mano.faces", faces, ndim=2)
        if verts.shape[1] != 3:
            raise ValueError(f"pred_vertices has shape {verts.shape}; expected Nx3")
        if joints.shape[1] != 3:
            raise ValueError(f"pred_keypoints_3d has shape {joints.shape}; expected Nx3")
        if faces.shape[1] != 3:
            raise ValueError(f"mano.faces has shape {faces.shape}; expected Fx3")

        side = entries[idx].get("hand_side_for_hamer") or "unknown"
        safe_side = (side.lower() if isinstance(side, str) else "unknown")
        idx_str = f"{idx:02d}_{safe_side}"

        verts_path = out_dir / f"hand_{idx_str}_vertices.npy"
        joints_path = out_dir / f"hand_{idx_str}_joints.npy"
        faces_path = out_dir / f"hand_{idx_str}_faces.npy"
        obj_path = out_dir / f"hand_{idx_str}_mesh.obj"

        np.save(verts_path, verts)
        np.save(joints_path, joints)
        np.save(faces_path, faces)
        save_obj(verts, faces, obj_path)

        all_metadata["outputs"].append({
            "index": idx,
            "hand_side_for_hamer": entries[idx].get("hand_side_for_hamer"),
            "raw_handedness": entries[idx].get("raw_handedness"),
            "confidence": entries[idx].get("confidence"),
            "bbox_xyxy": entries[idx]["bbox"],
            "bbox_center": bbox_center,
            "bbox_size": bbox_size,
            "img_size": img_size_json,
            "scaled_focal_length": float(scaled_focal_length) if scaled_focal_length is not None else None,
            "pred_cam_t_full": pred_cam_t_full,
            "vertices": str(verts_path),
            "joints": str(joints_path),
            "faces": str(faces_path),
            "obj": str(obj_path),
            "hamer_output": hand_summary,
        })

        raw_hand_summaries.append({
            "index": idx,
            "hand_side_for_hamer": entries[idx].get("hand_side_for_hamer"),
            "raw_handedness": entries[idx].get("raw_handedness"),
            "confidence": entries[idx].get("confidence"),
            "summary": hand_summary,
        })

    # Save metadata atomically after normalizing JSON values.
    metadata_path = out_dir / "hamer_metadata.json"
    metadata_tmp_path = out_dir / "hamer_metadata.json.tmp"
    normalized_metadata = to_jsonable(all_metadata)
    with open(metadata_tmp_path, "w", encoding="utf-8") as f:
        json.dump(normalized_metadata, f, indent=2, ensure_ascii=False, allow_nan=False)
    metadata_tmp_path.replace(metadata_path)

    _dump_raw_output_summary(raw_output_keys, raw_summary_path, raw_hand_summaries)
    print(f"Saved camera/debug metadata to {raw_summary_path}")

    # Draw debug bboxes on image
    debug_img = img_cv2.copy()
    for i, e in enumerate(entries):
        x1, y1, x2, y2 = e["bbox"]
        label = f"{i:02d}_{e.get('hand_side_for_hamer') or 'unk'} {e.get('confidence',0.0):.2f}"
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, label, (x1, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    debug_path = out_dir / "hamer_bbox_debug.png"
    cv2.imwrite(str(debug_path), debug_img)

    print(f"Saved {len(all_metadata['outputs'])} hand outputs to {out_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
