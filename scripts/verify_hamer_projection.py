#!/usr/bin/env python3
"""Verify HaMeR hand mesh projections against original exocentric image.

Run from project root with the main .venv:

    python scripts/verify_hamer_projection.py --image inputs/exo.jpg --hamer_dir outputs/hamer --bbox_json outputs/hand_bboxes.json --out outputs/hamer_projection
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Verify HaMeR projection against MediaPipe bboxes")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--hamer_dir", required=True, help="Path to HaMeR outputs directory")
    parser.add_argument("--bbox_json", required=True, help="Path to MediaPipe hand bboxes JSON")
    parser.add_argument("--out", required=True, help="Output directory for projection results")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_hamer_projection_keys(metadata):
    keys = set()
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if isinstance(v, (dict, list)):
                keys.add(k)
                if isinstance(v, dict):
                    keys.update(get_hamer_projection_keys(v))
            else:
                if any(sub in k.lower() for sub in ["cam", "camera", "focal", "center", "bbox", "trans", "keypoint", "vertices", "joints"]):
                    keys.add(k)
    return keys


def safe_path(path_str, base_dir):
    path = Path(path_str)
    if not path.is_absolute():
        return (base_dir / path).resolve()
    return path


def normalize_focal(focal):
    if focal is None:
        raise ValueError("Missing focal value for camera projection")
    focal_arr = np.asarray(focal, dtype=float)
    if focal_arr.ndim == 0:
        fx = fy = float(focal_arr)
    elif focal_arr.size == 1:
        fx = fy = float(focal_arr.reshape(-1)[0])
    else:
        fx, fy = float(focal_arr.reshape(-1)[:2])
    return fx, fy


def normalize_center(center, img_size=None):
    if center is None:
        if img_size is not None:
            img_arr = np.asarray(img_size, dtype=float)
            if img_arr.size >= 2:
                return float(img_arr[0]) / 2.0, float(img_arr[1]) / 2.0
        raise ValueError("Missing camera center and no img_size fallback available")
    center_arr = np.asarray(center, dtype=float)
    if center_arr.ndim == 0:
        raise ValueError("Camera center must contain both x and y values")
    if center_arr.size < 2:
        raise ValueError("Camera center must contain both x and y values")
    cx, cy = float(center_arr.reshape(-1)[:2][0]), float(center_arr.reshape(-1)[:2][1])
    return cx, cy


def normalize_cam_t(camera_translation):
    if camera_translation is None:
        raise ValueError("Missing camera translation for projection")
    cam_arr = np.asarray(camera_translation, dtype=float).reshape(-1)
    if cam_arr.size < 3:
        raise ValueError("Camera translation must contain at least 3 values")
    tx, ty, tz = float(cam_arr[0]), float(cam_arr[1]), float(cam_arr[2])
    return tx, ty, tz


def project_camera(vertices, cam_t, focal, center, img_size=None):
    tx, ty, tz = normalize_cam_t(cam_t)
    fx, fy = normalize_focal(focal)
    cx, cy = normalize_center(center, img_size=img_size)
    X_cam = vertices[:, 0] + tx
    Y_cam = vertices[:, 1] + ty
    Z_cam = vertices[:, 2] + tz
    Z_cam = np.where(Z_cam == 0, 1e-6, Z_cam)
    u = fx * X_cam / Z_cam + cx
    v = fy * Y_cam / Z_cam + cy
    return np.stack([u, v], axis=1)


def mirror_vertices_x(vertices):
    mirrored = vertices.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def project_bbox_fallback(vertices, bbox):
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max - x_min == 0:
        x_max = x_min + 1.0
    if y_max - y_min == 0:
        y_max = y_min + 1.0
    nx = (x - x_min) / (x_max - x_min)
    ny = (y - y_min) / (y_max - y_min)
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    u = left + nx * width
    v = top + ny * height
    return np.stack([u, v], axis=1)


def draw_wireframe(image, projected, faces, color, max_edges=4000):
    edges = set()
    h, w = image.shape[:2]
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    count = 0
    for i, j in sorted(edges):
        if count >= max_edges:
            break
        pt1 = projected[i]
        pt2 = projected[j]
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 1, lineType=cv2.LINE_AA)
            count += 1


def draw_points(image, projected, color, radius=1):
    h, w = image.shape[:2]
    for pt in projected:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def mirror_2d_projection(projected, bbox):
    left, top, right, bottom = bbox
    u = projected[:, 0]
    mirrored_u = left + right - u
    return np.stack([mirrored_u, projected[:, 1]], axis=1)


def has_native_left_mesh(entry):
    handedness = entry.get("output_handedness") or entry.get("hamer_output", {}).get("output_handedness") or {}
    return handedness.get("mesh_handedness") == "left" and bool(
        handedness.get("left_hand_x_mirror_applied_to_vertices_and_joints")
    )


def bbox_from_points(points):
    x = points[:, 0]
    y = points[:, 1]
    return [float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))]


def bbox_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / args.image
    hamerdir = project_root / args.hamer_dir
    bbox_json_path = project_root / args.bbox_json
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    bbox_data = load_json(bbox_json_path)
    metadata = load_json(hamerdir / "hamer_metadata.json")

    print("Loaded HaMeR metadata keys:")
    print(list(metadata.keys()))

    outputs = metadata.get("outputs", [])
    if not outputs:
        raise ValueError("No outputs found in hamer_metadata.json")

    final_projection_image = image.copy()
    projected_bbox_image = image.copy()
    report_lines = []
    report_lines.append("# HaMeR Projection Verification Report")
    report_lines.append("")
    report_lines.append(f"Image: {image_path}")
    report_lines.append(f"HaMeR metadata: {hamerdir / 'hamer_metadata.json'}")
    report_lines.append(f"BBox JSON: {bbox_json_path}")
    report_lines.append("")

    hand_index = 0
    for entry in outputs:
        hand_index += 1
        hand_label = entry.get("hand_side_for_hamer", "unknown")
        raw_handedness = entry.get("raw_handedness")
        bbox_xyxy = entry.get("bbox_xyxy", [0, 0, 0, 0])
        ham_out = entry.get("hamer_output", {})
        raw_keys = [k for k in ham_out.keys() if any(sub in k.lower() for sub in ["cam", "camera", "focal", "center", "bbox", "trans", "keypoint", "vertices", "joints"])]
        print(f"Hand {hand_index} ({hand_label}) HaMeR output keys:", raw_keys)
        report_lines.append(f"## Hand {hand_index} ({hand_label})")
        report_lines.append(f"hand_side_for_hamer: {hand_label}")
        report_lines.append(f"raw_handedness: {raw_handedness}")
        report_lines.append(f"MediaPipe bbox: {bbox_xyxy}")
        report_lines.append(f"Available HaMeR keys: {raw_keys}")

        vertices_path = (project_root / entry["vertices"]) if not Path(entry["vertices"]).is_absolute() else Path(entry["vertices"])
        faces_path = (project_root / entry["faces"]) if not Path(entry["faces"]).is_absolute() else Path(entry["faces"])
        vertices = np.load(str(vertices_path))
        faces = np.load(str(faces_path))

        prefixed = f"hand_{hand_index - 1:02d}"
        preprocessing = ham_out.get("preprocessing", {})
        original_width = preprocessing.get("original_image_width", image.shape[1])
        original_height = preprocessing.get("original_image_height", image.shape[0])
        crop_center = preprocessing.get("crop_center")
        img_size = preprocessing.get("img_size")
        scaled_focal_length = ham_out.get("scaled_focal_length") or entry.get("scaled_focal_length")

        camera_mode_ok = False
        camera_projection = None
        camera_info = {
            "translation_source": None,
            "pred_cam_t": None,
            "focal": None,
            "center": None,
            "used_center_source": None,
        }

        pred_cam_t_full = ham_out.get("pred_cam_t_full") or entry.get("pred_cam_t_full") or ham_out.get("full_img_cam_t") or entry.get("full_img_cam_t")
        pred_cam_t = ham_out.get("pred_cam_t")
        camera_translation = pred_cam_t_full if pred_cam_t_full is not None else pred_cam_t
        camera_info["translation_source"] = "pred_cam_t_full" if pred_cam_t_full is not None else ("pred_cam_t" if pred_cam_t is not None else None)

        focal = scaled_focal_length or ham_out.get("focal_length") or ham_out.get("focal")
        center = None
        if camera_info["translation_source"] == "pred_cam_t_full":
            if img_size is not None and len(img_size) >= 2:
                center = [float(img_size[0]) / 2.0, float(img_size[1]) / 2.0]
                camera_info["used_center_source"] = "img_size/2 (full_image)"
            else:
                center = [float(original_width) / 2.0, float(original_height) / 2.0]
                camera_info["used_center_source"] = "original_image_center"
        else:
            center = ham_out.get("center")
            if center is None and img_size is not None and len(img_size) >= 2:
                center = [float(img_size[0]) / 2.0, float(img_size[1]) / 2.0]
                camera_info["used_center_source"] = "img_size/2"
            if center is None and crop_center is not None and len(crop_center) >= 2:
                center = crop_center
                camera_info["used_center_source"] = "crop_center"
            if center is None:
                center = [float(original_width) / 2.0, float(original_height) / 2.0]
                camera_info["used_center_source"] = "original_image_center"

        camera_info["pred_cam_t"] = camera_translation
        camera_info["focal"] = focal
        camera_info["center"] = center

        original_projection = None
        left_bbox_2d_proj = None
        original_iou = None

        if camera_translation is not None and focal is not None and center is not None:
            camera_mode_ok = True
            if camera_info["translation_source"] == "pred_cam_t":
                print(f"Warning: using raw pred_cam_t for hand {hand_index} camera projection")
                report_lines.append(f"Warning: using raw pred_cam_t for hand {hand_index} camera projection")
            original_projection = project_camera(
                vertices,
                camera_translation,
                focal,
                center,
                img_size=img_size or [original_width, original_height],
            )
            if hand_label.lower() == "left" and not has_native_left_mesh(entry):
                left_bbox_2d_proj = mirror_2d_projection(original_projection, bbox_xyxy)

        bbox_proj = project_bbox_fallback(vertices, bbox_xyxy)

        def draw_hand_debug(base_img, projected_pts, mode_name, color):
            img = base_img
            # draw bbox
            left, top, right, bottom = [int(v) for v in bbox_xyxy]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 255), 2)
            draw_points(img, projected_pts, color, radius=1)
            draw_wireframe(img, projected_pts, faces, color)
            cv2.putText(img, f"Hand {hand_index-1}: {hand_label}", (left, max(top - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)
            cv2.putText(img, f"{mode_name}", (left, min(bottom + 20, img.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)
            return img

        if camera_mode_ok:
            final_projection = original_projection
            mirror_correction_applied = False
            if hand_label.lower() == "left" and left_bbox_2d_proj is not None:
                final_projection = left_bbox_2d_proj
                mirror_correction_applied = True
            final_projection_image = draw_hand_debug(final_projection_image, final_projection, "final_projection", (0, 255, 0) if hand_label.lower() != "left" else (0, 128, 255))
            proj_bbox = bbox_from_points(final_projection)
            overlap = bbox_iou(proj_bbox, bbox_xyxy)
            report_lines.append(f"camera_projection: ok")
            report_lines.append(f"  translation source: {camera_info['translation_source']}")
            report_lines.append(f"  focal source: {'scaled_focal_length' if scaled_focal_length is not None else 'raw_focal'}")
            report_lines.append(f"  projected bbox: {proj_bbox}")
            report_lines.append(f"  MediaPipe bbox: {bbox_xyxy}")
            report_lines.append(f"  left-hand 2D mirror applied: {mirror_correction_applied}")
            report_lines.append(f"  overlap IoU: {overlap:.3f}")
            report_lines.append(f"  camera center source: {camera_info['used_center_source']}")
        else:
            report_lines.append("camera_projection: unavailable due to missing pred_cam_t, focal, or center")

        projected_bbox_image = draw_hand_debug(projected_bbox_image, bbox_proj, "bbox_fallback_projection", (255, 0, 0))
        proj_bbox2 = bbox_from_points(bbox_proj)
        overlap2 = bbox_iou(proj_bbox2, bbox_xyxy)
        report_lines.append(f"bbox_fallback_projection: projected vertex bbox: {proj_bbox2}")
        report_lines.append(f"  overlap IoU with MediaPipe bbox: {overlap2:.3f}")
        report_lines.append("")

    final_out_path = out_dir / "final_projection_debug.png"
    bbox_out_path = out_dir / "bbox_fallback_projection_debug.png"
    report_path = out_dir / "final_projection_report.md"
    cv2.imwrite(str(final_out_path), final_projection_image)
    cv2.imwrite(str(bbox_out_path), projected_bbox_image)

    report_lines.append("")
    report_lines.append(f"final_projection_debug: {final_out_path}")
    report_lines.append(f"bbox_fallback_projection_debug: {bbox_out_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved projection report to {report_path}")
    print(f"Saved final projection debug image to {final_out_path}")
    print(f"Saved bbox fallback debug image to {bbox_out_path}")


if __name__ == "__main__":
    main()
