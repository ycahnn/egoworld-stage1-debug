#!/usr/bin/env python3
"""Render hand depth from HaMeR vertex outputs and camera metadata."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Render hand depth from HaMeR outputs")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--hamer_dir", required=True, help="Path to HaMeR outputs directory")
    parser.add_argument("--bbox_json", required=True, help="Path to MediaPipe hand bboxes JSON")
    parser.add_argument("--out", required=True, help="Output directory for rendered depth results")
    parser.add_argument(
        "--reference_depth",
        default=None,
        help="Optional depth npy in the target point-cloud frame, e.g. MoGE depth_raw.npy.",
    )
    parser.add_argument(
        "--align_depth_to_reference",
        action="store_true",
        help="Per-hand median-align rendered HaMeR depth to --reference_depth before saving hand_depth.npy.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str, base_dir):
    path = Path(path_str)
    if not path.is_absolute():
        return (base_dir / path).resolve()
    return path


def normalize_focal(focal):
    if focal is None:
        raise ValueError("Missing focal value for camera projection")
    arr = np.asarray(focal, dtype=float)
    if arr.ndim == 0:
        return float(arr), float(arr)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    return float(arr[0]), float(arr[1])


def normalize_center(center, img_size, original_size):
    if center is not None:
        arr = np.asarray(center, dtype=float).reshape(-1)
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])
    if img_size is not None:
        arr = np.asarray(img_size, dtype=float).reshape(-1)
        if arr.size >= 2:
            return float(arr[0]) / 2.0, float(arr[1]) / 2.0
    width, height = original_size
    return float(width) / 2.0, float(height) / 2.0


def normalize_cam_t(cam_t):
    if cam_t is None:
        raise ValueError("Missing camera translation for camera projection")
    arr = np.asarray(cam_t, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("Camera translation must contain at least 3 values")
    return float(arr[0]), float(arr[1]), float(arr[2])


def project_vertices(vertices, cam_t, focal, center, img_size, original_size):
    tx, ty, tz = normalize_cam_t(cam_t)
    fx, fy = normalize_focal(focal)
    cx, cy = normalize_center(center, img_size, original_size)
    x_cam = vertices[:, 0] + tx
    y_cam = vertices[:, 1] + ty
    z_cam = vertices[:, 2] + tz
    z_cam = np.where(z_cam == 0.0, 1e-6, z_cam)
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    return np.stack([u, v], axis=1), z_cam


def bbox_2d_mirror(projected, bbox):
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


def triangle_bbox(p0, p1, p2, width, height):
    min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
    max_x = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), width - 1)
    min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
    max_y = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), height - 1)
    return min_x, max_x, min_y, max_y


def barycentric_weights(px, py, p0, p1, p2, denom):
    w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def rasterize_mesh(projected, depth_values, faces, zbuffer, hand_buffer, hand_id):
    height, width = zbuffer.shape
    for face in faces:
        i0, i1, i2 = [int(idx) for idx in face]
        p0 = projected[i0]
        p1 = projected[i1]
        p2 = projected[i2]
        z0 = depth_values[i0]
        z1 = depth_values[i1]
        z2 = depth_values[i2]
        denom = ((p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]))
        if abs(denom) < 1e-9:
            continue
        min_x, max_x, min_y, max_y = triangle_bbox(p0, p1, p2, width, height)
        if min_x > max_x or min_y > max_y:
            continue
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w0, w1, w2 = barycentric_weights(x + 0.5, y + 0.5, p0, p1, p2, denom)
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z <= 0.0:
                    continue
                if z < zbuffer[y, x]:
                    zbuffer[y, x] = z
                    hand_buffer[y, x] = hand_id


def compute_depth_alignment_scale(projected, depth_values, faces, reference_depth):
    height, width = reference_depth.shape
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    hand_buffer = np.full((height, width), -1, dtype=np.int32)
    rasterize_mesh(projected, depth_values, faces, zbuffer, hand_buffer, 0)
    valid = (
        (hand_buffer == 0)
        & np.isfinite(zbuffer)
        & (zbuffer > 0)
        & np.isfinite(reference_depth)
        & (reference_depth > 0)
    )
    if not np.any(valid):
        return 1.0, 0, None
    ratios = reference_depth[valid].astype(np.float64) / zbuffer[valid].astype(np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    ratios = ratios[(ratios >= 0.01) & (ratios <= 100.0)]
    if ratios.size == 0:
        return 1.0, 0, None
    p05, p50, p95 = np.percentile(ratios, [5, 50, 95])
    filtered = ratios[(ratios >= p05) & (ratios <= p95)]
    if filtered.size == 0:
        filtered = ratios
    return float(np.median(filtered)), int(filtered.size), {
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def draw_wireframe(image, projected, faces, color):
    h, w = image.shape[:2]
    edges = set()
    for face in faces:
        a, b, c = [int(idx) for idx in face]
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    for a, b in edges:
        p0 = projected[a]
        p1 = projected[b]
        if 0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.line(image, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), color, 1, lineType=cv2.LINE_AA)


def draw_projection(image, projected, faces, bbox, label, color):
    img = image
    left, top, right, bottom = [int(v) for v in bbox]
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 255), 2)
    draw_wireframe(img, projected, faces, color)
    for pt in projected:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (left, max(top - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)
    return img


def normalize_depth_vis(depth_map):
    vis = np.zeros(depth_map.shape, dtype=np.uint8)
    finite = np.isfinite(depth_map)
    if not np.any(finite):
        return np.dstack([vis, vis, vis])
    valid_depths = depth_map[finite]
    min_val = float(np.min(valid_depths))
    max_val = float(np.max(valid_depths))
    if max_val <= min_val:
        vis[finite] = 255
        return np.dstack([vis, vis, vis])
    normalized = np.zeros_like(depth_map, dtype=np.float32)
    normalized[finite] = (depth_map[finite] - min_val) / (max_val - min_val)
    scaled = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


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

    metadata = load_json(hamerdir / "hamer_metadata.json")
    bbox_data = load_json(bbox_json_path)
    outputs = metadata.get("outputs", [])
    if not outputs:
        raise ValueError("No outputs found in hamer_metadata.json")

    reference_depth = None
    if args.reference_depth is not None:
        reference_depth_path = project_root / args.reference_depth
        reference_depth = np.load(str(reference_depth_path))
        if reference_depth.shape != image.shape[:2]:
            raise ValueError(f"reference_depth shape {reference_depth.shape} does not match image shape {image.shape[:2]}")
        if not args.align_depth_to_reference:
            print("Warning: --reference_depth was provided without --align_depth_to_reference; it will only be recorded.")

    image_shape = image.shape
    height, width = image_shape[0], image_shape[1]
    inf = np.inf
    zbuffer = np.full((height, width), inf, dtype=np.float32)
    hand_buffer = np.full((height, width), -1, dtype=np.int32)

    depth_debug_image = image.copy()
    hand_records = []

    for hand_index, entry in enumerate(outputs):
        hand_label = entry.get("hand_side_for_hamer", "unknown")
        raw_handedness = entry.get("raw_handedness")
        bbox_xyxy = entry.get("bbox_xyxy", [0, 0, 0, 0])
        vertices_path = resolve_path(entry["vertices"], project_root)
        faces_path = resolve_path(entry["faces"], project_root)
        vertices = np.load(str(vertices_path))
        faces = np.load(str(faces_path))

        # Access nested hamer_output structure
        ham_out = entry.get("hamer_output", {})
        preprocessing = ham_out.get("preprocessing", {})
        original_width = preprocessing.get("original_image_width", width)
        original_height = preprocessing.get("original_image_height", height)
        img_size = preprocessing.get("img_size")
        scaled_focal_length = ham_out.get("scaled_focal_length") or entry.get("scaled_focal_length")

        pred_cam_t_full = ham_out.get("pred_cam_t_full") or entry.get("pred_cam_t_full") or ham_out.get("full_img_cam_t") or entry.get("full_img_cam_t")
        pred_cam_t = ham_out.get("pred_cam_t")
        cam_t = pred_cam_t_full if pred_cam_t_full is not None else pred_cam_t
        if cam_t is None:
            raise ValueError(f"Missing pred_cam_t_full or full_img_cam_t for hand index {hand_index}")
        focal = scaled_focal_length or ham_out.get("focal_length") or ham_out.get("focal")
        if focal is None:
            raise ValueError(f"Missing focal length for hand index {hand_index}")
        center = ham_out.get("center")
        camera_translation_source = "pred_cam_t_full" if pred_cam_t_full is not None else ("pred_cam_t" if pred_cam_t is not None else None)

        # Determine center based on source
        if camera_translation_source == "pred_cam_t_full":
            if img_size is not None and len(img_size) >= 2:
                center = [float(img_size[0]) / 2.0, float(img_size[1]) / 2.0]
            else:
                center = [float(original_width) / 2.0, float(original_height) / 2.0]
        else:
            if center is None and img_size is not None and len(img_size) >= 2:
                center = [float(img_size[0]) / 2.0, float(img_size[1]) / 2.0]
            if center is None:
                center = [float(original_width) / 2.0, float(original_height) / 2.0]

        original_projection, depth_values = project_vertices(
            vertices,
            cam_t,
            focal,
            center,
            img_size or [original_width, original_height],
            (width, height),
        )

        projected_bbox = bbox_from_points(original_projection)
        left_mirror_applied = False
        projected_for_render = original_projection
        if hand_label.lower() == "left" and not has_native_left_mesh(entry):
            projected_for_render = bbox_2d_mirror(original_projection, bbox_xyxy)
            left_mirror_applied = True
            left_mirror_bbox = bbox_from_points(projected_for_render)
        else:
            left_mirror_bbox = None

        depth_alignment_scale = 1.0
        depth_alignment_count = 0
        depth_alignment_stats = None
        depth_values_for_render = depth_values
        if args.align_depth_to_reference:
            if reference_depth is None:
                raise ValueError("--align_depth_to_reference requires --reference_depth")
            depth_alignment_scale, depth_alignment_count, depth_alignment_stats = compute_depth_alignment_scale(
                projected_for_render,
                depth_values,
                faces,
                reference_depth,
            )
            depth_values_for_render = depth_values * depth_alignment_scale

        rasterize_mesh(projected_for_render, depth_values_for_render, faces, zbuffer, hand_buffer, hand_index)

        record = {
            "hand_index": hand_index,
            "hand_side_for_hamer": hand_label,
            "raw_handedness": raw_handedness,
            "camera_translation_source": camera_translation_source,
            "left_bbox_2d_mirror_applied": left_mirror_applied,
            "projected_bbox": projected_bbox,
            "left_bbox_2d_mirror_bbox": left_mirror_bbox,
            "depth_alignment_scale_to_reference": depth_alignment_scale,
            "depth_alignment_valid_pixels": depth_alignment_count,
            "depth_alignment_ratio_stats": depth_alignment_stats,
        }
        hand_records.append(record)

        label = f"hand_{hand_index}_{hand_label}"
        debug_color = (0, 255, 0) if hand_label.lower() == "right" else (255, 0, 255)
        depth_debug_image = draw_projection(depth_debug_image, projected_for_render, faces, bbox_xyxy, label, debug_color)

    hand_depth = np.full((height, width), np.nan, dtype=np.float32)
    valid_mask = hand_buffer >= 0
    hand_depth[valid_mask] = zbuffer[valid_mask]
    global_valid_count = int(np.sum(valid_mask))

    valid_counts = {}
    for hand_index in range(len(outputs)):
        valid_counts[hand_index] = int(np.sum(hand_buffer == hand_index))

    hand_depth_vis = normalize_depth_vis(hand_depth)
    hand_mask = (valid_mask.astype(np.uint8) * 255)

    hand_depth_path = out_dir / "hand_depth.npy"
    hand_depth_vis_path = out_dir / "hand_depth_vis.png"
    hand_mask_path = out_dir / "hand_mask.png"
    hand_depth_debug_path = out_dir / "hand_depth_debug.png"
    report_path = out_dir / "hand_depth_report.md"

    np.save(hand_depth_path, hand_depth)
    cv2.imwrite(str(hand_depth_vis_path), hand_depth_vis)
    cv2.imwrite(str(hand_mask_path), hand_mask)
    cv2.imwrite(str(hand_depth_debug_path), depth_debug_image)

    report_lines = ["# Hand Depth Render Report", "", f"Image: {image_path}", f"HaMeR metadata: {hamerdir / 'hamer_metadata.json'}", f"BBox JSON: {bbox_json_path}", "", f"Global valid depth pixels: {global_valid_count}", ""]
    for record in hand_records:
        report_lines.append(f"## Hand {record['hand_index']} ({record['hand_side_for_hamer']})")
        report_lines.append(f"raw_handedness: {record['raw_handedness']}")
        report_lines.append(f"camera_translation_source: {record['camera_translation_source']}")
        report_lines.append(f"left_bbox_2d_mirror_applied: {record['left_bbox_2d_mirror_applied']}")
        report_lines.append(f"projected_bbox: {record['projected_bbox']}")
        if record['left_bbox_2d_mirror_applied']:
            report_lines.append(f"left_bbox_2d_mirror_bbox: {record['left_bbox_2d_mirror_bbox']}")
        report_lines.append(f"valid depth pixels: {valid_counts[record['hand_index']]}" )
        report_lines.append(f"depth_alignment_scale_to_reference: {record['depth_alignment_scale_to_reference']}")
        report_lines.append(f"depth_alignment_valid_pixels: {record['depth_alignment_valid_pixels']}")
        report_lines.append(f"depth_alignment_ratio_stats: {record['depth_alignment_ratio_stats']}")
        report_lines.append("")

    report_lines.append(f"hand_depth: {hand_depth_path}")
    report_lines.append(f"hand_depth_vis: {hand_depth_vis_path}")
    report_lines.append(f"hand_mask: {hand_mask_path}")
    report_lines.append(f"hand_depth_debug: {hand_depth_debug_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved hand depth to {hand_depth_path}")
    print(f"Saved hand depth visualization to {hand_depth_vis_path}")
    print(f"Saved hand mask to {hand_mask_path}")
    print(f"Saved hand depth debug image to {hand_depth_debug_path}")
    print(f"Saved hand depth report to {report_path}")


if __name__ == "__main__":
    main()
