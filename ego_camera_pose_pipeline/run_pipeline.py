#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from ego_camera_pose_pipeline.geometry import (
    backproject_pixel,
    backproject_pointcloud,
    blend_directions,
    camera_extrinsics,
    gaze3d_to_pointcloud,
    sample_depth_median,
    stabilize_direction,
    synthetic_intrinsics,
    weighted_target_direction,
)
from src.depth_estimator import DepthEstimator


IRIS_GROUPS = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))
EYE_CONTOUR_GROUPS = ((33, 133, 159, 145), (362, 263, 386, 374))
PALM_LANDMARKS = (0, 5, 9, 13, 17)
FINGERTIP_LANDMARKS = (4, 8, 12, 16, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate per-frame eye-centered camera extrinsics.")
    parser.add_argument("--video", type=Path, default=Path("inputs/exo.mp4"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ego_camera_pose_exo"))
    parser.add_argument("--gaze_csv", type=Path, help="Reuse an existing Gaze3D prediction CSV.")
    parser.add_argument("--gaze3d_dir", type=Path, default=Path("external/gaze3d"))
    parser.add_argument("--gaze_env", default="gazeCVPR")
    parser.add_argument("--gaze_device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gaze_pid", type=int, help="Track ID to use. Defaults to the most frequent PID.")
    parser.add_argument("--depth_model", choices=("moge", "dummy"), default="moge")
    parser.add_argument(
        "--horizontal_fov",
        type=float,
        default=60.0,
        help="Exo-camera horizontal FOV used for depth backprojection.",
    )
    parser.add_argument(
        "--ego_horizontal_fov",
        type=float,
        default=80.0,
        help="Virtual ego-camera horizontal FOV used for output projection.",
    )
    parser.add_argument("--point_stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--face_detection_confidence", type=float, default=0.3)
    parser.add_argument(
        "--task-aware",
        "--task_aware",
        dest="task_aware",
        action="store_true",
        help="Blend gaze with backprojected hand/workspace targets for camera orientation.",
    )
    parser.add_argument("--hand_detection_confidence", type=float, default=0.3)
    parser.add_argument("--hand_tracking_confidence", type=float, default=0.3)
    parser.add_argument("--task_gaze_weight", type=float, default=0.3)
    parser.add_argument("--task_direction_smoothing", type=float, default=0.8)
    parser.add_argument("--task_deadzone_degrees", type=float, default=2.0)
    parser.add_argument("--task_max_step_degrees", type=float, default=3.0)
    parser.add_argument("--sample_radii", type=int, nargs="+", default=[2, 4, 8, 12])
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_frames(video_path: Path, frames_dir: Path, max_frames: int | None) -> tuple[list[Path], dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames is not None and index >= max_frames):
            break
        path = frames_dir / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to save frame: {path}")
        frames.append(path)
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError("No frames were extracted")
    return frames, {"fps": fps, "width": width, "height": height, "frame_count": len(frames)}


def run_gaze3d(args: argparse.Namespace, video_path: Path, gaze_dir: Path) -> Path:
    gaze_dir.mkdir(parents=True, exist_ok=True)
    destination = gaze_dir / "predicted_gaze.csv"
    if args.gaze_csv:
        source = resolve(args.gaze_csv)
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        return destination

    gaze3d_dir = resolve(args.gaze3d_dir)
    command = [
        "/home/youngchan/miniconda3/bin/conda", "run", "-n", args.gaze_env,
        "python", "demo.py",
        "--input-filename", str(video_path),
        "--output-dir", str(gaze_dir),
        "--modality", "video",
        "--device", args.gaze_device,
    ]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=gaze3d_dir, check=True)
    generated = gaze_dir / f"{video_path.stem}_video_predicted_gaze.csv"
    if not generated.exists():
        raise FileNotFoundError(f"Gaze3D did not create {generated}")
    shutil.copy2(generated, destination)
    return destination


def gaze_timeline(csv_path: Path, frame_count: int, requested_pid: int | None) -> tuple[pd.DataFrame, int]:
    gaze = pd.read_csv(csv_path)
    required = {"frame_id", "pid", "xmin", "ymin", "xmax", "ymax", "gaze_x", "gaze_y", "gaze_z"}
    if not required.issubset(gaze.columns):
        raise ValueError(f"Gaze CSV missing columns: {sorted(required - set(gaze.columns))}")
    pid = requested_pid
    if pid is None:
        pid = int(gaze.groupby("pid").size().sort_values(ascending=False).index[0])
    selected = gaze[np.isclose(gaze["pid"], pid)].copy()
    selected["frame_index"] = selected["frame_id"].round().astype(int) - 1
    selected = selected.drop_duplicates("frame_index").set_index("frame_index")
    timeline = selected.reindex(range(frame_count))
    value_columns = ["xmin", "ymin", "xmax", "ymax", "gaze_x", "gaze_y", "gaze_z"]
    timeline["gaze_observed"] = timeline["gaze_x"].notna()
    timeline[value_columns] = timeline[value_columns].interpolate(limit_direction="both")
    vectors = timeline[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    timeline[["gaze_x", "gaze_y", "gaze_z"]] = vectors / np.maximum(norms, 1e-12)
    timeline["pid"] = pid
    return timeline.reset_index(names="frame_index"), pid


class FaceEyeDetector:
    def __init__(self, min_confidence: float) -> None:
        self.detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def close(self) -> None:
        self.detector.close()

    def detect(self, bgr: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        height, width = bgr.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        box_width, box_height = xmax - xmin, ymax - ymin
        x0 = max(0, int(np.floor(xmin - 0.4 * box_width)))
        y0 = max(0, int(np.floor(ymin - 0.25 * box_height)))
        x1 = min(width, int(np.ceil(xmax + 0.4 * box_width)))
        y1 = min(height, int(np.ceil(ymax + 0.25 * box_height)))
        attempts = [(bgr[y0:y1, x0:x1], x0, y0), (bgr, 0, 0)]
        for image, offset_x, offset_y in attempts:
            result = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                continue
            face = result.multi_face_landmarks[0]
            crop_height, crop_width = image.shape[:2]
            landmarks = np.array(
                [
                    [
                        point.x * (crop_width - 1) + offset_x,
                        point.y * (crop_height - 1) + offset_y,
                    ]
                    for point in face.landmark
                ],
                dtype=np.float32,
            )
            groups = IRIS_GROUPS if len(landmarks) >= 478 else EYE_CONTOUR_GROUPS
            eyes = np.stack([landmarks[list(group)].mean(axis=0) for group in groups])
            if eyes[0, 0] > eyes[1, 0]:
                eyes = eyes[::-1]
            return eyes, landmarks
        return None, None


def detect_eye_timeline(
    frames: list[Path],
    gaze: pd.DataFrame,
    face_dir: Path,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    face_dir.mkdir(parents=True, exist_ok=True)
    detector = FaceEyeDetector(confidence)
    raw_eyes = np.full((len(frames), 2, 2), np.nan, dtype=np.float64)
    observed = np.zeros(len(frames), dtype=bool)
    try:
        for index, frame_path in enumerate(frames):
            image = cv2.imread(str(frame_path))
            bbox = gaze.loc[index, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
            eyes, landmarks = detector.detect(image, bbox)
            if eyes is None:
                continue
            raw_eyes[index] = eyes
            observed[index] = True
            np.savez_compressed(
                face_dir / f"frame_{index:06d}.npz",
                eye_pixels=eyes.astype(np.float32),
                face_landmarks_pixels=landmarks.astype(np.float32),
                gaze_bbox=bbox.astype(np.float32),
            )
    finally:
        detector.close()
    if not observed.any():
        raise RuntimeError("MediaPipe Face Mesh did not detect a face in any frame")
    flat = raw_eyes.reshape(len(frames), 4)
    interpolated = pd.DataFrame(flat).interpolate(limit_direction="both").to_numpy().reshape(-1, 2, 2)
    return interpolated, observed


class HandTaskDetector:
    def __init__(self, detection_confidence: float, tracking_confidence: float) -> None:
        self.detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self) -> None:
        self.detector.close()

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
        result = self.detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return np.empty((0, 21, 2), dtype=np.float32), [], np.empty(0, dtype=np.float32)
        height, width = bgr.shape[:2]
        landmarks = []
        labels = []
        scores = []
        for index, hand in enumerate(result.multi_hand_landmarks):
            landmarks.append(
                np.array(
                    [[point.x * (width - 1), point.y * (height - 1)] for point in hand.landmark],
                    dtype=np.float32,
                )
            )
            classification = result.multi_handedness[index].classification[0]
            labels.append(classification.label)
            scores.append(float(classification.score))
        order = np.argsort([points[list(PALM_LANDMARKS), 0].mean() for points in landmarks])
        return (
            np.stack([landmarks[index] for index in order]),
            [labels[index] for index in order],
            np.asarray([scores[index] for index in order], dtype=np.float32),
        )


def detect_hand_timeline(
    frames: list[Path],
    hand_dir: Path,
    detection_confidence: float,
    tracking_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    hand_dir.mkdir(parents=True, exist_ok=True)
    timeline = np.full((len(frames), 2, 21, 2), np.nan, dtype=np.float32)
    counts = np.zeros(len(frames), dtype=np.int32)
    detector = HandTaskDetector(detection_confidence, tracking_confidence)
    try:
        for index, frame_path in enumerate(frames):
            image = cv2.imread(str(frame_path))
            landmarks, labels, scores = detector.detect(image)
            count = min(len(landmarks), 2)
            if count == 0:
                continue
            timeline[index, :count] = landmarks[:count]
            counts[index] = count
            np.savez_compressed(
                hand_dir / f"frame_{index:06d}.npz",
                hand_landmarks_pixels=landmarks[:count],
                handedness=np.asarray(labels[:count]),
                handedness_score=scores[:count],
            )
    finally:
        detector.close()
    np.savez_compressed(
        hand_dir / "hand_timeline.npz",
        hand_landmarks_pixels=timeline,
        hand_count=counts,
    )
    return timeline, counts


def task_target_from_hands(
    hand_landmarks: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_center: np.ndarray,
    sample_radii: tuple[int, ...],
) -> dict[str, np.ndarray] | None:
    palm_points = []
    fingertip_points = []
    fingertip_pixels = []
    for hand in hand_landmarks:
        palm_pixel = hand[list(PALM_LANDMARKS)].mean(axis=0)
        palm_depth, _ = sample_depth_median(depth, palm_pixel, sample_radii)
        if np.isfinite(palm_depth):
            palm_points.append(backproject_pixel(palm_pixel, palm_depth, intrinsics))
        for fingertip_index in FINGERTIP_LANDMARKS:
            pixel = hand[fingertip_index]
            value, _ = sample_depth_median(depth, pixel, sample_radii)
            if np.isfinite(value):
                fingertip_points.append(backproject_pixel(pixel, value, intrinsics))
                fingertip_pixels.append(pixel)
    if not palm_points and not fingertip_points:
        return None

    targets = []
    weights = []
    if palm_points:
        targets.extend(palm_points)
        weights.extend([0.5 / len(palm_points)] * len(palm_points))
    if fingertip_points:
        targets.extend(fingertip_points)
        weights.extend([0.5 / len(fingertip_points)] * len(fingertip_points))
    if not palm_points or not fingertip_points:
        weights = [1.0 / len(targets)] * len(targets)
    direction = weighted_target_direction(camera_center, np.asarray(targets), np.asarray(weights))
    return {
        "direction": direction,
        "palm_points": np.asarray(palm_points, dtype=np.float64).reshape(-1, 3),
        "fingertip_points": np.asarray(fingertip_points, dtype=np.float64).reshape(-1, 3),
        "workspace_point": np.mean(fingertip_points or palm_points, axis=0),
        "workspace_pixel": np.mean(fingertip_pixels, axis=0)
        if fingertip_pixels
        else hand_landmarks[:, list(PALM_LANDMARKS)].mean(axis=(0, 1)),
    }


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    lo, hi = np.percentile(valid, [2, 98])
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def draw_overlay(
    image: np.ndarray,
    eyes: np.ndarray,
    gaze_raw: np.ndarray,
    observed_eye: bool,
    observed_gaze: bool,
    hand_landmarks: np.ndarray | None = None,
    workspace_pixel: np.ndarray | None = None,
) -> np.ndarray:
    output = image.copy()
    color = (0, 255, 0) if observed_eye else (0, 180, 255)
    for point in eyes:
        cv2.circle(output, tuple(np.round(point).astype(int)), 5, color, -1, cv2.LINE_AA)
    midpoint = eyes.mean(axis=0)
    screen_direction = np.array([-gaze_raw[0], -gaze_raw[1]])
    screen_direction /= max(np.linalg.norm(screen_direction), 1e-8)
    end = midpoint + 100.0 * screen_direction
    gaze_color = (255, 80, 30) if observed_gaze else (255, 180, 30)
    cv2.arrowedLine(
        output, tuple(np.round(midpoint).astype(int)), tuple(np.round(end).astype(int)),
        gaze_color, 3, cv2.LINE_AA, tipLength=0.2,
    )
    if hand_landmarks is not None:
        for hand in hand_landmarks:
            integer_points = np.round(hand).astype(int)
            for start, end_index in mp.solutions.hands.HAND_CONNECTIONS:
                cv2.line(
                    output,
                    tuple(integer_points[start]),
                    tuple(integer_points[end_index]),
                    (50, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            for landmark_index in PALM_LANDMARKS + FINGERTIP_LANDMARKS:
                cv2.circle(output, tuple(integer_points[landmark_index]), 3, (0, 255, 255), -1)
    if workspace_pixel is not None:
        workspace = tuple(np.round(workspace_pixel).astype(int))
        cv2.circle(output, workspace, 9, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(output, tuple(np.round(midpoint).astype(int)), workspace, (255, 0, 255), 2)
    return output


def save_preview(overlays: list[Path], output: Path, fps: float, size: tuple[int, int]) -> None:
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create preview video: {output}")
    for path in overlays:
        writer.write(cv2.imread(str(path)))
    writer.release()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.task_gaze_weight <= 1.0:
        raise ValueError("--task_gaze_weight must be in [0, 1]")
    if not 0.0 <= args.task_direction_smoothing < 1.0:
        raise ValueError("--task_direction_smoothing must be in [0, 1)")
    video_path, out_dir = resolve(args.video), resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, video_meta = extract_frames(video_path, out_dir / "frames", args.max_frames)
    intrinsics = synthetic_intrinsics(video_meta["width"], video_meta["height"], args.horizontal_fov)
    ego_intrinsics = synthetic_intrinsics(
        video_meta["width"], video_meta["height"], args.ego_horizontal_fov
    )
    np.save(out_dir / "intrinsics.npy", intrinsics)
    (out_dir / "intrinsics.json").write_text(json.dumps({
        "model": "synthetic_pinhole",
        "horizontal_fov_degrees": args.horizontal_fov,
        "width": video_meta["width"], "height": video_meta["height"],
        "fx": intrinsics[0, 0], "fy": intrinsics[1, 1],
        "cx": intrinsics[0, 2], "cy": intrinsics[1, 2],
    }, indent=2), encoding="utf-8")
    np.save(out_dir / "ego_intrinsics.npy", ego_intrinsics)
    (out_dir / "ego_intrinsics.json").write_text(json.dumps({
        "model": "synthetic_pinhole",
        "horizontal_fov_degrees": args.ego_horizontal_fov,
        "width": video_meta["width"], "height": video_meta["height"],
        "fx": ego_intrinsics[0, 0], "fy": ego_intrinsics[1, 1],
        "cx": ego_intrinsics[0, 2], "cy": ego_intrinsics[1, 2],
    }, indent=2), encoding="utf-8")

    gaze_csv = run_gaze3d(args, video_path, out_dir / "gaze3d")
    gaze, gaze_pid = gaze_timeline(gaze_csv, len(frames), args.gaze_pid)
    eyes, eye_observed = detect_eye_timeline(
        frames, gaze, out_dir / "face_landmarks", args.face_detection_confidence
    )
    np.savez_compressed(
        out_dir / "face_landmarks" / "eye_timeline.npz",
        eye_pixels=eyes.astype(np.float32),
        mediapipe_observed=eye_observed,
    )
    if args.task_aware:
        hands, hand_counts = detect_hand_timeline(
            frames,
            out_dir / "hand_landmarks",
            args.hand_detection_confidence,
            args.hand_tracking_confidence,
        )
    else:
        hands = np.full((len(frames), 2, 21, 2), np.nan, dtype=np.float32)
        hand_counts = np.zeros(len(frames), dtype=np.int32)

    directories = {name: out_dir / name for name in ("depth", "depth_vis", "pointcloud", "camera", "overlay")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    estimator = DepthEstimator(mode=args.depth_model)
    rows = []
    overlays = []
    camera_centers = []
    gaze_vectors = []
    task_vectors = []
    orientation_vectors = []
    rotations = []
    camera_to_exo = []
    exo_to_camera = []
    previous_orientation = None
    task_target_frames = 0

    for index, frame_path in enumerate(frames):
        print(f"[{index + 1}/{len(frames)}] depth, point cloud, eyes, extrinsics", flush=True)
        bgr = cv2.imread(str(frame_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth, model_intrinsics = estimator.predict(str(frame_path))
        points, colors, pixels = backproject_pointcloud(rgb, depth, intrinsics, args.point_stride)
        np.savez_compressed(
            directories["depth"] / f"frame_{index:06d}.npz",
            depth=depth.astype(np.float32), assumed_intrinsics=intrinsics.astype(np.float32),
            model_intrinsics=np.asarray(model_intrinsics, dtype=np.float32),
        )
        cv2.imwrite(str(directories["depth_vis"] / f"frame_{index:06d}.png"), depth_visualization(depth))
        np.savez_compressed(
            directories["pointcloud"] / f"frame_{index:06d}.npz",
            points_xyz=points.astype(np.float32), colors_rgb=colors, pixels_uv=pixels,
            stride=np.int32(args.point_stride),
        )

        eye_depths, eye_radii, eye_points = [], [], []
        for pixel in eyes[index]:
            value, radius = sample_depth_median(depth, pixel, tuple(args.sample_radii))
            eye_depths.append(value)
            eye_radii.append(radius)
            eye_points.append(backproject_pixel(pixel, value, intrinsics))
        eye_points = np.asarray(eye_points)
        camera_center = eye_points.mean(axis=0)
        gaze_raw = gaze.loc[index, ["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
        gaze_pointcloud = gaze3d_to_pointcloud(gaze_raw)
        frame_hands = hands[index, : hand_counts[index]]
        task_target = None
        task_direction = None
        orientation_direction = gaze_pointcloud
        if args.task_aware and len(frame_hands):
            task_target = task_target_from_hands(
                frame_hands,
                depth,
                intrinsics,
                camera_center,
                tuple(args.sample_radii),
            )
            if task_target is not None:
                task_direction = task_target["direction"]
                orientation_direction = blend_directions(
                    gaze_pointcloud, task_direction, args.task_gaze_weight
                )
                task_target_frames += 1
        if args.task_aware and previous_orientation is not None:
            orientation_direction = stabilize_direction(
                previous_orientation,
                orientation_direction,
                smoothing=args.task_direction_smoothing,
                deadzone_degrees=args.task_deadzone_degrees,
                max_step_degrees=args.task_max_step_degrees,
            )
        previous_orientation = orientation_direction
        rotation, t_camera_to_exo, t_exo_to_camera = camera_extrinsics(
            camera_center, orientation_direction
        )

        stem = f"frame_{index:06d}"
        np.save(directories["camera"] / f"{stem}_camera_to_exo.npy", t_camera_to_exo)
        np.save(directories["camera"] / f"{stem}_exo_to_camera.npy", t_exo_to_camera)
        record = {
            "frame_index": index, "source_frame_id": index + 1, "timestamp_seconds": index / video_meta["fps"],
            "gaze_pid": gaze_pid,
            "eye_pixels": eyes[index].tolist(), "eye_depths": eye_depths,
            "eye_depth_sample_radii": eye_radii, "eye_points_exo": eye_points.tolist(),
            "ego_cam_point_exo": camera_center.tolist(), "gaze3d_raw": gaze_raw.tolist(),
            "gaze_direction_exo": gaze_pointcloud.tolist(), "camera_y_roll_hint_exo": [0.0, 1.0, 0.0],
            "orientation_mode": "task-aware" if args.task_aware else "gaze-aware",
            "task_direction_exo": task_direction.tolist() if task_direction is not None else None,
            "camera_direction_exo": orientation_direction.tolist(),
            "hand_count": int(hand_counts[index]),
            "hand_landmarks_pixels": frame_hands.tolist(),
            "hand_palm_points_exo": task_target["palm_points"].tolist()
            if task_target is not None
            else [],
            "hand_fingertip_points_exo": task_target["fingertip_points"].tolist()
            if task_target is not None
            else [],
            "workspace_point_exo": task_target["workspace_point"].tolist()
            if task_target is not None
            else None,
            "rotation_camera_to_exo": rotation.tolist(),
            "transform_camera_to_exo": t_camera_to_exo.tolist(),
            "transform_exo_to_camera": t_exo_to_camera.tolist(),
            "mediapipe_observed": bool(eye_observed[index]),
            "gaze_observed": bool(gaze.loc[index, "gaze_observed"]),
        }
        (directories["camera"] / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        overlay_path = directories["overlay"] / f"{stem}.png"
        cv2.imwrite(
            str(overlay_path),
            draw_overlay(
                bgr,
                eyes[index],
                gaze_raw,
                eye_observed[index],
                bool(gaze.loc[index, "gaze_observed"]),
                hand_landmarks=frame_hands if len(frame_hands) else None,
                workspace_pixel=task_target["workspace_pixel"]
                if task_target is not None
                else None,
            ),
        )
        overlays.append(overlay_path)

        flat = {
            "frame_index": index, "source_frame_id": index + 1, "timestamp_seconds": index / video_meta["fps"],
            "gaze_pid": gaze_pid, "mediapipe_observed": bool(eye_observed[index]),
            "gaze_observed": bool(gaze.loc[index, "gaze_observed"]),
            "orientation_mode": "task-aware" if args.task_aware else "gaze-aware",
            "hand_count": int(hand_counts[index]),
            "task_target_observed": task_target is not None,
        }
        for name, values in {
            "left_eye_uv": eyes[index, 0], "right_eye_uv": eyes[index, 1],
            "ego_cam": camera_center, "gaze_raw": gaze_raw, "gaze_exo": gaze_pointcloud,
            "task_exo": task_direction
            if task_direction is not None
            else np.full(3, np.nan),
            "camera_direction_exo": orientation_direction,
        }.items():
            for axis, value in zip(("x", "y", "z")[:len(values)], values):
                flat[f"{name}_{axis}"] = float(value)
        for row_index in range(4):
            for column_index in range(4):
                flat[f"T_exo_to_camera_{row_index}{column_index}"] = float(t_exo_to_camera[row_index, column_index])
        rows.append(flat)
        camera_centers.append(camera_center)
        gaze_vectors.append(gaze_pointcloud)
        task_vectors.append(task_direction if task_direction is not None else np.full(3, np.nan))
        orientation_vectors.append(orientation_direction)
        rotations.append(rotation)
        camera_to_exo.append(t_camera_to_exo)
        exo_to_camera.append(t_exo_to_camera)

    pd.DataFrame(rows).to_csv(out_dir / "camera_trajectory.csv", index=False)
    np.savez_compressed(
        out_dir / "camera_trajectory.npz",
        frame_index=np.arange(len(frames)),
        timestamp_seconds=np.arange(len(frames)) / video_meta["fps"],
        ego_cam_point_exo=np.asarray(camera_centers),
        gaze_direction_exo=np.asarray(gaze_vectors),
        task_direction_exo=np.asarray(task_vectors),
        camera_direction_exo=np.asarray(orientation_vectors),
        rotation_camera_to_exo=np.asarray(rotations),
        transform_camera_to_exo=np.asarray(camera_to_exo),
        transform_exo_to_camera=np.asarray(exo_to_camera),
        intrinsics=intrinsics,
        ego_intrinsics=ego_intrinsics,
        mediapipe_observed=eye_observed,
        gaze_observed=gaze["gaze_observed"].to_numpy(dtype=bool),
        hand_count=hand_counts,
        task_target_observed=np.isfinite(np.asarray(task_vectors)).all(axis=1),
    )
    save_preview(overlays, out_dir / "eye_gaze_overlay.mp4", video_meta["fps"], (video_meta["width"], video_meta["height"]))
    summary = {
        "input_video": str(video_path), "output_directory": str(out_dir), **video_meta,
        "depth_model": args.depth_model, "point_cloud_stride": args.point_stride,
        "exo_horizontal_fov_degrees": args.horizontal_fov,
        "ego_horizontal_fov_degrees": args.ego_horizontal_fov,
        "gaze_pid": gaze_pid, "mediapipe_observed_frames": int(eye_observed.sum()),
        "gaze_observed_frames": int(gaze["gaze_observed"].sum()),
        "orientation_mode": "task-aware" if args.task_aware else "gaze-aware",
        "hand_observed_frames": int(np.count_nonzero(hand_counts)),
        "two_hand_observed_frames": int(np.count_nonzero(hand_counts == 2)),
        "task_target_frames": task_target_frames,
        "task_aware_parameters": {
            "gaze_weight": args.task_gaze_weight,
            "task_weight": 1.0 - args.task_gaze_weight,
            "direction_smoothing": args.task_direction_smoothing,
            "deadzone_degrees": args.task_deadzone_degrees,
            "max_step_degrees_per_frame": args.task_max_step_degrees,
            "palm_total_weight": 0.5,
            "fingertip_workspace_total_weight": 0.5,
        },
        "coordinate_system": "exo OpenCV point-cloud: +X right, +Y down, +Z forward",
        "gaze_mapping": "Gaze3D (gx,gy,gz) -> exo (-gx,-gy,gz), then unit normalization",
        "depth_units": "MoGe model output units; monocular absolute scale is not externally calibrated",
        "extrinsics": {
            "camera_to_exo": "p_exo = T_camera_to_exo @ p_camera",
            "exo_to_camera": "p_camera = T_exo_to_camera @ p_exo",
        },
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
