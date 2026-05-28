#!/usr/bin/env python3
"""Validate generated EgoWorld Stage 1 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EgoWorld Stage 1 outputs")
    parser.add_argument("--outputs", default="outputs", help="Stage 1 outputs directory")
    parser.add_argument("--allow-no-hands", action="store_true", help="Pass bbox/crop/HaMeR checks when no hands were detected")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mark(ok: bool, label: str, detail: str | None = None) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_array(path: Path, label: str, ndim: int | None = None, min_first_dim: int = 1) -> bool:
    if not mark(path.exists(), f"{label} exists", str(path)):
        return False
    try:
        arr = np.load(path)
    except Exception as exc:
        return mark(False, f"{label} loads with numpy", repr(exc))
    checks = [
        mark(isinstance(arr, np.ndarray), f"{label} is ndarray", str(type(arr))),
        mark(arr.size > 0, f"{label} is non-empty", f"shape={arr.shape}"),
        mark(np.issubdtype(arr.dtype, np.number), f"{label} is numeric", str(arr.dtype)),
        mark(np.all(np.isfinite(arr)), f"{label} is finite", f"shape={arr.shape}"),
    ]
    if ndim is not None:
        checks.append(mark(arr.ndim == ndim, f"{label} has {ndim} dimensions", f"shape={arr.shape}"))
    if arr.ndim > 0:
        checks.append(mark(arr.shape[0] >= min_first_dim, f"{label} first dimension is non-empty", f"shape={arr.shape}"))
    return all(checks)


def ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("element vertex "):
                    return int(line.strip().split()[-1])
                if line.strip() == "end_header":
                    break
    except Exception:
        return None
    return None


def validate_bbox_json(path: Path) -> tuple[bool, int]:
    if not mark(path.exists(), "hand_bboxes.json exists", str(path)):
        return False, 0
    try:
        data = load_json(path)
    except Exception as exc:
        mark(False, "hand_bboxes.json is valid JSON", repr(exc))
        return False, 0
    hands = data.get("hands")
    ok = mark(isinstance(hands, list), "hand_bboxes.json contains hands list")
    if not ok:
        return False, 0
    image_width = data.get("image_width")
    image_height = data.get("image_height")
    for idx, hand in enumerate(hands):
        coords_ok = all(k in hand for k in ("x1", "y1", "x2", "y2"))
        if coords_ok:
            x1, y1, x2, y2 = (hand[k] for k in ("x1", "y1", "x2", "y2"))
            coords_ok = isinstance(x1, int) and isinstance(y1, int) and isinstance(x2, int) and isinstance(y2, int) and x2 > x1 and y2 > y1
            if isinstance(image_width, int) and isinstance(image_height, int):
                coords_ok = coords_ok and 0 <= x1 < image_width and 0 <= x2 < image_width and 0 <= y1 < image_height and 0 <= y2 < image_height
        ok = mark(coords_ok, f"hand bbox {idx} is valid", str(hand)) and ok
    return ok, len(hands)


def validate_hamer_outputs(hamer_dir: Path, expected_hands: int, allow_no_hands: bool) -> bool:
    if expected_hands == 0 and allow_no_hands:
        return mark(True, "HaMeR outputs skipped because no hands were detected")

    ok = mark(hamer_dir.exists(), "outputs/hamer exists", str(hamer_dir))
    metadata_path = hamer_dir / "hamer_metadata.json"
    ok = mark(metadata_path.exists(), "hamer_metadata.json exists", str(metadata_path)) and ok
    if not metadata_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
    except Exception as exc:
        mark(False, "hamer_metadata.json is valid JSON", repr(exc))
        return False

    outputs = metadata.get("outputs")
    ok = mark(isinstance(outputs, list) and len(outputs) > 0, "hamer_metadata.json contains hand outputs") and ok
    if not isinstance(outputs, list):
        return False
    if expected_hands > 0:
        ok = mark(len(outputs) == expected_hands, "HaMeR output count matches detected hands", f"hamer={len(outputs)} bboxes={expected_hands}") and ok

    for idx, entry in enumerate(outputs):
        for key, ndim in (("vertices", 2), ("joints", 2), ("faces", 2)):
            value = entry.get(key)
            if value is None:
                ok = mark(False, f"hand {idx} metadata has {key} path") and ok
                continue
            ok = check_array(resolve(value), f"hand {idx} {key}", ndim=ndim) and ok
        obj_value = entry.get("obj")
        if obj_value is not None:
            obj_path = resolve(obj_value)
            ok = mark(obj_path.exists() and obj_path.stat().st_size > 0, f"hand {idx} OBJ exists and is non-empty", str(obj_path)) and ok
        projection = entry.get("pred_cam_t_full")
        if projection is not None:
            arr = np.asarray(projection, dtype=float)
            ok = mark(arr.size >= 3 and np.all(np.isfinite(arr)), f"hand {idx} pred_cam_t_full is finite", str(projection)) and ok
    return ok


def main() -> int:
    args = parse_args()
    out = resolve(args.outputs)
    failures = 0
    print("EgoWorld Stage 1 output verification")
    print(f"Outputs directory: {out}")

    failures += not check_array(out / "depth_raw.npy", "depth_raw.npy", ndim=2)
    try:
        depth = np.load(out / "depth_raw.npy")
        failures += not mark(np.any(np.isfinite(depth) & (depth > 0)), "depth has positive finite values")
    except Exception:
        failures += 1

    ply_path = out / "exo_point_cloud.ply"
    failures += not mark(ply_path.exists() and ply_path.stat().st_size > 0, "exo_point_cloud.ply exists and is non-empty", str(ply_path))
    count = ply_vertex_count(ply_path)
    failures += not mark(count is not None and count > 0, "exo_point_cloud.ply declares actual vertices", str(count))

    bbox_ok, hand_count = validate_bbox_json(out / "hand_bboxes.json")
    failures += not bbox_ok
    if hand_count == 0:
        print("[WARN] No hands detected in hand_bboxes.json.")
        if not args.allow_no_hands:
            failures += not mark(False, "hands detected for HaMeR stage", "use --allow-no-hands to accept this case")

    crop_dir = out / "hand_crops"
    failures += not mark(crop_dir.exists() and crop_dir.is_dir(), "hand_crops directory exists", str(crop_dir))
    crop_files = sorted(crop_dir.glob("hand_*.png")) if crop_dir.exists() else []
    if hand_count > 0:
        failures += not mark(len(crop_files) == hand_count, "hand crop count matches detected hands", f"crops={len(crop_files)} hands={hand_count}")
    elif args.allow_no_hands:
        mark(True, "hand crop count accepted with no detected hands", f"crops={len(crop_files)}")

    failures += not validate_hamer_outputs(out / "hamer", hand_count, args.allow_no_hands)

    projection_report = out / "hamer_projection" / "final_projection_report.md"
    projection_debug = out / "hamer_projection" / "final_projection_debug.png"
    if projection_report.exists() or projection_debug.exists():
        failures += not mark(projection_report.exists() and projection_report.stat().st_size > 0, "projection report exists and is non-empty", str(projection_report))
        failures += not mark(projection_debug.exists() and projection_debug.stat().st_size > 0, "projection debug image exists and is non-empty", str(projection_debug))
    else:
        print("[WARN] Projection diagnostics not found; run scripts/verify_hamer_projection.py after HaMeR inference.")

    print(f"Output verification result: {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
