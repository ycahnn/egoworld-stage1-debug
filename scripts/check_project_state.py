#!/usr/bin/env python3
"""Read-only workspace state check for EgoWorld Stage 1.

This script is intentionally lightweight and can run from egoworld-main. Use
scripts/check_stage1_env.py from egoworld-hamer for HaMeR import/model checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MARKDOWN = PROJECT_ROOT / "outputs" / "environment_report.md"

MAIN_IMPORTS = ["numpy", "cv2", "open3d", "mediapipe", "moge"]
REQUIRED_PATHS = [
    PROJECT_ROOT / "inputs" / "exo.jpg",
    PROJECT_ROOT / "external" / "hamer",
    PROJECT_ROOT / "external" / "hamer" / "hamer",
    PROJECT_ROOT / "external" / "hamer" / "demo.py",
    PROJECT_ROOT / "external" / "hamer" / "setup.py",
    PROJECT_ROOT / "external" / "hamer" / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
    PROJECT_ROOT / "external" / "hamer" / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
    PROJECT_ROOT / "external" / "hamer" / "_DATA" / "data" / "mano" / "MANO_LEFT.pkl",
    PROJECT_ROOT / "external" / "hamer" / "_DATA" / "data" / "mano_mean_params.npz",
]
OUTPUT_PATHS = [
    PROJECT_ROOT / "outputs" / "depth_raw.npy",
    PROJECT_ROOT / "outputs" / "exo_point_cloud.ply",
    PROJECT_ROOT / "outputs" / "hand_bboxes.json",
    PROJECT_ROOT / "outputs" / "hand_crops",
    PROJECT_ROOT / "outputs" / "hamer" / "hamer_metadata.json",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def status_line(ok: bool, label: str) -> str:
    return f"[{'PASS' if ok else 'FAIL'}] {label}"


def import_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def path_lines(paths: Iterable[Path]) -> list[str]:
    return [status_line(path.exists(), rel(path)) for path in paths]


def build_report() -> str:
    lines: list[str] = []
    lines.append("# EgoWorld Stage 1 Workspace Report")
    lines.append("")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Python executable: {sys.executable}")
    lines.append(f"Python version: {sys.version.split()[0]}")
    lines.append("")
    lines.append("## Required workspace paths")
    lines.extend(path_lines(REQUIRED_PATHS))
    lines.append("")
    lines.append("## Current output paths")
    lines.extend(path_lines(OUTPUT_PATHS))
    lines.append("")
    lines.append("## Main environment import availability")
    for package in MAIN_IMPORTS:
        lines.append(status_line(import_available(package), package))
    lines.append("")
    lines.append("## Notes")
    lines.append("- This project uses conda envs: egoworld-main (Python 3.11) and egoworld-hamer (Python 3.10).")
    lines.append("- Do not create or use external/hamer/.hamer for the current Linux conda workflow.")
    lines.append("- Run scripts/check_stage1_env.py inside egoworld-hamer for HaMeR import, renderer compatibility, and model asset checks.")
    lines.append("- MANO_RIGHT.pkl and MANO_LEFT.pkl must be manually downloaded because of MANO license restrictions.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check EgoWorld Stage 1 workspace state")
    parser.add_argument("--write-report", action="store_true", help="Write outputs/environment_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    print(report)
    if args.write_report:
        OUTPUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_MARKDOWN.write_text(report, encoding="utf-8")
        print(f"\nEnvironment report written to: {OUTPUT_MARKDOWN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
