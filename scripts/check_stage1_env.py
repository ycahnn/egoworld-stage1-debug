#!/usr/bin/env python3
"""Preflight checks for EgoWorld Stage 1 environments and HaMeR assets."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAMER_ROOT = PROJECT_ROOT / "external" / "hamer"
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
REQUIRED_HAMER_FILES = [
    HAMER_ROOT / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
    HAMER_ROOT / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
    HAMER_ROOT / "_DATA" / "data" / "mano" / "MANO_LEFT.pkl",
    HAMER_ROOT / "_DATA" / "data" / "mano_mean_params.npz",
]
FORBIDDEN_IMPORTS = ["pyrender", "OpenGL", "detectron2"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def mark(ok: bool, label: str, detail: str | None = None) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def ensure_hamer_path_first() -> None:
    hamer_root_str = str(HAMER_ROOT)
    if hamer_root_str in sys.path:
        sys.path.remove(hamer_root_str)
    sys.path.insert(0, hamer_root_str)


def module_file(module: Any) -> str:
    return str(getattr(module, "__file__", "<no __file__>"))


def main() -> int:
    failures = 0
    print("EgoWorld Stage 1 preflight")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    version_ok = sys.version_info[:2] in {(3, 10), (3, 11)}
    failures += not mark(version_ok, "Python version is supported", "expected 3.10 for egoworld-hamer or 3.11 for egoworld-main")

    failures += not mark(HAMER_ROOT.exists(), "external/hamer exists", rel(HAMER_ROOT))
    failures += not mark((HAMER_ROOT / "hamer").exists(), "external/hamer/hamer package exists", rel(HAMER_ROOT / "hamer"))

    ensure_hamer_path_first()
    failures += not mark(sys.path[0] == str(HAMER_ROOT), "external/hamer is first on sys.path", sys.path[0])

    try:
        from src.hamer_stage1_compat import install_hamer_renderer_stub

        install_hamer_renderer_stub()
        failures += not mark(True, "Stage 1 renderer compatibility stub installed")
    except Exception as exc:
        failures += not mark(False, "Stage 1 renderer compatibility stub installed", repr(exc))

    try:
        hamer = importlib.import_module("hamer")
        hamer_file = module_file(hamer)
        expected = str(HAMER_ROOT / "hamer")
        failures += not mark(hamer_file.startswith(expected), "hamer imports from external/hamer", hamer_file)
    except Exception as exc:
        failures += not mark(False, "hamer imports", repr(exc))

    try:
        renderer = importlib.import_module("hamer.utils.renderer")
        renderer_file = module_file(renderer)
        has_symbols = hasattr(renderer, "Renderer") and hasattr(renderer, "cam_crop_to_full")
        failures += not mark(has_symbols, "renderer compatibility exports Renderer and cam_crop_to_full", renderer_file)
        loaded_forbidden = [name for name in FORBIDDEN_IMPORTS if name in sys.modules]
        failures += not mark(not loaded_forbidden, "renderer compatibility did not import forbidden renderer dependencies", ", ".join(loaded_forbidden) if loaded_forbidden else renderer_file)
    except Exception as exc:
        failures += not mark(False, "renderer compatibility imports", repr(exc))

    try:
        from hamer.models import load_hamer  # noqa: F401

        failures += not mark(True, "from hamer.models import load_hamer")
    except Exception as exc:
        failures += not mark(False, "from hamer.models import load_hamer", repr(exc))

    for path in REQUIRED_HAMER_FILES:
        failures += not mark(path.exists(), f"required file exists: {rel(path)}")

    missing_mano = [p for p in REQUIRED_HAMER_FILES if p.name.startswith("MANO_") and not p.exists()]
    if missing_mano:
        print("MANO_RIGHT.pkl and MANO_LEFT.pkl cannot be auto-downloaded because of MANO license restrictions.")
        print("Download them manually from MANO and place them under external/hamer/_DATA/data/mano/.")

    for package in FORBIDDEN_IMPORTS:
        already_loaded = package in sys.modules
        spec = importlib.util.find_spec(package)
        if already_loaded:
            failures += not mark(False, f"forbidden dependency not imported: {package}", "already loaded")
        else:
            detail = "installed but not imported" if spec is not None else "not installed/imported"
            failures += not mark(True, f"forbidden dependency not required: {package}", detail)

    try:
        import torch

        cuda_info = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        mark(True, "torch import", json.dumps(cuda_info))
        if not torch.cuda.is_available():
            print("[WARN] CUDA is not available in this Python process; HaMeR can run on CPU but may be slow.")
    except Exception as exc:
        failures += not mark(False, "torch import", repr(exc))

    print(f"Preflight result: {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
