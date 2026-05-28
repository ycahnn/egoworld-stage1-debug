"""Stage 1 compatibility helpers for using HaMeR without rendering.

HaMeR's public demo path imports ``hamer.utils.renderer``, which normally pulls
in pyrender/OpenGL/EGL. Stage 1 only needs model inference, so this module
installs a tiny import-compatible renderer module that keeps those heavy GUI
backends out of the process.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import torch


RENDERER_MODULE_NAME = "hamer.utils.renderer"


class Stage1RendererNotAvailable(NotImplementedError):
    """Raised if rendering is accidentally requested in the Stage 1 path."""


def _rendering_error() -> Stage1RendererNotAvailable:
    return Stage1RendererNotAvailable(
        "Stage 1 uses HaMeR as a pure inference backend. Rendering via "
        "HaMeR Renderer/pyrender/OpenGL/EGL is intentionally unsupported. "
        "Use the Stage 1 projection/diagnostic scripts instead."
    )


class Renderer:
    """Import-compatible no-render replacement for HaMeR's Renderer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise _rendering_error()

    def visualize_all_tb(self, *args: Any, **kwargs: Any) -> Any:
        raise _rendering_error()

    def render_rgba(self, *args: Any, **kwargs: Any) -> Any:
        raise _rendering_error()


SkeletonRenderer = Renderer
MeshRenderer = Renderer


def cam_crop_to_full(
    cam_bbox: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length: float = 5000.0,
) -> torch.Tensor:
    """Convert HaMeR crop camera parameters to full-image translation."""

    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def install_hamer_renderer_stub() -> types.ModuleType:
    """Install a safe renderer module into ``sys.modules`` and return it."""

    existing = sys.modules.get(RENDERER_MODULE_NAME)
    if existing is not None:
        has_required = hasattr(existing, "Renderer") and hasattr(existing, "cam_crop_to_full")
        existing_file = str(getattr(existing, "__file__", ""))
        if has_required and "pyrender" not in existing_file.lower():
            return existing

    module = types.ModuleType(RENDERER_MODULE_NAME)
    module.__file__ = str(Path(__file__).resolve())
    module.Renderer = Renderer
    module.SkeletonRenderer = SkeletonRenderer
    module.MeshRenderer = MeshRenderer
    module.cam_crop_to_full = cam_crop_to_full
    module.Stage1RendererNotAvailable = Stage1RendererNotAvailable
    module.__all__ = [
        "Renderer",
        "SkeletonRenderer",
        "MeshRenderer",
        "cam_crop_to_full",
        "Stage1RendererNotAvailable",
    ]
    sys.modules[RENDERER_MODULE_NAME] = module
    return module
