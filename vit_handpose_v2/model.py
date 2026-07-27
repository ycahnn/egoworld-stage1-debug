from __future__ import annotations

import timm
import torch
from torch import nn


class RootPoseViT(nn.Module):
    """Pretrained ViT with independent wrist and wrist-relative pose heads."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.orig_in21k",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.root_head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 256), nn.GELU(), nn.Linear(256, 6)
        )
        self.pose_head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 512), nn.GELU(), nn.Linear(512, 120)
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        roots = self.root_head(features).reshape(-1, 2, 3)
        relative = self.pose_head(features).reshape(-1, 2, 20, 3)
        return roots, relative
