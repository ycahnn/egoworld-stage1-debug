import torch
import torch.nn as nn
import timm


class EgoHandPoseViT(nn.Module):
    """
    Exocentric image -> ViT-B/16 feature -> MLP regressor -> 3D egocentric hand pose

    Input:
        x: [B, 3, 224, 224]

    Output:
        pred: [B, 126]
              126 = 2 hands * 21 joints * 3 xyz
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ViT-B/16, 224x224 input, hidden dim 768
        # num_classes=0 removes the original classification head.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features  # ViT-B/16 -> 768

        if feature_dim != 768:
            raise ValueError(f"Expected ViT feature dim 768, but got {feature_dim}")

        self.regressor = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 126),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224]
        return: [B, 768]
        """
        feat = self.backbone(x)
        return feat

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        """
        x: [B, 3, 224, 224]
        """
        feat = self.extract_feature(x)      # [B, 768]
        pred = self.regressor(feat)         # [B, 126]

        if return_feature:
            return pred, feat

        return pred