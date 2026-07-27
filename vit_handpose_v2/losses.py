from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RootPoseLossOutput:
    total: torch.Tensor
    root: torch.Tensor
    pose: torch.Tensor


def root_pose_loss(
    pred_root: torch.Tensor,
    pred_pose: torch.Tensor,
    target_root: torch.Tensor,
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    root_weight: float = 0.1,
    pose_weight: float = 1.0,
) -> RootPoseLossOutput:
    hand_mask = valid.float()
    root_error = torch.square(pred_root - target_root) * hand_mask[..., None]
    pose_error = torch.square(pred_pose - target_pose) * hand_mask[..., None, None]
    valid_hands = hand_mask.sum().clamp_min(1.0)
    root_loss = root_error.sum() / (valid_hands * 3)
    pose_loss = pose_error.sum() / (valid_hands * 20 * 3)
    total = root_weight * root_loss + pose_weight * pose_loss
    return RootPoseLossOutput(total=total, root=root_loss, pose=pose_loss)


@torch.no_grad()
def metric_sums(
    pred_root: torch.Tensor,
    pred_pose: torch.Tensor,
    target_root: torch.Tensor,
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    compute_pa: bool = False,
) -> dict[str, float]:
    mask = valid.float()
    root_distance = torch.linalg.vector_norm(pred_root - target_root, dim=-1)
    pose_distance = torch.linalg.vector_norm(pred_pose - target_pose, dim=-1)
    pred_joints = torch.cat([pred_root[:, :, None], pred_pose + pred_root[:, :, None]], dim=2)
    target_joints = torch.cat([target_root[:, :, None], target_pose + target_root[:, :, None]], dim=2)
    absolute_distance = torch.linalg.vector_norm(pred_joints - target_joints, dim=-1)
    valid_hands = mask.sum().item()
    result = {
        "root_distance_sum": (root_distance * mask).sum().item(),
        "pose_distance_sum": (pose_distance * mask[..., None]).sum().item(),
        "absolute_distance_sum": (absolute_distance * mask[..., None]).sum().item(),
        "valid_hands": valid_hands,
    }
    if compute_pa:
        selected = valid.reshape(-1) > 0.5
        source = pred_joints.reshape(-1, 21, 3)[selected].float()
        target = target_joints.reshape(-1, 21, 3)[selected].float()
        source_mean = source.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        source_centered = source - source_mean
        target_centered = target - target_mean
        source_norm = torch.linalg.vector_norm(
            source_centered, dim=(1, 2), keepdim=True
        ).clamp_min(1e-8)
        target_norm = torch.linalg.vector_norm(
            target_centered, dim=(1, 2), keepdim=True
        ).clamp_min(1e-8)
        source_unit = source_centered / source_norm
        target_unit = target_centered / target_norm
        covariance = source_unit.transpose(1, 2) @ target_unit
        u, singular_values, vh = torch.linalg.svd(covariance)
        correction = torch.ones_like(singular_values)
        correction[:, -1] = torch.sign(torch.det(u @ vh))
        rotation = u @ torch.diag_embed(correction) @ vh
        scale = (singular_values * correction).sum(dim=1, keepdim=True).unsqueeze(-1)
        scale = scale * target_norm / source_norm
        aligned = scale * (source_centered @ rotation) + target_mean
        result["pa_distance_sum"] = torch.linalg.vector_norm(
            aligned - target, dim=-1
        ).sum().item()
    return result
