import torch


def masked_hand_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    pred:   [B, 126]
    target: [B, 126]
    valid:  [B, 2]
            valid[:, 0] = left hand valid
            valid[:, 1] = right hand valid

    Left hand target coordinates are target[:, 0:63].
    Right hand target coordinates are target[:, 63:126].
    Invalid hands do not contribute to the loss.
    """

    if pred.shape[-1] != 126:
        raise ValueError(f"Expected pred shape [B, 126], got {pred.shape}")

    if target.shape[-1] != 126:
        raise ValueError(f"Expected target shape [B, 126], got {target.shape}")

    if valid.shape[-1] != 2:
        raise ValueError(f"Expected valid shape [B, 2], got {valid.shape}")

    pred = pred.view(-1, 2, 21, 3)
    target = target.view(-1, 2, 21, 3)
    valid = valid.float().view(-1, 2, 1, 1)

    squared_error = (pred - target) ** 2
    masked_error = squared_error * valid

    denom = (valid.sum() * 21 * 3).clamp_min(1.0)
    return masked_error.sum() / denom
