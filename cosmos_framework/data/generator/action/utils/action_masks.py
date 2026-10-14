# SPDX-License-Identifier: OpenMDW-1.1
"""Broadcast static or time-dependent measurement validity at action boundaries."""

import torch
import torch.nn.functional as F


def expand_action_valid_mask(mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Return boolean [T,D] validity, padding absent channels with False.

    A static [D] mask broadcasts across time. A [T,D] mask must match the
    actual time axis; flattening it would silently supervise different joints.
    Wider masks are cropped when the caller has already removed model padding.
    """
    if mask.dtype != torch.bool or mask.ndim not in (1, 2):
        raise ValueError("action_valid_mask must be a bool tensor with shape [D] or [T,D]")
    if action.ndim != 2:
        raise ValueError("action values must have shape [T,D]")
    if mask.ndim == 2 and mask.shape[0] != action.shape[0]:
        raise ValueError(f"action_valid_mask time axis {mask.shape[0]} != action time axis {action.shape[0]}")
    mask = mask.to(device=action.device)[..., :action.shape[-1]]
    mask = F.pad(mask, (0, action.shape[-1] - mask.shape[-1]), value=False)
    return mask.expand_as(action)


def mask_action_values(action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero invalid values, including NaNs, before projection or diffusion."""
    return action.masked_fill(~expand_action_valid_mask(mask, action), 0)
