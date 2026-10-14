# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rectified-flow matching loss (vision / action / sound modalities).

Extracted from OmniMoTModel._compute_flow_matching_loss. The loss math is
unchanged; the only structural change is that ``tensor_kwargs_fp32`` is now
passed explicitly instead of being read from ``self``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.action.utils.action_masks import expand_action_valid_mask
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_SLOT_GROUPS,
)
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow

ACTION_SLOT_SAMPLE_LOSS_KEY = "_action_slot_sample_loss"
ACTION_SLOT_SAMPLE_COUNT_KEY = "_action_slot_sample_count"


@dataclass(frozen=True)
class ActionSlotLossStats:
    """Detached per-sample slot losses and contribution indicators, shape ``[B,S]``."""

    sample_loss: torch.Tensor
    sample_count: torch.Tensor


def compute_flow_matching_loss(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    timesteps: torch.Tensor,
    has_valid_tokens: bool,
    rectified_flow: RectifiedFlow,
    tensor_kwargs_fp32: dict,
    raw_action_dim: list[torch.Tensor] | None = None,
    action_valid_mask: list[torch.Tensor] | None = None,
    normalize_by_active: bool = False,
    action_slot_stats: ActionSlotLossStats | None = None,
    action_body_part_ids: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow matching loss for a modality.

    Args:
        pred: Predicted velocity field (list of tensors, one per sample).
        target: Target velocity field (list of tensors, one per sample).
            Under rectified flow the target is ``v = eps - x0``.
        condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
        timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
            base/teacher_forcing (all frames share one timestep) or [B,T_max]
            for diffusion_forcing (per-frame independent timesteps). Time weights
            are applied per-frame before averaging, so non-uniform weight functions
            are handled correctly.
        has_valid_tokens: Whether this modality has valid noisy tokens.
        rectified_flow: The rectified flow object for time weighting.
        tensor_kwargs_fp32: Dict of dtype/device kwargs forwarded to
            ``rectified_flow.train_time_weight``.
        raw_action_dim: Optional unpadded Action width for each sample.
        action_valid_mask: Optional per-sample [D] or [T,D] validity masks. Invalid
            action channels are excluded from both numerator and denominator.
        normalize_by_active: When True, normalize per-instance loss by the count of
            active (noisy) elements rather than all elements. Preserves the
            ``sum / active_count`` semantics needed for distillation critics where
            conditioned frames contribute no signal and should not dilute the
            denominator.
        action_slot_stats: Optional collector for detached normalized per-sample
            losses over the canonical unified Action slots.
        action_body_part_ids: Optional [D] or [T,D] IDs (legs/body/arms/hands =
            0..3, unmapped = -1). Average valid coordinate losses within each
            part, then average contributing parts. GreenWAM semantic path only.

    Returns:
        tuple: A tuple containing two elements:
            - Flow matching loss (or dummy loss for gradient consistency).
            - Per-instance loss (or dummy loss for gradient consistency).
    """
    if not has_valid_tokens:
        # Dummy loss to maintain backward graph consistency across ranks. The per-instance
        # view keeps one element per packed item: consumers pair it with the per-item sigma
        # (sigma_loss_analysis), and a packed microbatch of conditioning-only items (GreenWAM inverse,
        # max_samples_per_batch > 1) otherwise fails that pairing with a single dummy element.
        dummy_loss = 0.0 * sum(p.sum() for p in pred)
        return dummy_loss, dummy_loss.unsqueeze(0).expand(max(1, len(pred)))

    # condition_mask[i] is T-first with trailing singletons: [T,1,1] vision, [T,1] action.
    # tw_i gets the same shape so w(σ_t) broadcasts element-wise over non-T dims.
    per_instance_losses = []
    per_instance_weighted_losses = []
    for i in range(len(pred)):
        T_i = condition_mask[i].shape[0]
        error_i = pred[i] - target[i]  # vision:[C,T,H,W]  action/sound:[T,D]
        noisy_mask_i = 1.0 - condition_mask[i]  # vision:[T,1,1]  action/sound:[T,1]
        if raw_action_dim is not None and raw_action_dim[i] is not None:
            error_i = error_i[:, : raw_action_dim[i]]
        slot_mask_i = None
        if action_valid_mask is not None and action_valid_mask[i] is not None:
            if action_valid_mask[i].shape[-1] < error_i.shape[-1]:
                raise ValueError(f"action_valid_mask width {action_valid_mask[i].shape[-1]} < action width {error_i.shape[-1]}")
            slot_mask_i = expand_action_valid_mask(action_valid_mask[i], error_i)
            # Mask before squaring: multiplying NaN by zero after the square
            # still produces NaN in both the loss and its backward pass.
            error_i = error_i.masked_fill(~slot_mask_i, 0)
        sqerr_i = error_i.square()
        if normalize_by_active:
            if slot_mask_i is not None:
                active_count = (noisy_mask_i * slot_mask_i).sum().clamp(min=1)
            else:
                active_channels = sqerr_i.numel() // noisy_mask_i.numel()
                active_count = (noisy_mask_i.sum() * active_channels).clamp(min=1)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_count)  # []
        elif slot_mask_i is not None:
            # Preserve the default ``.mean()`` contract: conditioned timesteps
            # remain in the denominator, while only invalid channels are removed.
            # ``normalize_by_active=True`` intentionally uses noisy timesteps only.
            active_count = slot_mask_i.sum().clamp(min=1)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_count)  # []
        else:
            per_instance_losses.append((sqerr_i * noisy_mask_i).mean())  # []

        ts_i = timesteps[i, :T_i] if timesteps.dim() > 1 else timesteps[i]  # DF:[T_i]  TF:[1]
        tw_i = rectified_flow.train_time_weight(ts_i, tensor_kwargs_fp32)  # DF:[T_i]  TF:[1]
        tw_i = tw_i.reshape(-1, *([1] * (condition_mask[i].ndim - 1)))  # vision:[T_i,1,1]  action/sound:[T_i,1]
        weighted_sqerr_i = sqerr_i * tw_i * noisy_mask_i
        if normalize_by_active or slot_mask_i is not None:
            per_instance_weighted_losses.append(weighted_sqerr_i.sum() / active_count)
        else:
            per_instance_weighted_losses.append(weighted_sqerr_i.mean())

        if action_body_part_ids is not None:
            parts = action_body_part_ids[i].to(device=error_i.device)
            if parts.dtype not in (torch.int32, torch.int64):
                raise ValueError("action_body_part_ids must contain integer part IDs")
            if parts.ndim == 1:
                parts = parts.unsqueeze(0).expand(error_i.shape[0], -1)
            parts = parts[:, :error_i.shape[-1]]
            if parts.shape != error_i.shape or ((parts < -1) | (parts > 3)).any():
                raise ValueError("action_body_part_ids must match action shape and use -1..3")
            valid = slot_mask_i if slot_mask_i is not None else torch.ones_like(error_i, dtype=torch.bool)
            if (valid & (parts < 0)).any():
                raise ValueError("Valid action coordinates must have a semantic body part")
            losses, weighted_losses, contributions = [], [], []
            for part in range(4):
                selected = valid & (parts == part)
                active = selected * noisy_mask_i
                count = active.sum() if normalize_by_active else selected.sum()
                contributes = active.sum().gt(0).to(sqerr_i.dtype)
                losses.append((sqerr_i * active).sum() / count.clamp(min=1) * contributes)
                weighted_losses.append((weighted_sqerr_i * selected).sum() / count.clamp(min=1) * contributes)
                contributions.append(contributes)
            part_count = torch.stack(contributions).sum().clamp(min=1)
            per_instance_losses[-1] = torch.stack(losses).sum() / part_count
            per_instance_weighted_losses[-1] = torch.stack(weighted_losses).sum() / part_count

        if (
            action_slot_stats is not None
            and raw_action_dim is not None
            and raw_action_dim[i] is not None
            and slot_mask_i is not None
        ):
            with torch.no_grad():
                is_unified = torch.as_tensor(raw_action_dim[i], device=weighted_sqerr_i.device).eq(UNIFIED_ACTION_DIM)
                is_unified = is_unified.to(dtype=torch.float32)
                detached_sqerr = weighted_sqerr_i.detach()
                sample_slot_losses: list[torch.Tensor] = []
                sample_slot_counts: list[torch.Tensor] = []
                for _, slot in UNIFIED_ACTION_SLOT_GROUPS:
                    slot_valid = slot_mask_i[:, slot]
                    slot_count = (slot_valid * noisy_mask_i).sum() if normalize_by_active else slot_valid.sum()
                    sample_count = slot_count.gt(0).to(dtype=torch.float32) * is_unified
                    slot_denominator = slot_count.clamp(min=1)
                    slot_loss_sum = detached_sqerr[..., slot].sum(dtype=torch.float32)
                    sample_slot_losses.append(slot_loss_sum / slot_denominator * sample_count)
                    sample_slot_counts.append(sample_count)
                action_slot_stats.sample_loss[i].add_(torch.stack(sample_slot_losses))
                action_slot_stats.sample_count[i].add_(torch.stack(sample_slot_counts))

    per_instance_loss = torch.stack(per_instance_losses)  # [B]
    per_instance_weighted_loss = torch.stack(per_instance_weighted_losses)  # [B]
    return per_instance_weighted_loss.mean(), per_instance_loss
