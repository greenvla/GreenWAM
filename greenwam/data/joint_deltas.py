# SPDX-License-Identifier: OpenMDW-1.1
"""GreenWAM q0-anchored targets with an explicit per-layout delta slot policy.

These are differences of selected state values from one observed window anchor,
including changes in velocity-valued channels when explicitly selected. No dt
division or adjacent differencing is applied. Adapters apply physical units first; normalization
and sampling happen outside this codec. The clean q0 prefix uses the existing
shared action input projector, not a separately trained state2llm.
"""

import re
from dataclasses import dataclass

import torch

Q0_CONDITIONING_SCHEMA = "q0_prefix_shared_projector_v1"
Q0_STATE_CODEC = "physical_identity_v1"
DELTA_CHANNEL_POLICY_SCHEMA = "explicit_target_slots_else_named_body_v2"
DELTA_MODES = frozenset({"forward", "forward_ar", "inverse", "wam", "policy"})


@dataclass(frozen=True)
class JointDeltaResult:
    action: torch.Tensor
    valid_mask: torch.Tensor
    conditioning_state: torch.Tensor
    conditioning_valid_mask: torch.Tensor
    joint_mask: torch.Tensor


def joint_mask_from_adapter(adapter):
    """Use declared canonical names and destination slots, never source width.

    Explicit delta_target_slots overrides name classification, permitting a
    declared full-state displacement contract. Otherwise grasp/hand/fingers
    stay absolute and only named body/arm joints are subtracted. Static and
    per-frame validity remain authoritative regardless of the selected policy.
    """
    recipe = adapter.recipe
    names, slots = recipe["canonical_names"], recipe["destination_slots"]
    if len(names) != len(slots):
        raise ValueError("Delta joint semantics require names for every mapped slot")
    mask = torch.zeros(adapter.target_dim, dtype=torch.bool)
    available = set(adapter.available_target_slots)
    if "delta_target_slots" in recipe:
        selected = recipe["delta_target_slots"]
        if (
            not isinstance(selected, list)
            or not selected
            or any(
                isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < adapter.target_dim
                for slot in selected
            )
            or len(set(selected)) != len(selected)
            or not set(selected) <= set(slots)
        ):
            raise ValueError("delta_target_slots must be unique mapped integer destination slots")
        for slot in selected:
            mask[slot] = slot in available
        if not mask.any():
            raise ValueError("delta_target_slots requires at least one available target")
        return mask
    grasp = {"gripper", "grip", "claw", "hand", "finger", "thumb", "pinky", "ring", "middle", "index"}
    body = {"joint", "axis", "hip", "knee", "ankle", "waist", "torso", "neck", "head", "shoulder", "elbow", "wrist"}
    for name, slot in zip(names, slots):
        words = set(re.findall(r"[a-z]+", name.lower()))
        if words & grasp:
            continue
        if not words & body:
            raise ValueError(f"Undeclared joint channel semantics: {name!r}")
        mask[slot] = slot in available
    if not mask.any():
        raise ValueError("delta_from_q0 requires mapped body joints")
    return mask


def _bool_mask(value, shape, name, device):
    value = torch.as_tensor(value, device=device)
    if value.dtype != torch.bool or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must be bool with shape {tuple(shape)}")
    return value


def _convert(values, q0, *, future_valid_mask, q0_valid_mask, joint_mask, inverse):
    values = torch.as_tensor(values).to(dtype=torch.float32)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("Joint targets must have nonempty shape [T,D]")
    if q0 is None:
        raise ValueError("delta_from_q0 requires one observed q0")
    q0 = torch.as_tensor(q0, device=values.device, dtype=values.dtype)
    if q0.shape != (values.shape[-1],):
        raise ValueError("q0 must be one state vector [D]")
    joints = _bool_mask(joint_mask, q0.shape, "joint_mask", values.device)
    anchor_valid = _bool_mask(q0_valid_mask, q0.shape, "q0_valid_mask", values.device) & torch.isfinite(q0)
    valid = _bool_mask(future_valid_mask, values.shape, "future_valid_mask", values.device) & torch.isfinite(values)
    if not (anchor_valid & joints).any():
        raise ValueError("delta_from_q0 requires at least one valid joint in q0")
    valid = valid & (~joints | anchor_valid).unsqueeze(0)
    anchor = torch.where(anchor_valid, q0, 0.0)
    safe = torch.where(valid, values, 0.0)
    displacement = anchor if inverse else -anchor
    converted = safe + torch.where(joints, displacement, 0.0).unsqueeze(0)
    valid = valid & torch.isfinite(converted)
    converted = torch.where(valid, converted, 0.0)
    return JointDeltaResult(converted, valid, anchor.unsqueeze(0), anchor_valid.unsqueeze(0), joints)


def encode_joint_deltas(future, q0, *, future_valid_mask, q0_valid_mask, joint_mask):
    """Encode q[k]-q0 on selected slots; all other channels stay absolute."""
    return _convert(
        future,
        q0,
        future_valid_mask=future_valid_mask,
        q0_valid_mask=q0_valid_mask,
        joint_mask=joint_mask,
        inverse=False,
    )


def decode_joint_deltas(delta, q0, *, future_valid_mask, q0_valid_mask, joint_mask):
    """After undoing normalization, reconstruct q0+delta[k], never cumsum."""
    return _convert(
        delta, q0, future_valid_mask=future_valid_mask, q0_valid_mask=q0_valid_mask, joint_mask=joint_mask, inverse=True
    )


def q0_conditioned_mode_spec(mode, *, video_length, action_length):
    """Plan for [one clean q0, N future targets]; action_length excludes q0.

    The caller must prepend the actual masked conditioning state and extend
    the validity mask too. Ordinary video includes q0 at frame zero; AR video
    contains only future frames, so its boundary q0 is at frame minus one.
    """
    from greenwam.configs.greenwam_modes import mode_spec

    if mode not in DELTA_MODES:
        raise ValueError(f"delta_from_q0 is not validated for mode {mode!r}")
    spec = mode_spec(mode, video_length=video_length, action_length=action_length)
    spec["condition_frame_indexes_action"] = (
        list(range(action_length + 1)) if mode in {"forward", "forward_ar"} else [0]
    )
    spec["action_start_frame_offset"] = -1 if mode == "forward_ar" else 0
    return spec


@dataclass(frozen=True)
class Q0PrefixActionCodec:
    """Decode normalized futures while keeping the observed state in physical units."""

    q0: torch.Tensor
    q0_valid_mask: torch.Tensor
    joint_mask: torch.Tensor
    future_valid_mask: torch.Tensor
    future_normalizer: object = None

    def decode_to_unified(self, action):
        if action.ndim != 2 or action.shape != (self.future_valid_mask.shape[0] + 1, self.q0.numel()):
            raise ValueError("q0 action decode requires one prefix and the complete future window")
        future = action[1:]
        if self.future_normalizer is not None:
            future = self.future_normalizer.denormalize_action(future)
        decoded = decode_joint_deltas(
            future,
            self.q0,
            future_valid_mask=self.future_valid_mask,
            q0_valid_mask=self.q0_valid_mask,
            joint_mask=self.joint_mask,
        )
        return torch.cat((decoded.conditioning_state, decoded.action), dim=0)


def prepare_q0_prefix(sample, *, normalizer=None):
    """Return raw absolute input, encoded input, validity and its inverse codec.

    Raw readers retain future-only deltas for statistics. This boundary alone
    adds q0. Its physical identity encoding is separate from future statistics.
    """
    required = {"action_q0", "action_q0_valid_mask", "action_joint_mask", "action_valid_mask"}
    if (
        sample.get("greenwam_joint_encoding") != "delta_from_q0"
        or sample.get("greenwam_state_conditioning") != Q0_CONDITIONING_SCHEMA
        or sample.get("greenwam_q0_codec") != Q0_STATE_CODEC
        or not required <= sample.keys()
    ):
        raise ValueError("delta_from_q0 requires observed q0 and the declared conditioning/codec contract")
    if sample.get("greenwam_action_representation") != "joint" or sample.get("history_action") is not None:
        raise ValueError("q0 prefix requires a joint view without another action history")
    q0, q0_valid = torch.as_tensor(sample["action_q0"]), torch.as_tensor(sample["action_q0_valid_mask"])
    if q0.ndim != 2 or q0.shape[0] != 1 or q0_valid.shape != q0.shape:
        raise ValueError("q0 prefix requires state and validity with shape [1,D]")
    future = sample["action"]
    decoded = decode_joint_deltas(
        future,
        q0[0],
        future_valid_mask=sample["action_valid_mask"],
        q0_valid_mask=q0_valid[0],
        joint_mask=sample["action_joint_mask"],
    )
    normalized = normalizer.normalize_action(future) if normalizer is not None else future
    normalized = torch.where(decoded.valid_mask, normalized, 0.0)
    if not torch.isfinite(normalized).all():
        raise ValueError("q0 future normalization produced nonfinite active values")
    raw = torch.cat((decoded.conditioning_state, decoded.action), dim=0)
    encoded = torch.cat((decoded.conditioning_state, normalized), dim=0)
    valid = torch.cat((decoded.conditioning_valid_mask, decoded.valid_mask), dim=0)
    codec = Q0PrefixActionCodec(
        decoded.conditioning_state[0],
        decoded.conditioning_valid_mask[0],
        decoded.joint_mask,
        decoded.valid_mask,
        normalizer,
    )
    return raw, encoded, valid, codec
