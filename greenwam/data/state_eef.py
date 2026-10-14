# SPDX-License-Identifier: OpenMDW-1.1
"""Qualified measured-state EEF targets; separate from native checkpoint replay.

Each action is inverse(T_anchor) @ T_target in the previous TCP frame, followed
by target-state hand/gripper positions. Family-specific contracts deliberately
do not assert cross-robot TCP axes or native Cosmos domain compatibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from greenwam.data.shared_projectors import AdaptedState

DEFAULT_STATE_EEF_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "registry" / "state_eef.json"
STATE_EEF_BANK_OFFSET = 66
# Relative EEF families occupy [66, 80); absolute EEF ids mirror them from 80 (room for 14 families).
STATE_EEF_ABSOLUTE_BANK_OFFSET = 80
ROTATION_FORMATS = {"quat_xyzw": 7, "euler_rpy_rad": 6}


@dataclass(frozen=True)
class StateEEFAdapter:
    layout_id: str
    family_id: str
    family_key: str
    action_contract_id: int
    recipe: Mapping[str, Any]
    encoding: str = "relative"

    def with_encoding(self, encoding: str) -> StateEEFAdapter:
        if encoding not in {"absolute", "relative"}:
            raise ValueError("EEF encoding must be absolute or relative")
        offset = STATE_EEF_ABSOLUTE_BANK_OFFSET - STATE_EEF_BANK_OFFSET
        relative_id = self.action_contract_id - (offset if self.encoding == "absolute" else 0)
        return replace(
            self, encoding=encoding, action_contract_id=relative_id + (offset if encoding == "absolute" else 0)
        )

    @property
    def ready(self) -> bool:
        return self.recipe["geometry_status"] == "verified_state_geometry"

    @property
    def target_dim(self) -> int:
        return self.recipe["target_dim"]

    @property
    def native_width(self) -> int:
        return self.target_dim

    @property
    def n_effectors(self) -> int:
        return len(self.recipe["pose_indices"])

    @property
    def rotation_format(self) -> str:
        return self.recipe.get("rotation_format", "quat_xyzw")

    @property
    def action_contract(self) -> str:
        prefix = "state_eef_absolute_v1/" if self.encoding == "absolute" else "state_eef_v1/"
        return prefix + self.family_id

    def adapt(self, states, *, pair_valid=None) -> AdaptedState:
        """Absolute source-frame poses, independently valid at each target time."""
        import torch

        if self.encoding != "absolute":
            raise ValueError("Relative EEF requires measured endpoint pairs")
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.ndim != 2 or states.shape[-1] != self.recipe["raw_state_width"]:
            raise ValueError(f"{self.layout_id}: expected [T,{self.recipe['raw_state_width']}] state shape")
        # Identity is only a coordinate conversion, never a measured input or
        # conditioning anchor. Each effector retains its source reference frame.
        origin = torch.zeros_like(states)
        if self.rotation_format == "quat_xyzw":
            for indices in self.recipe["pose_indices"]:
                origin[:, indices[-1]] = 1.0  # identity quaternion; Euler zeros are already the identity
        return self.with_encoding("relative").adapt_pairs(origin, states, pair_valid=pair_valid)

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(self.recipe["source_columns"])

    @property
    def source_column_shapes(self) -> Mapping[str, Any]:
        return self.recipe["source_column_shapes"]

    def _concat_columns(self, columns: Mapping[str, Any]) -> Any:
        import torch

        parts = []
        for key in self.source_columns:
            value = torch.as_tensor(columns[key])
            width = math.prod(self.source_column_shapes[key])
            if value.ndim == 1 and width == 1:
                value = value.unsqueeze(-1)
            if value.ndim != 2 or value.shape[-1] != width:
                raise ValueError(f"{self.layout_id}: invalid source column shape for {key}")
            parts.append(value)
        return torch.cat(parts, dim=-1)

    def adapt_column_pairs(self, anchors, targets, *, pair_valid=None) -> AdaptedState:
        return self.adapt_pairs(self._concat_columns(anchors), self._concat_columns(targets), pair_valid=pair_valid)

    def adapt_pairs(self, anchor_states, target_states, *, pair_valid=None) -> AdaptedState:
        """Convert aligned [T,D] measured-state pairs without reading native actions.

        Invalid pose at either endpoint masks all nine channels of that effector.
        Quaternions must already be unit length within source float tolerance;
        normalization only removes numerical drift, never repairs invalid poses.
        Gripper validity is independent and never inferred from geometry.
        """
        import torch

        if self.encoding != "relative":
            raise ValueError("adapt_pairs is reserved for relative EEF; use adapt for absolute poses")

        from greenwam.proj.se3 import relative_se3, rot_to_6d, se3_from_pos_quat, se3_from_pos_rpy

        anchor = torch.as_tensor(anchor_states).to(dtype=torch.float32)
        target = torch.as_tensor(target_states, device=anchor.device).to(dtype=torch.float32)
        if anchor.ndim != 2 or anchor.shape != target.shape or anchor.shape[-1] != self.recipe["raw_state_width"]:
            raise ValueError(f"{self.layout_id}: expected equal [T,{self.recipe['raw_state_width']}] state shapes")
        count = anchor.shape[0]
        action = anchor.new_zeros(count, self.target_dim)
        valid = torch.zeros_like(action, dtype=torch.bool)
        available = torch.zeros(self.target_dim, device=anchor.device, dtype=torch.bool)
        for effector, indices in enumerate(self.recipe["pose_indices"]):
            previous, future = anchor[:, indices], target[:, indices]
            pose_valid = torch.isfinite(previous).all(-1) & torch.isfinite(future).all(-1)
            if self.rotation_format == "quat_xyzw":
                pose_valid &= (previous[:, 3:].norm(dim=-1) - 1).abs() <= self.recipe["quaternion_norm_tolerance"]
                pose_valid &= (future[:, 3:].norm(dim=-1) - 1).abs() <= self.recipe["quaternion_norm_tolerance"]
            previous = torch.where(pose_valid[:, None], previous, torch.zeros_like(previous))
            future = torch.where(pose_valid[:, None], future, torch.zeros_like(future))
            if self.rotation_format == "quat_xyzw":
                # Invalid rows become finite identity transforms before any arithmetic.
                previous[:, 6] = torch.where(pose_valid, previous[:, 6], 1.0)
                future[:, 6] = torch.where(pose_valid, future[:, 6], 1.0)
                delta = relative_se3(
                    se3_from_pos_quat(previous[:, :3], previous[:, 3:]), se3_from_pos_quat(future[:, :3], future[:, 3:])
                )
            else:  # euler_rpy_rad: zeros are the identity, no normalisation to apply
                delta = relative_se3(
                    se3_from_pos_rpy(previous[:, :3], previous[:, 3:]), se3_from_pos_rpy(future[:, :3], future[:, 3:])
                )
            block = slice(effector * 9, (effector + 1) * 9)
            values = torch.cat([delta[:, :3, 3], rot_to_6d(delta[:, :3, :3])], dim=-1)
            action[:, block] = values
            valid[:, block] = pose_valid[:, None] & torch.isfinite(values)
            available[block] = True
        for channel in self.recipe["hand_channels"]:
            slot = channel["target_slot"]
            if not channel["available"]:
                continue
            raw = target[:, channel["source_index"]]
            value = raw * channel["scale"] + channel["offset"]
            channel_valid = torch.isfinite(raw) & torch.isfinite(value)
            if channel.get("valid_range") is not None:
                lo, hi = channel["valid_range"]
                channel_valid &= (raw >= lo) & (raw <= hi)
            action[:, slot], valid[:, slot], available[slot] = value, channel_valid, True
        if pair_valid is not None:
            pair_valid = torch.as_tensor(pair_valid, device=anchor.device)
            if pair_valid.dtype != torch.bool or pair_valid.shape != (count,):
                raise ValueError("pair_valid must be bool [T]")
            valid &= pair_valid[:, None]
        return AdaptedState(torch.where(valid, action, torch.zeros_like(action)), available, valid)


class StateEEFRegistry:
    def __init__(self, doc: Mapping[str, Any]) -> None:
        if doc.get("schema_version") != 1 or doc.get("representation") != "state_eef":
            raise ValueError("expected state_eef schema_version=1")
        self.doc = doc
        self.layouts: dict[str, StateEEFAdapter] = {}
        families = {f["family_id"]: f for f in doc["families"]}
        ids = [f["action_contract_id"] for f in doc["families"]]
        if (
            not families
            or len(families) != len(ids)
            or sorted(ids) != list(range(STATE_EEF_BANK_OFFSET, STATE_EEF_BANK_OFFSET + len(ids)))
        ):
            raise ValueError("state EEF families require contiguous new IDs starting at 66")
        self.bank_size = STATE_EEF_ABSOLUTE_BANK_OFFSET + len(ids)
        self.family_count = len(families)
        for entry in doc["layouts"]:
            lid, fid, recipe = entry["layout_id"], entry["family_id"], entry["recipe"]
            if lid in self.layouts or fid not in families:
                raise ValueError(f"duplicate layout or unknown family: {lid}")
            self._validate(lid, recipe, families[fid])
            self.layouts[lid] = StateEEFAdapter(
                lid, fid, families[fid]["family_key"], families[fid]["action_contract_id"], recipe
            )

    @staticmethod
    def _validate(lid, r, family):
        fmt = r.get("rotation_format", "quat_xyzw")
        if fmt not in ROTATION_FORMATS or (fmt == "quat_xyzw") != (r["quat_order"] == "xyzw"):
            raise ValueError(f"{lid}: unsupported rotation format / quaternion order")
        if (
            r["geometry_status"] != "verified_state_geometry"
            or r["position_unit"] != "m"
            or r["native_action_used"]
            or r["native_checkpoint_compatible"]
            or r["temporal_convention"] != "backward_framewise"
            or not r["evidence"]
        ):
            raise ValueError(f"{lid}: unsupported or unqualified EEF geometry")
        width, dim = r["raw_state_width"], r["target_dim"]
        if width <= 0 or dim != family["target_dim"] or dim < 18:
            raise ValueError(f"{lid}: invalid source/family shape")
        pose_width = ROTATION_FORMATS[fmt]
        if len(r["pose_indices"]) != 2 or any(
            len(ix) != pose_width or len(set(ix)) != pose_width for ix in r["pose_indices"]
        ):
            raise ValueError(f"{lid}: two xyz+{'quat_xyzw' if fmt == 'quat_xyzw' else 'rpy'} effectors required")
        if set(r["pose_indices"][0]) & set(r["pose_indices"][1]):
            raise ValueError(f"{lid}: effectors must use distinct source pose blocks")
        indices = [i for ix in r["pose_indices"] for i in ix] + [c["source_index"] for c in r["hand_channels"]]
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < width for i in indices):
            raise ValueError(f"{lid}: source index outside state width")
        slots = [c["target_slot"] for c in r["hand_channels"]]
        if len(set(slots)) != len(slots) or any(not 18 <= s < dim for s in slots):
            raise ValueError(f"{lid}: invalid hand destination slot")
        for c in r["hand_channels"]:
            if not all(math.isfinite(v) for v in [c["scale"], c["offset"]]):
                raise ValueError(f"{lid}: invalid hand codec")
            if not isinstance(c["available"], bool) or c["unit"] not in {
                "open_fraction",
                "source_fraction",
                "source_normalized_position",
                "source_hand_position",
                "rad",
            }:
                raise ValueError(f"{lid}: ambiguous hand availability or unit")
            limits = c.get("valid_range")
            if limits is not None and (
                not isinstance(limits, (list, tuple))
                or len(limits) != 2
                or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in limits)
                or limits[0] > limits[1]
            ):
                raise ValueError(f"{lid}: invalid hand validity range")
        columns, shapes = r["source_columns"], r["source_column_shapes"]
        if len(set(columns)) != len(columns) or any(not c.startswith("observation.state") for c in columns):
            raise ValueError(f"{lid}: only declared observation.state columns are allowed")
        if sum(math.prod(shapes[c]) for c in columns) != width:
            raise ValueError(f"{lid}: source column shape mismatch")
        if not 0 < r["quaternion_norm_tolerance"] <= 0.01:
            raise ValueError(f"{lid}: invalid quaternion norm tolerance")

    def resolve(self, layout_id: str, *, encoding="relative") -> StateEEFAdapter | None:
        adapter = self.layouts.get(layout_id)
        return None if adapter is None else adapter.with_encoding(encoding)


def load_state_eef_registry(path: str | Path | None = None) -> StateEEFRegistry:
    path = DEFAULT_STATE_EEF_REGISTRY_PATH if path is None else Path(path)
    with path.open(encoding="utf-8") as handle:
        return StateEEFRegistry(json.load(handle))
