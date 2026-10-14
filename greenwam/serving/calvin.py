# SPDX-License-Identifier: OpenMDW-1.1
"""CALVIN in the GreenWAM boundary for serving: canvas, q0 codec and the decode to ``cartesian_abs`` targets.

Copy of the serving half of the original CALVIN GreenWAM boundary (``calvin.py`` of the fixed-layout runs) with the
bodies unchanged; the training targets (``encode_training``, the relative window encoder) and the statistics fit
are left out.

* targets: row ``k`` is ``inverse(T_{k-1}) @ T_k`` of the measured TCP pose (``robot_obs[:6]``, URDF rpy), as
  xyz + rotation-6d, then the gripper as an open fraction of the commanded gripper action;
* q0 prefix: row 0 is the absolute measured pose + gripper, normalized with the absolute statistics and embedded with
  the absolute representation (contract 90);
* execution: poses are chained from the observed q0 and sent as CALVIN ``cartesian_abs`` targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from greenwam.serving.franka_eef import (
    CALVIN_ACTION_DIM,
    CALVIN_WIDTH,
    _resize_uint8,
    gripper_open_fraction,
    matrix_to_rpy,
    rot_from_6d,
    rot_to_6d,
    rpy_to_matrix,
)
from greenwam.serving.policy_prompt import CANVAS_CAPTION

CALVIN_LAYOUT_ID = 44
CALVIN_REL_CONTRACT_ID = 76
CALVIN_ABS_CONTRACT_ID = 90
CALVIN_ROBOT = "Franka Emika Panda arm with a parallel gripper, in simulation"

# GreenWAM canvas for a square head plus one other camera: bucket 3:4, head on the top 2/3, the other camera across the
# full bottom row, sized in the 256 class.
CALVIN_CANVAS_HW = (384, 288)
CALVIN_HEAD_HW = (256, 288)
CALVIN_BOTTOM_HW = (128, 288)
CALVIN_STORED_CAMERA_HW = (192, 192)  # stored rgb_static / rgb_gripper size of the training data

CALVIN_TCP_POS = slice(0, 3)
CALVIN_TCP_RPY = slice(3, 6)
CALVIN_GRIPPER_ACTION_INDEX = 14  # robot_obs[14]: last commanded gripper action (-1 close, 1 open)
CALVIN_STATE_DIM = 15


def _pose_matrices(robot_obs: torch.Tensor) -> torch.Tensor:
    T = torch.zeros(*robot_obs.shape[:-1], 4, 4, dtype=robot_obs.dtype, device=robot_obs.device)
    T[..., :3, :3] = rpy_to_matrix(robot_obs[..., CALVIN_TCP_RPY])
    T[..., :3, 3] = robot_obs[..., CALVIN_TCP_POS]
    T[..., 3, 3] = 1.0
    return T


def calvin_absolute_physical(robot_obs: torch.Tensor) -> torch.Tensor:
    """[..., 15] robot_obs -> [..., 10] absolute pose (xyz, rotation-6d) + gripper open fraction."""
    R = rpy_to_matrix(robot_obs[..., CALVIN_TCP_RPY])
    grip = gripper_open_fraction(robot_obs[..., CALVIN_GRIPPER_ACTION_INDEX])
    return torch.cat([robot_obs[..., CALVIN_TCP_POS], rot_to_6d(R), grip[..., None]], dim=-1)


@dataclass(frozen=True)
class CalvinStats:
    relative_offset: torch.Tensor
    relative_scale: torch.Tensor
    absolute_offset: torch.Tensor
    absolute_scale: torch.Tensor

    @classmethod
    def load(cls, path: str | Path) -> CalvinStats:
        doc = json.loads(Path(path).read_text())
        if doc.get("schema_name") != "calvin_eef_statistics" or doc.get("schema_version") != 1:
            raise ValueError(f"{path}: expected calvin_eef_statistics v1")
        rows = {}
        for key in ("relative", "absolute"):
            row = doc[key]
            offset = torch.tensor(row["offset"], dtype=torch.float32)
            scale = torch.tensor(row["scale"], dtype=torch.float32)
            if offset.shape != (CALVIN_WIDTH,) or scale.shape != (CALVIN_WIDTH,):
                raise ValueError(f"{path}: {key} statistics must have {CALVIN_WIDTH} channels")
            if not (torch.isfinite(offset).all() and torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                raise ValueError(f"{path}: {key} statistics must be finite with positive scales")
            rows[key] = (offset, scale)
        return cls(*rows["relative"], *rows["absolute"])


class CalvinEEFCodec:
    """Encode the observed q0 into a GreenWAM transport128 row and decode predictions into ``cartesian_abs`` targets."""

    def __init__(self, stats: CalvinStats) -> None:
        self.stats = stats

    def _transport(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.zeros(rows.shape[0], CALVIN_ACTION_DIM, dtype=torch.float32)
        valid = torch.zeros_like(encoded, dtype=torch.bool)
        finite = torch.isfinite(rows)
        encoded[:, :CALVIN_WIDTH] = torch.where(finite, rows, torch.zeros_like(rows))
        valid[:, :CALVIN_WIDTH] = finite
        return encoded, valid

    def encode_q0(self, robot_obs: torch.Tensor) -> torch.Tensor:
        absolute = calvin_absolute_physical(robot_obs.reshape(1, -1).float())
        return (absolute - self.stats.absolute_offset) / self.stats.absolute_scale

    def encode_inference(self, robot_obs: torch.Tensor, pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Observed q0 in row 0, zeros (noised by the sampler) after it; the future rows are all valid."""
        rows = torch.zeros(int(pred_len) + 1, CALVIN_WIDTH, dtype=torch.float32)
        rows[0] = self.encode_q0(robot_obs)[0]
        return self._transport(rows)

    def decode(self, predicted_future: torch.Tensor, robot_obs: torch.Tensor) -> np.ndarray:
        """Normalized [L,>=10] framewise predictions + observed robot_obs -> [L,7] cartesian_abs rows.

        Each row is (x, y, z, roll, pitch, yaw, gripper) with the gripper as a CALVIN command in {-1, 1}.
        """
        pred = predicted_future[:, :CALVIN_WIDTH].float().cpu()
        physical = pred * self.stats.relative_scale + self.stats.relative_offset
        T = _pose_matrices(robot_obs.reshape(1, -1).float().cpu())[0]
        rows = []
        for k in range(physical.shape[0]):
            delta = torch.eye(4)
            delta[:3, :3] = rot_from_6d(physical[k, 3:9])
            delta[:3, 3] = physical[k, :3]
            T = T @ delta
            grip = 1.0 if float(physical[k, 9]) >= 0.5 else -1.0
            rows.append(torch.cat([T[:3, 3], matrix_to_rpy(T[:3, :3]), torch.tensor([grip])]))
        return torch.stack(rows).numpy().astype(np.float32)


def load_calvin_codec(stats_path: str | Path) -> CalvinEEFCodec:
    return CalvinEEFCodec(CalvinStats.load(stats_path))


def build_calvin_canvas(rgb_static: np.ndarray, rgb_gripper: np.ndarray) -> np.ndarray:
    """Serving canvas matching training: raw -> stored 192x192 -> head 256x288 on top, gripper 128x288 below."""
    head = _resize_uint8(_resize_uint8(rgb_static, CALVIN_STORED_CAMERA_HW), CALVIN_HEAD_HW)
    grip = _resize_uint8(_resize_uint8(rgb_gripper, CALVIN_STORED_CAMERA_HW), CALVIN_BOTTOM_HW)
    return np.concatenate([head, grip], axis=0)


def _ensure_rgb_uint8_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key!r} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def build_calvin_sample(
    obs: dict[str, Any],
    prediction_horizon: int,
    codec: CalvinEEFCodec,
    *,
    conditioning_fps: float,
    conditioning_fps_action: float,
    domain_id: int,
) -> dict[str, Any]:
    """One served observation -> the raw GreenWAM policy sample of the original CALVIN server (before the transform).

    The canvas is frame 0 of an otherwise empty [3, H+1, 384, 288] clip; the action is the q0 row plus H zero rows,
    all valid on the ten layout channels. A missing gripper frame repeats the static one.
    """
    static = _ensure_rgb_uint8_image(obs["rgb_static"], "rgb_static")
    gripper_raw = obs.get("rgb_gripper")
    gripper = static if gripper_raw is None else _ensure_rgb_uint8_image(gripper_raw, "rgb_gripper")
    image = build_calvin_canvas(static, gripper)
    state = torch.as_tensor(np.asarray(obs["state"], dtype=np.float32)).reshape(-1)
    if state.numel() != CALVIN_STATE_DIM or not bool(torch.isfinite(state).all()):
        raise ValueError("CALVIN GreenWAM serving needs the finite 15-dim robot_obs as 'state'")
    action, valid = codec.encode_inference(state, prediction_horizon)
    video = torch.zeros((3, prediction_horizon + 1, *CALVIN_CANVAS_HW), dtype=torch.uint8)
    video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)
    return {
        "ai_caption": obs["prompt"],
        "video": video,
        "action": action,
        "action_valid_mask": valid,
        "conditioning_fps": torch.tensor(conditioning_fps, dtype=torch.float32),
        "conditioning_fps_action": torch.tensor(conditioning_fps_action, dtype=torch.float32),
        "mode": "policy",
        "domain_id": torch.tensor(domain_id, dtype=torch.long),
        "viewpoint": "concat_view",
        "additional_view_description": CANVAS_CAPTION,
        "greenwam_robot": CALVIN_ROBOT,
    }
