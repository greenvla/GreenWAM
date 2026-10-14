# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO preprocessing of the GreenWAM policy server: copies of the LIBERO serving pieces of the training reader
(`LIBEROLeRobotDataset`).

The served model input is the GreenWAM canvas of the agentview over the wrist camera, the observed EEF pose as the q0 row
of the action and the task as the prompt:

- canvas 384 x 288 (H x W): agentview resized to 256 x 288 on top, wrist camera resized to 128 x 288 below
  (bilinear with antialiasing, as the training reader composed it);
- q0 row: EEF position, rotation-6d and gripper open fraction from the 8-D LIBERO state (EEF position, EEF axis-angle,
  finger joints), normalized with the absolute statistics; future rows are zeros, all valid;
- reply rows: de-normalized with the relative statistics into NVIDIA's ``frame_wise_relative`` rotation-6d form
  (delta position, rotation-6d, gripper open fraction in [0, 1]).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from greenwam.serving.franka_eef import CALVIN_ACTION_DIM, CALVIN_WIDTH, rot_to_6d
from greenwam.serving.policy_prompt import VIEWPOINT_TEMPLATES

LIBERO_ROBOT = "Franka Emika Panda arm with a parallel gripper, in simulation"
LIBERO_STATS_SCHEMA = "libero_eef_statistics"
LIBERO_CANVAS_HW = (384, 288)
LIBERO_HEAD_HW = (256, 288)
LIBERO_BOTTOM_HW = (128, 288)
# Panda finger joints at full opening (+0.0388 / -0.0388); their difference over this is the gripper open fraction.
LIBERO_FINGER_SPAN = 0.0776
# The sentence the LIBERO client appends to the task for a concatenated multi-view image; the model was trained on
# the bare task (the GreenWAM JSON prompt carries the canvas framing).
CONCAT_VIEW_SENTENCE = VIEWPOINT_TEMPLATES["concat_view"]


def task_only(prompt: str) -> str:
    """Client prompt -> the bare LIBERO task: cut the concat-view sentence and the trailing period."""
    head = prompt.split(CONCAT_VIEW_SENTENCE, 1)[0]
    return head.rstrip().rstrip(".").rstrip()


def libero_absolute_physical(state: torch.Tensor) -> torch.Tensor:
    """[..., 8] LIBERO state -> [..., 10] absolute EEF xyz, rotation-6d and gripper open fraction."""
    rotation = convert_rotation(state[..., 3:6].float(), input_format="axisangle", output_format="matrix")
    opening = ((state[..., 6] - state[..., 7]) / LIBERO_FINGER_SPAN).clamp(0.0, 1.0)
    return torch.cat([state[..., 0:3].float(), rot_to_6d(rotation), opening[..., None].float()], dim=-1)


class LiberoStats:
    """Relative (future rows) and absolute (q0 row) normalization statistics of a LIBERO GreenWAM checkpoint."""

    def __init__(self, path: str | Path) -> None:
        doc = json.loads(Path(path).read_text())
        if doc.get("schema_name") != LIBERO_STATS_SCHEMA or doc.get("schema_version") != 1:
            raise ValueError(f"{path}: expected {LIBERO_STATS_SCHEMA} v1")
        for key in ("relative", "absolute"):
            offset = torch.tensor(doc[key]["offset"], dtype=torch.float32)
            scale = torch.tensor(doc[key]["scale"], dtype=torch.float32)
            if offset.shape != (CALVIN_WIDTH,) or scale.shape != (CALVIN_WIDTH,):
                raise ValueError(f"{path}: {key} statistics must have {CALVIN_WIDTH} channels")
            if not (torch.isfinite(offset).all() and torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                raise ValueError(f"{path}: {key} statistics must be finite with positive scales")
            setattr(self, f"{key}_offset", offset)
            setattr(self, f"{key}_scale", scale)


def _transport(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = torch.zeros(rows.shape[0], CALVIN_ACTION_DIM, dtype=torch.float32)
    valid = torch.zeros_like(encoded, dtype=torch.bool)
    finite = torch.isfinite(rows)
    encoded[:, :CALVIN_WIDTH] = torch.where(finite, rows, torch.zeros_like(rows))
    valid[:, :CALVIN_WIDTH] = finite
    return encoded, valid


class LiberoEEFCodec:
    """The q0 row of a served request and the de-normalization of the predicted rows."""

    def __init__(self, stats: LiberoStats) -> None:
        self.stats = stats

    def encode_q0(self, state: torch.Tensor) -> torch.Tensor:
        absolute = libero_absolute_physical(state.reshape(1, -1).float())
        return (absolute - self.stats.absolute_offset) / self.stats.absolute_scale

    def encode_inference(self, state: torch.Tensor, pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Observed q0 in row 0, zeros (noised by the sampler) after it; the future rows are all valid."""
        rows = torch.zeros(int(pred_len) + 1, CALVIN_WIDTH, dtype=torch.float32)
        rows[0] = self.encode_q0(state)[0]
        return _transport(rows)

    def relative_physical(self, predicted_future: torch.Tensor) -> torch.Tensor:
        """Normalized [L, >=10] future rows -> [L, 10] delta position, rotation-6d and gripper open fraction in [0, 1]."""
        physical = (
            predicted_future[:, :CALVIN_WIDTH].float().cpu() * self.stats.relative_scale + self.stats.relative_offset
        )
        physical[:, 9] = physical[:, 9].clamp(0.0, 1.0)
        return physical


def compose_libero_canvas(head: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    """[T, C, H, W] float views in [0, 1] -> [T, C, 384, 288] GreenWAM canvas."""
    head = F.interpolate(head, size=LIBERO_HEAD_HW, mode="bilinear", align_corners=False, antialias=True)
    wrist = F.interpolate(wrist, size=LIBERO_BOTTOM_HW, mode="bilinear", align_corners=False, antialias=True)
    return torch.cat([head, wrist], dim=-2).clamp(0.0, 1.0)


def canvas_video(image_uint8: torch.Tensor, chunk: int) -> torch.Tensor:
    """[H, 2H, 3] uint8 agentview | wrist concat -> [3, chunk + 1, 384, 288] uint8 video with the canvas in frame 0."""
    views = image_uint8.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1, 3, H, 2H]
    width = views.shape[-1] // 2
    canvas = compose_libero_canvas(views[..., :width], views[..., width:])[0]
    video = torch.zeros((3, chunk + 1, *LIBERO_CANVAS_HW), dtype=torch.uint8)
    video[:, 0] = (canvas * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    return video
