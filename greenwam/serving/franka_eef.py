# SPDX-License-Identifier: OpenMDW-1.1
"""Single-arm Franka EEF in the GreenWAM transport: pose math and camera resize shared by the CALVIN and LIBERO servers.

Copies of the functions of the original CALVIN GreenWAM boundary (``calvin.py`` of the fixed-layout runs) with their
bodies unchanged. The layout puts one effector on the right-arm EEF slots of the shared projector (x, y, z,
rotation-6d) plus the right gripper, ``CALVIN_WIDTH`` channels of a ``CALVIN_ACTION_DIM``-wide transport row
(``benchmarks/semantic_extensions/franka_eef_v1.json``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CALVIN_ACTION_DIM = 128
CALVIN_WIDTH = 10  # right EEF xyz (3) + rotation-6d (6) + right gripper (1)


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """URDF / PyBullet rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (GreenWAM ``se3_from_pos_rpy``)."""
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    return torch.stack(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ],
        dim=-1,
    ).reshape(*rpy.shape[:-1], 3, 3)


def matrix_to_rpy(R: torch.Tensor) -> torch.Tensor:
    """Inverse of ``rpy_to_matrix`` (pitch in [-pi/2, pi/2])."""
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1.0, 1.0))
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


def rot_to_6d(R: torch.Tensor) -> torch.Tensor:
    """GreenWAM ``rot_to_6d``: the first two rotation columns."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot_from_6d(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt back to a rotation whose first two columns follow ``rot_to_6d``."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def gripper_open_fraction(command: torch.Tensor) -> torch.Tensor:
    """CALVIN gripper command (-1 close, 1 open) -> GreenWAM open fraction (0 closed, 1 open)."""
    return (command > 0).to(command.dtype if command.dtype.is_floating_point else torch.float32)


def _resize_uint8(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=hw, mode="bilinear", align_corners=False, antialias=True)
    return resized.round().clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
