# SPDX-License-Identifier: OpenMDW-1.1
"""Rigid-transform helpers shared by the action projectors and the end-effector state codec."""

import torch


def se3_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    """(B,...,3), (B,...,4) -> (B,...,4,4). The quaternion is normalised."""
    if order == "wxyz":
        quat = quat[..., [1, 2, 3, 0]]
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    T = torch.zeros(*q.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def se3_from_pos_rpy(pos: torch.Tensor, rpy: torch.Tensor) -> torch.Tensor:
    """(B,...,3), (B,...,3 roll/pitch/yaw rad) -> (B,...,4,4). URDF rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    (extrinsic x-y-z), the convention an FK check (rpy2R) verified against the
    RoboCOIN/GM-100 FK-patched Euler columns."""
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    R = torch.stack(
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
    T = torch.zeros(*rpy.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def relative_se3(T_prev: torch.Tensor, T_curr: torch.Tensor) -> torch.Tensor:
    """dT = T_prev^-1 * T_curr. Analytic inverse, not a solve."""
    R, t = T_prev[..., :3, :3], T_prev[..., :3, 3]
    Rt = R.transpose(-1, -2)
    T_inv = torch.zeros_like(T_prev)
    T_inv[..., :3, :3] = Rt
    T_inv[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rt, t)
    T_inv[..., 3, 3] = 1.0
    return T_inv @ T_curr


def rot_to_6d(R: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix: a continuous representation (CVPR'19)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
