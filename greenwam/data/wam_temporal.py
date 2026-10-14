# SPDX-License-Identifier: OpenMDW-1.1
"""WAM measured time budget; VAE padding is applied only after reading."""

import math

WAM_HORIZON_POLICY = "physical_time_floor_effective_fps_v1"


def validate_wam_horizon(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("wam_horizon_s must be positive finite seconds or None")
    return float(value)


def wam_future_frames(horizon_s, fps):
    horizon = validate_wam_horizon(horizon_s)
    if horizon is None or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("WAM physical horizon requires positive effective FPS")
    # Avoid losing the exact16th frame for the configured float16/30 seconds.
    count = math.floor(horizon * fps + 1e-9)
    return count
