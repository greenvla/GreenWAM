# SPDX-License-Identifier: OpenMDW-1.1
"""Live GreenWAM canvas for RoboTwin 2.0: the geometry the training clips were materialized with.

The train canvas came from ``greenwam_convert.writer._write_pano_clip`` → ``canvas_plan((320, 240), 3)``:
640x640, head on top (640x426), Left | Right below (320x214 each), no head crop, panels **stretched**
(ffmpeg ``scale=w:h``), not cropped. Serving must reproduce that, so the policy sees the same pixel
geometry it was trained on.

Recomposing the three camera frames of training episodes and comparing with the materialized canvas gives
48.1 dB mean PSNR with ``bilinear``, 45.6 dB ``bicubic``, 44.1 dB ``lanczos`` — hence the default below. The
residual is the x264 CRF 18 re-encode of the canvas clip.
"""

from __future__ import annotations

import numpy as np

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 640
# canvas_plan((320, 240), 3, canvas_profile="legacy_480") of the converter.
PANELS = ((0, 0, 640, 426), (0, 426, 320, 214), (320, 426, 320, 214))
LAYOUT = "head_top_others_bottom"


def _uint8_rgb(image: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"{name}: expected HxWx3 RGB, got {tuple(arr.shape)}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def stretch(image: np.ndarray, width: int, height: int, *, resample: str = "bilinear") -> np.ndarray:
    """Resize to the panel without preserving the aspect ratio — what ``scale=w:h`` does."""
    from PIL import Image

    if width <= 0 or height <= 0:
        raise ValueError("panel size must be positive")
    filters = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}
    if resample not in filters:
        raise ValueError(f"resample must be one of {sorted(filters)}")
    return np.asarray(Image.fromarray(image).resize((width, height), filters[resample]))


def compose_head_left_right(
    head: np.ndarray, left: np.ndarray, right: np.ndarray, *, resample: str = "bilinear"
) -> np.ndarray:
    """Head / Left | Right canvas, 640x640 uint8 RGB, from the three D435 views at their native size."""
    views = (_uint8_rgb(head, "head"), _uint8_rgb(left, "left_wrist"), _uint8_rgb(right, "right_wrist"))
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    for image, (x, y, width, height) in zip(views, PANELS):
        canvas[y : y + height, x : x + width] = stretch(image, width, height, resample=resample)
    return canvas
