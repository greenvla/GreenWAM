# SPDX-License-Identifier: OpenMDW-1.1
"""Video resize of a compiled sample: copy of `CanvasVideoResize` of the training pipeline.

A served sample carries no `greenwam_canvas` block (no canvas release), so the wrapped resize of the transform pipeline
runs; the canvas check is kept for samples that do carry one."""

from __future__ import annotations

import torch


class CanvasVideoResize:
    """Keep audited panel pixels exactly; legacy raw-camera samples still resize."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.pad_keys = fallback.pad_keys

    def __call__(self, sample, resolution):
        canvas = sample.get("greenwam_canvas")
        if canvas is None:
            return self.fallback(sample, resolution)
        width, height = canvas["width"], canvas["height"]
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("Canvas dimensions must be positive multiples of 32")
        for key in self.pad_keys:
            if key in sample and tuple(sample[key].shape[-2:]) != (height, width):
                raise ValueError(f"Canvas {key} geometry differs from frozen release")
        sample["image_size"] = torch.tensor([height, width, height, width])
        return sample
