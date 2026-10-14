# SPDX-License-Identifier: OpenMDW-1.1
"""Physical-time GreenWAM rollout budgets and exact block rollout block partitioning.

A reference budget B denotes B/30 seconds of future video. Effective FPS is
selected upstream (native rate capped at 30). Camera and canvas dimensions are
retained as provenance and compute metadata, never as a duration multiplier.
Budgets exclude the initial observation and four reused anchors per block.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from numbers import Integral

ROLLOUT_REFERENCE_FPS = 30
ROLLOUT_REFERENCE_FRAMES = (400, 600, 800, 900)
ROLLOUT_REFERENCE_HEIGHT = 448
ROLLOUT_REFERENCE_WIDTH = 448
ROLLOUT_SOURCE_GEOMETRY_POLICY = "native_camera_frame_v1"
# Historical wire fields named rollout_geometry_policy carry this budget identity.
ROLLOUT_BUDGET_POLICY = "physical_time_v2"


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def scaled_rollout_frames(
    reference_frames: int,
    *,
    fps: float,
    height: int,
    width: int,
    reference_height: int = ROLLOUT_REFERENCE_HEIGHT,
    reference_width: int = ROLLOUT_REFERENCE_WIDTH,
) -> int:
    """Floor(B * effective_fps / 30), independent of any image dimensions.

    Geometry arguments remain validated for existing callers and provenance.
    The caller chooses effective FPS; this helper never resamples a source.
    A sub-frame budget returns zero; rounding up would exceed the ceiling.
    """
    budget = _positive_int(reference_frames, "reference_frames")
    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    reference_height = _positive_int(reference_height, "reference_height")
    reference_width = _positive_int(reference_width, "reference_width")
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    # Decimal FPS metadata should not floor an exact integer one frame too low
    # because binary floating-point multiplication produced n - epsilon.
    frames = Fraction(budget, ROLLOUT_REFERENCE_FPS)
    frames *= Fraction(str(fps))
    return frames.numerator // frames.denominator


def rollout_training_budgets(
    *,
    fps: float,
    height: int,
    width: int,
    reference_frames: Iterable[int] = ROLLOUT_REFERENCE_FRAMES,
    reference_height: int = ROLLOUT_REFERENCE_HEIGHT,
    reference_width: int = ROLLOUT_REFERENCE_WIDTH,
) -> tuple[int, ...]:
    """Sorted unique positive scaled curriculum lengths; empty if none fits."""
    budgets = {
        scaled_rollout_frames(
            frames,
            fps=fps,
            height=height,
            width=width,
            reference_height=reference_height,
            reference_width=reference_width,
        )
        for frames in reference_frames
    }
    return tuple(sorted(budget for budget in budgets if budget > 0))


def partition_block_rollout(total_frames: int, max_block_frames: int = 17) -> tuple[int, ...]:
    """Exactly partition future frames into the fewest positive 1-mod-4 blocks.

    A partition into n blocks requires n >= ceil(total/max) and n == total
    (mod 4). The smallest such n is feasible because max itself is 1 mod 4.
    Starting each block at one and assigning available groups of four proves
    sufficiency. Larger blocks come first; no frame is padded or duplicated.
    For 900 frames and H17 the result is 52*17 + 13 + 1 + 1 + 1.
    """
    total = _positive_int(total_frames, "total_frames")
    maximum = _positive_int(max_block_frames, "max_block_frames")
    if maximum % 4 != 1:
        raise ValueError("max_block_frames must equal 1 + 4*k")
    minimum = (total + maximum - 1) // maximum
    count = minimum + (total - minimum) % 4
    groups_left = (total - count) // 4
    groups_per_block = (maximum - 1) // 4
    blocks = []
    for _ in range(count):
        groups = min(groups_left, groups_per_block)
        blocks.append(1 + 4 * groups)
        groups_left -= groups
    return tuple(blocks)


from dataclasses import dataclass


@dataclass(frozen=True)
class WholeAtomBlock:
    atom_id: str
    start_frame: int
    end_frame: int
    target_frame_indexes: tuple[int, ...]
    memory_frame_indexes: tuple[int, int, int, int]
    segment_index: int
    seed_frame: int | None
    budget_frames: int
    overlong: bool

    @property
    def target_frames(self):
        return len(self.target_frame_indexes)


@dataclass(frozen=True)
class WholeAtomRolloutPlan:
    blocks: tuple[WholeAtomBlock, ...]
    bootstrap_only_atom_ids: tuple[str, ...]

    @property
    def overlong(self):
        return tuple(block for block in self.blocks if block.overlong)


def plan_whole_atom_rollouts(atoms, *, source_fps, target_fps, budget_frames):
    """Partition contiguous admitted frames into complete target atoms, never clip a block.

    Input atoms expose atom_id/start_frame/end_frame/root/episode_index and
    admission_camera as attributes or mapping keys. Each atom keeps its own
    nearest-row clock. The first frame after an episode/camera/admission gap is the
    explicit clean bootstrap seed; all remaining measured frames are targets.
    A single-frame bootstrap is reported separately. Overlong atoms remain in
    the plan, marked explicitly, so a census can gate execution without loss.
    ``budget_frames`` is already scaled to effective FPS by ``scaled_rollout_frames``.
    """
    budget = _positive_int(budget_frames, "budget_frames")
    if not (math.isfinite(source_fps) and math.isfinite(target_fps) and 0 < target_fps <= source_fps):
        raise ValueError("Whole atom FPS must be positive, finite and at most native FPS")

    def field(atom, key):
        return atom[key] if isinstance(atom, dict) else getattr(atom, key)

    groups = {}
    seen = set()
    for atom in atoms:
        atom_id = field(atom, "atom_id")
        if atom_id in seen:
            raise ValueError(f"Duplicate whole atom id {atom_id}")
        seen.add(atom_id)
        start, end = field(atom, "start_frame"), field(atom, "end_frame")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Integral)
            or not isinstance(end, Integral)
            or not 0 <= start < end
        ):
            raise ValueError(f"Invalid whole atom bounds: {atom_id}")
        key = tuple(field(atom, name) for name in ("root", "episode_index", "admission_camera"))
        groups.setdefault(key, []).append(atom)
    blocks, bootstrap_only = [], []
    segment = -1
    step = source_fps / target_fps
    for _, group in sorted(groups.items()):
        previous_end, history, used = None, [], 0
        for atom in sorted(group, key=lambda a: (field(a, "start_frame"), field(a, "end_frame"))):
            start, end, atom_id = (field(atom, key) for key in ("start_frame", "end_frame", "atom_id"))
            if previous_end is not None and start < previous_end:
                raise ValueError(f"Overlapping whole admitted atoms: {atom_id}")
            frames = tuple(
                start + math.floor(i * step + 0.5 + 1e-9)
                for i in range(math.ceil((end - start) / step))
                if start + math.floor(i * step + 0.5 + 1e-9) < end
            )
            seed = None
            if start != previous_end:
                segment += 1
                history, used = [frames[0]], 0
                seed, frames = frames[0], frames[1:]
            elif used and used + len(frames) > budget:
                segment += 1
                history, used = [history[-1]], 0
            previous_end = end
            if not frames:
                bootstrap_only.append(atom_id)
                continue
            memory = tuple(history[i * (len(history) - 1) // 3] for i in range(4))
            blocks.append(
                WholeAtomBlock(atom_id, start, end, frames, memory, segment, seed, budget, len(frames) > budget)
            )
            history.extend(frames)
            used += len(frames)
    return WholeAtomRolloutPlan(tuple(blocks), tuple(bootstrap_only))
