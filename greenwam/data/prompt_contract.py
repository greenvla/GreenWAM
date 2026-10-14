# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit visual-caption scope; source frames are episode-relative, [start,end).

Legacy ``visual`` describes the admitted atom. Optional ``visual_windows`` may
override it only when one binding covers every sampled target RGB frame. AR
history is conditioning and does not expand the target caption's scope.
"""

import hashlib
import math
from collections.abc import Mapping

DEFAULT_MAX_TEXT_TOKENS = 512


def validate_max_text_tokens(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ValueError("max_text_tokens must be an integer in [1, 4096]")
    return value


def validate_visual_windows(prompts, *, atom_id="unknown", atom_start_frame=None, atom_end_frame=None):
    windows = prompts.get("visual_windows", [])
    if not isinstance(windows, list):
        raise ValueError(f"Prompt {atom_id}: visual_windows must be a list")
    bounded = atom_start_frame is not None or atom_end_frame is not None
    if bounded and (
        type(atom_start_frame) is not int
        or type(atom_end_frame) is not int
        or not 0 <= atom_start_frame < atom_end_frame
    ):
        raise ValueError(f"Prompt {atom_id}: visual_windows need valid source atom bounds")
    previous_end = -1
    for window in windows:
        if not isinstance(window, Mapping):
            raise ValueError(f"Prompt {atom_id}: visual_windows entries must be mappings")
        start, end, text = window.get("start_frame"), window.get("end_frame"), window.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(f"Prompt {atom_id}: visual_windows need valid [start_frame,end_frame) and nonempty text")
        if start < previous_end:
            raise ValueError(f"Prompt {atom_id}: visual_windows must be sorted and non-overlapping")
        if bounded and not atom_start_frame <= start < end <= atom_end_frame:
            raise ValueError(f"Prompt {atom_id}: visual_windows must remain inside their source atom")
        previous_end = end
    if windows and (not isinstance(prompts.get("visual"), str) or not prompts["visual"].strip()):
        raise ValueError(f"Prompt {atom_id}: visual_windows require a legacy visual fallback")
    return windows


def validate_action_windows(prompts, *, atom_id="unknown", atom_start_frame=None, atom_end_frame=None):
    """Validate supplied phase timings; overlap and source order are intentional."""
    windows = prompts.get("action_windows", [])
    if not isinstance(windows, list):
        raise ValueError(f"Prompt {atom_id}: action_windows must be a list")
    bounded = atom_start_frame is not None or atom_end_frame is not None
    if bounded and (
        type(atom_start_frame) is not int
        or type(atom_end_frame) is not int
        or not 0 <= atom_start_frame < atom_end_frame
    ):
        raise ValueError(f"Prompt {atom_id}: action_windows need valid source atom bounds")
    indexes = set()
    for window in windows:
        if not isinstance(window, Mapping):
            raise ValueError(f"Prompt {atom_id}: action_windows entries must be mappings")
        start, end, text = window.get("start_frame"), window.get("end_frame"), window.get("text")
        index = window.get("source_index")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or not isinstance(text, str)
            or not text.strip()
            or type(index) is not int
            or index < 0
            or index in indexes
            or window.get("provenance") != "demo_derived"
        ):
            raise ValueError(
                f"Prompt {atom_id}: action_windows need valid bounds, text, unique source_index and demo_derived provenance"
            )
        if bounded and not atom_start_frame <= start < end <= atom_end_frame:
            raise ValueError(f"Prompt {atom_id}: action_windows must remain inside their source atom")
        indexes.add(index)
    if windows and (not isinstance(prompts.get("forward"), str) or not prompts["forward"].strip()):
        raise ValueError(f"Prompt {atom_id}: action_windows require a full forward fallback")
    return windows


def select_action_window_text(windows, frames, fallback):
    """Select validated phases only when their union covers every target frame."""
    audit = dict(fallback_reason="", action_window_source_indexes="")
    if not windows:
        return fallback, dict(audit, fallback_reason="no_action_windows")
    if frames is None:
        return fallback, dict(audit, fallback_reason="missing_target_frame_indexes")
    selected = [w for w in windows if any(w["start_frame"] <= f < w["end_frame"] for f in frames)]
    if not frames or any(not any(w["start_frame"] <= f < w["end_frame"] for w in selected) for f in frames):
        return fallback, dict(audit, fallback_reason="uncovered_target_frames")
    audit["action_window_source_indexes"] = ",".join(str(w["source_index"]) for w in selected)
    return " ".join(w["text"].strip() for w in selected), audit


def bind_prompt(sample, mode, text, provenance, *, visual=False):
    """Return selected prose and a collatable audit record; never infer prose."""
    audit = dict(
        schema_version=1,
        scope="atom",
        binding=mode,
        provenance=provenance or "",
        start_frame=-1,
        end_frame=-1,
        sample_start_frame=-1,
        sample_end_frame=-1,
        atom_start_frame=-1,
        atom_end_frame=-1,
        source_fps=-1.0,
        instruction_source_route="",
        instruction_source_field="",
        instruction_source_sha256="",
        fallback_reason="",
        action_window_source_indexes="",
        timebase="episode_source_frames",
        temporal_relation="unknown",
    )
    indexes = sample.get("vision_target_frame_indexes")
    if indexes is not None:
        frames = indexes.tolist() if hasattr(indexes, "tolist") else list(indexes)
        if (
            not frames
            or any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames)
            or any(a >= b for a, b in zip(frames, frames[1:]))
        ):
            raise ValueError(f"Prompt {sample.get('atom_id', 'unknown')}: invalid source target frame indexes")
        audit.update(sample_start_frame=frames[0], sample_end_frame=frames[-1] + 1)
    if visual:
        audit["binding"] = "legacy_visual"
        start, end = sample.get("greenwam_atom_start_frame"), sample.get("greenwam_atom_end_frame")
        windows = validate_visual_windows(
            sample["greenwam_prompts"],
            atom_id=sample.get("atom_id", "unknown"),
            atom_start_frame=start,
            atom_end_frame=end,
        )
        if start is not None:
            source_fps = sample.get("greenwam_source_fps")
            if (
                isinstance(source_fps, bool)
                or not isinstance(source_fps, (int, float))
                or not math.isfinite(source_fps)
                or source_fps <= 0
            ):
                raise ValueError("Visual prompt source atom requires its native FPS")
            audit.update(
                atom_start_frame=start,
                atom_end_frame=end,
                start_frame=start,
                end_frame=end,
                source_fps=float(source_fps),
            )
            if indexes is not None and not start <= frames[0] <= frames[-1] < end:
                raise ValueError("Visual prompt targets must remain inside their source atom")
        if indexes is not None:
            for window in windows:
                if window["start_frame"] <= frames[0] and frames[-1] < window["end_frame"]:
                    text = window["text"].strip()
                    audit.update(
                        scope="window",
                        binding="visual_windows",
                        start_frame=window["start_frame"],
                        end_frame=window["end_frame"],
                    )
                    break
        elif windows:
            audit["binding"] = "legacy_visual_missing_frame_indexes"
        if indexes is not None and audit["start_frame"] >= 0:
            audit["temporal_relation"] = (
                "equal_bounds"
                if (frames[0], frames[-1] + 1) == (audit["start_frame"], audit["end_frame"])
                else "contained_subwindow"
            )
    elif mode == "inverse":
        audit.update(scope="generic_inverse", binding="safe_representation_instruction")
    elif mode in {"forward", "forward_ar", "wam"}:
        # This is the admitted command's applicability to the admitted data, not a claim that
        # an original episode goal was a subtask or was inferred from actions[].
        start, end = sample.get("greenwam_atom_start_frame"), sample.get("greenwam_atom_end_frame")
        if start is not None or end is not None:
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError("Command prompt requires valid source atom bounds")
            fps = sample.get("greenwam_source_fps")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Command prompt source atom requires its native FPS")
            audit.update(
                atom_start_frame=start, atom_end_frame=end, start_frame=start, end_frame=end, source_fps=float(fps)
            )
            if indexes is not None:
                if not start <= frames[0] <= frames[-1] < end:
                    raise ValueError("Command prompt targets must remain inside their source atom")
                audit["temporal_relation"] = (
                    "equal_bounds" if (frames[0], frames[-1] + 1) == (start, end) else "contained_subwindow"
                )
        source = sample.get("greenwam_prompts", {}).get("instruction_source", {})
        for key, field in (("route", "route"), ("field", "field"), ("sha256", "record_sha256")):
            audit["instruction_source_" + key] = source.get(field, "")
        if mode == "forward_ar":
            windows = validate_action_windows(
                sample.get("greenwam_prompts", {}),
                atom_id=sample.get("atom_id", "unknown"),
                atom_start_frame=start,
                atom_end_frame=end,
            )
            if windows and start is None:
                raise ValueError("action_windows require source atom bounds at binding")
            text, phase_audit = select_action_window_text(windows, frames if indexes is not None else None, text)
            audit.update(phase_audit)
            if not phase_audit["fallback_reason"]:
                audit.update(
                    scope="window",
                    binding="action_windows",
                    provenance="demo_derived",
                    start_frame=frames[0],
                    end_frame=frames[-1] + 1,
                    temporal_relation="sampled_target_coverage",
                )
    audit["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, audit
