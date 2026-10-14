# SPDX-License-Identifier: OpenMDW-1.1
"""User-supplied text timelines. No dataset, RGB or action access at inference."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from cosmos_framework.utils.rollout_partition import partition_block_rollout


def _bounds(start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("Prompt schedule requires integer source-frame [start,end) bounds")


def _text(text):
    # '' is the training text-dropout condition (CFG-unconditional); WorldArena's action-driven track has no text input
    if not isinstance(text, str) or (text and not text.strip()):
        raise ValueError("Prompt schedule requires rendered text, or an empty string for the text-dropout condition")


def plan_prompt_blocks(prompt, num_frames, *, mode, fps, block_frames=None):
    """Bind each actual VAE block to one target fragment, before any sampling.

    Timelines use episode source-frame indexes and contain already rendered
    text. Partition each covered atom separately with the existing 1+4k policy.
    Optional visual windows follow the training rule: use a window only if it
    contains the entire block; otherwise retain that atom's legacy visual text.
    Forward AR phases use their union over sampled target frames, retaining the
    full command if any target frame is uncovered. Phase overlaps are allowed.
    An explicit positive whole-block partition has no 17-frame/codec-alignment
    constraint. It is validated, never silently relabelled/split.
    """
    default = partition_block_rollout(num_frames)
    if isinstance(prompt, str):
        _text(prompt)
        lengths = tuple(block_frames) if block_frames is not None else default
        _validate_lengths(lengths, num_frames)
        return [
            dict(
                frames=n,
                text=prompt,
                atom_id=None,
                start_frame=None,
                end_frame=None,
                provenance={"source": "external_prompt"},
                binding="single_external_prompt",
                text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            )
            for n in lengths
        ]
    if not isinstance(prompt, Mapping) or prompt.get("schema_version") != 1 or prompt.get("mode") != mode:
        raise ValueError("Invalid prompt schedule schema or mode")
    if type(prompt.get("allow_demo_derived", False)) is not bool:
        raise ValueError("Prompt schedule allow_demo_derived must be boolean")
    start, end = prompt.get("start_frame"), prompt.get("end_frame")
    _bounds(start, end)
    clock = prompt.get("fps")
    if (
        isinstance(clock, bool)
        or not isinstance(clock, (int, float))
        or not math.isfinite(clock)
        or not math.isclose(clock, fps, abs_tol=1e-6, rel_tol=0)
    ):
        raise ValueError("Prompt schedule FPS differs from rollout")
    if (
        not all(isinstance(prompt.get(k), str) and prompt[k] for k in ("root", "camera"))
        or type(prompt.get("episode_index")) is not int
        or prompt["episode_index"] < 0
    ):
        raise ValueError("Prompt schedule requires one source episode and camera")
    step = prompt.get("source_frame_stride", 1)
    if type(step) is not int or step < 1 or (end - start - 1) % step:
        raise ValueError("Prompt schedule requires a consistent source frame stride")
    if num_frames > (end - start - 1) // step + 1:
        raise ValueError("Prompt schedule does not cover the requested horizon")
    spans = prompt.get("spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("Prompt schedule requires target spans")
    previous, ids = None, set()
    for span in spans:
        if not isinstance(span, Mapping):
            raise ValueError("Prompt spans must be mappings")
        a, b = span.get("start_frame"), span.get("end_frame")
        _bounds(a, b)
        _text(span.get("text"))
        atom = span.get("atom_id")
        if not isinstance(atom, str) or not atom or atom in ids:
            raise ValueError("Prompt spans require unique atom IDs")
        ids.add(atom)
        if previous is not None and a != previous:
            raise ValueError("Prompt schedule has a gap or overlap in the admitted frames")
        previous = b
        if not isinstance(span.get("provenance"), Mapping):
            raise ValueError("Prompt span requires provenance")
        if span.get("text_provenance") not in {None, "source_task", "demo_derived"}:
            raise ValueError("Invalid prompt text provenance")
        if span.get("text_provenance") == "demo_derived" and not prompt.get("allow_demo_derived", False):
            raise ValueError("demo_derived full command requires explicit admission")
        windows = span.get("windows", [])
        if not isinstance(windows, list) or (windows and mode != "i2v_ar"):
            raise ValueError("Only visual schedules support windows")
        window_end = a
        for window in windows:
            if not isinstance(window, Mapping):
                raise ValueError("Prompt windows must be mappings")
            x, y = window.get("start_frame"), window.get("end_frame")
            _bounds(x, y)
            _text(window.get("text"))
            if not window_end <= x < y <= b:
                raise ValueError("Prompt windows overlap or leave their atom")
            window_end = y
        if "action_windows" in span:
            from greenwam.data.prompt_contract import validate_action_windows

            if mode != "forward_ar" or not isinstance(span.get("action_layout_caption", ""), str):
                raise ValueError("Only forward_ar schedules support action_windows")
            validate_action_windows(
                dict(action_windows=span["action_windows"], forward=span["text"]),
                atom_id=atom,
                atom_start_frame=a,
                atom_end_frame=b,
            )
    if (
        not spans[0]["start_frame"] <= start < spans[0]["end_frame"]
        or not spans[-1]["start_frame"] < end <= spans[-1]["end_frame"]
    ):
        raise ValueError("Prompt schedule endpoints must be covered by admitted frames")
    if block_frames is None:
        lengths = []
        for span in spans:
            first = max(0, (span["start_frame"] - start + step - 1) // step)
            stop = min(num_frames, (span["end_frame"] - start + step - 1) // step)
            if first < stop:
                lengths.extend(partition_block_rollout(stop - first))
    else:
        lengths = tuple(block_frames)
    _validate_lengths(lengths, num_frames)
    blocks, cursor, index = [], start, 0
    for n in lengths:
        while spans[index]["end_frame"] <= cursor:
            index += 1
        span = spans[index]
        stop = cursor + (n - 1) * step + 1
        if stop > span["end_frame"]:
            raise ValueError("Explicit rollout block crosses a target admitted fragment")
        text, binding = span["text"], "atom"
        phase_audit = dict(fallback_reason="", action_window_source_indexes="")
        for window in span.get("windows", []):
            if window["start_frame"] <= cursor and stop <= window["end_frame"]:
                text, binding = window["text"], "visual_windows"
                break
        if mode == "forward_ar":
            from greenwam.data.prompt_contract import select_action_window_text

            windows = span.get("action_windows", [])
            disabled = bool(windows) and not prompt.get("allow_demo_derived", False)
            text, phase_audit = select_action_window_text(
                [] if disabled else windows, list(range(cursor, stop, step)), text
            )
            if disabled:
                phase_audit["fallback_reason"] = "demo_derived_disabled"
            if not phase_audit["fallback_reason"]:
                from cosmos_framework.data.generator.action.utils.transforms import add_action_mode_metadata

                if isinstance(span.get("structured_prompt"), Mapping):
                    # JSON prompts (prompt_format=json): the phase prose replaces the task description, everything
                    # else (framing, robot, horizon) stays as in the training prompt of this layout
                    from greenwam.data.prompt_json import prompt_text, render_structured_prompt

                    text = prompt_text(
                        add_action_mode_metadata(
                            render_structured_prompt(text=text, **span["structured_prompt"]), "forward_dynamics"
                        )
                    )
                else:
                    layout = span.get("action_layout_caption", "")
                    text = add_action_mode_metadata(f"{layout} {text}" if layout else text, "forward_dynamics")
                binding = "action_windows"
        provenance = dict(span["provenance"])
        if binding == "action_windows":
            provenance.update(
                selected_text_provenance="demo_derived",
                action_window_source_indexes=phase_audit["action_window_source_indexes"],
            )
        blocks.append(
            dict(
                frames=n,
                text=text,
                atom_id=span["atom_id"],
                start_frame=cursor,
                end_frame=stop,
                provenance=provenance,
                binding=binding,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                **phase_audit,
            )
        )
        cursor += n * step
    return blocks


def _validate_lengths(lengths, total):
    if not lengths or any(type(n) is not int or n < 1 for n in lengths) or sum(lengths) != total:
        raise ValueError("Prompt blocks must exactly cover the horizon with positive whole-block lengths")
