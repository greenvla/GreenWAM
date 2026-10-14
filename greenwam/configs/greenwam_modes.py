# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit mode and prompt contracts for GreenWAM midtraining.

These functions have no model/runtime imports, so admission and visibility can
be checked before a CUDA worker or checkpoint is created.
"""

import math
from collections.abc import Mapping

MODES = ("t2v", "i2v", "i2v_ar", "forward", "forward_ar", "inverse", "wam", "policy")
VISUAL_MODES = frozenset({"t2v", "i2v", "i2v_ar"})
AR_MODES = frozenset({"i2v_ar", "forward_ar"})
# Starter sample shares, not an empirically optimized or compute-matched mixture.
DEFAULT_MODE_WEIGHTS = {
    "t2v": 20.0,
    "i2v": 10.0,
    "i2v_ar": 10.0,
    "forward": 15.0,
    "forward_ar": 20.0,
    "inverse": 10.0,
    "wam": 15.0,
}
INVERSE_PROMPT = "Infer the absolute joint states throughout the observed video, including the initial state."
INVERSE_EEF_PROMPT = (
    "Infer the absolute end-effector positions and orientations in each effector's "
    "source reference frame, and measured hand/gripper states throughout the "
    "observed video, including the initial state."
)
UPSTREAM_MODES = {
    "i2v_ar": "i2v",
    "forward": "forward_dynamics",
    "forward_ar": "forward_dynamics",
    "inverse": "inverse_dynamics",
    "wam": "wam",
    "policy": "policy",
}


def training_horizon(action_fps, horizon):
    """Current training positions encode one sampled-action-step lookahead."""
    if not math.isfinite(action_fps) or action_fps <= 0:
        raise ValueError("Action FPS must be finite and positive")
    expected = 1.0 / action_fps
    if horizon is not None and not math.isclose(float(horizon), expected, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            "Training action alignment requires future_horizon_s=1/f_a; other horizons need explicit RoPE support"
        )
    return expected


def resolve_mode_weights(weights=None, *, forced_mode=None):
    """Return positive sample shares; readers compile each mode independently."""
    if forced_mode is not None:
        if forced_mode not in MODES:
            raise ValueError(f"Unknown GreenWAM forced mode: {forced_mode!r}")
        return {forced_mode: 1.0}
    values = DEFAULT_MODE_WEIGHTS if weights is None else weights
    if not isinstance(values, Mapping):
        raise ValueError("Mode weights must be a mapping")
    out = {}
    for mode, weight in values.items():
        if mode not in MODES:
            raise ValueError(f"Unknown GreenWAM mode: {mode!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid weight for {mode}: {weight!r}")
        if weight > 0:
            out[mode] = float(weight)
    if not out:
        raise ValueError("At least one GreenWAM mode must have a positive weight")
    return out


def model_mode_settings(weights):
    resolve_mode_weights(weights)
    return {
        "video_temporal_causal": False,
        "causal_training_strategy": "none",
        "joint_attn_implementation": "two_way",
    }


def mode_spec(mode, *, video_length, action_length=0, temporal_downsample=4):
    """SequencePlan kwargs for state-future targets (no prepended target state)."""
    if mode not in MODES:
        raise ValueError(f"Unknown GreenWAM mode: {mode!r}")
    if video_length < 1 or temporal_downsample < 1:
        raise ValueError("Video length and temporal downsample must be positive")
    has_action = mode not in VISUAL_MODES
    if has_action and action_length < 1:
        raise ValueError("Action modes require future-state targets")
    if mode == "forward_ar" and action_length != video_length:
        raise ValueError("forward_ar needs one measured-state action for every future RGB frame at equal rates")
    if mode == "t2v":
        vision = []
    elif mode == "inverse":
        vision = list(range(1 + math.ceil((video_length - 1) / temporal_downsample)))
    elif mode in AR_MODES:
        vision = [0, 1, 2, 3]
    else:
        vision = [0]
    return {
        "has_text": True,
        "has_vision": True,
        "has_action": has_action,
        "condition_frame_indexes_vision": vision,
        "condition_frame_indexes_action": list(range(action_length)) if mode in {"forward", "forward_ar"} else [],
        "action_start_frame_offset": 0 if mode == "forward_ar" else 1,
        **({"vision_prefix_frame_count": 4} if mode in AR_MODES else {}),
    }


def is_null_instruction(text):
    """A literal serialized null is missing task text, not a prospective command."""
    return isinstance(text, str) and text.strip().casefold() == "null"


def _prompt(sample, mode, allow_demo_derived):
    if mode == "inverse":
        if sample.get("greenwam_action_representation") == "eef" and sample.get("eef_encoding") == "relative":
            raise ValueError("EEF inverse requires absolute pose targets")
        return (INVERSE_EEF_PROMPT if sample.get("greenwam_action_representation") == "eef" else INVERSE_PROMPT), None
    prompts = sample.get("greenwam_prompts")
    if not isinstance(prompts, Mapping):
        raise ValueError(f"GreenWAM {mode} requires a versioned mode-specific prompt record")
    if mode not in {"policy", "wam"}:
        key = "visual" if mode in VISUAL_MODES else "forward"
        text = prompts.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"GreenWAM {mode} requires a nonempty {key} prompt")
        provenance = None
        if mode in {"forward", "forward_ar"}:
            if is_null_instruction(text):
                raise ValueError("Literal null is not an action instruction")
            source = prompts.get("instruction_source", {})
            if not isinstance(source, Mapping):
                raise ValueError("instruction_source must be a mapping")
            provenance = source.get("provenance")
            if provenance is not None and provenance not in {"source_task", "demo_derived"}:
                raise ValueError("Forward instruction requires source_task or demo_derived provenance")
            if provenance == "demo_derived" and not allow_demo_derived:
                raise ValueError("demo_derived forward prompts require their explicit mixing control")
        return text.strip(), provenance
    policy = prompts.get(mode, prompts.get("policy"))
    if not isinstance(policy, Mapping) or policy.get("available_before_rollout") is not True:
        raise ValueError(f"GreenWAM {mode} requires an explicitly prospective instruction")
    provenance = policy.get("provenance")
    if provenance not in {"source_task", "demo_derived"}:
        raise ValueError(f"GreenWAM prospective {mode} prompt requires source_task or demo_derived provenance")
    if provenance == "demo_derived" and not allow_demo_derived:
        raise ValueError("demo_derived policy prompts require their explicit mixing control")
    text = policy.get("text")
    if is_null_instruction(text):
        raise ValueError("Literal null is not an action instruction")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"GreenWAM {mode} requires a nonempty prospective instruction")
    return text.strip(), provenance


def prepare_mode_sample(sample, mode, *, allow_demo_derived=False):
    """Remove target-only context before text tokenization and VAE encoding.

    Prompt provenance is a checked upstream contract, not a claim that arbitrary
    retrospective prose can be safely rewritten using a keyword blacklist.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown GreenWAM mode: {mode!r}")
    out = dict(sample)
    text, provenance = _prompt(sample, mode, allow_demo_derived)
    from greenwam.data.prompt_contract import bind_prompt

    binding_sample = sample
    disabled_phases = (
        mode == "forward_ar"
        and not allow_demo_derived
        and bool(sample.get("greenwam_prompts", {}).get("action_windows"))
    )
    if disabled_phases:
        binding_sample = dict(sample, greenwam_prompts=dict(sample["greenwam_prompts"], action_windows=[]))
    text, out["greenwam_prompt"] = bind_prompt(binding_sample, mode, text, provenance, visual=mode in VISUAL_MODES)
    if disabled_phases:
        out["greenwam_prompt"]["fallback_reason"] = "demo_derived_disabled"
    provenance = out["greenwam_prompt"]["provenance"] or provenance
    out["ai_caption"] = text
    out["mode"] = UPSTREAM_MODES.get(mode, mode)
    out["greenwam_mode"] = mode
    if mode in AR_MODES:
        memory = out.get("vision_pixel_prefixes")
        if memory is None or memory.ndim != 4 or memory.shape[0] != 3 or memory.shape[1] != 4:
            raise ValueError(f"{mode} requires exactly four RGB history anchors as [C,4,H,W]")
    else:
        out.pop("vision_pixel_prefixes", None)
    for key in (
        "greenwam_prompts",
        "source_task",
        "canonical_caption",
        "action_description",
        "relabel_prompt",
        "idle_frames",
        "idle_frames_total",
        "action_caption_attributes",
    ):
        out.pop(key, None)
    if mode in VISUAL_MODES:
        for key in tuple(out):
            if key.startswith("action") or key in {
                "raw_action_dim",
                "normalized_action",
                "history_action",
                "history_normalized_action",
                "conditioning_fps_action",
                "pseudo_action",
            }:
                out.pop(key, None)
    if provenance is not None:
        out[f"{mode}_provenance"] = provenance
    if mode == "policy":
        out["video"] = out["video"][:, :1]
        # This scalar drives VAE/plan inference in the upstream transform; a
        # future logical length here would reintroduce future vision positions.
        out.pop("video_num_frames", None)
    return out
