# SPDX-License-Identifier: OpenMDW-1.1
"""Exact block rollout visual or action-conditioned rollout using OmniMoTModel's sampling API.

Pass an already loaded block-rollout-enabled model and shared-projector actions (not raw
controller commands). Only the initial RGB observation is accepted: subsequent
memory can therefore only come from decoded generated frames. Images must be
prepared at the model's output resolution before calling this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from cosmos_framework.utils.prompt_schedule import plan_prompt_blocks
from cosmos_framework.utils.rollout_partition import partition_block_rollout, scaled_rollout_frames


@dataclass
class RolloutActionChunk:
    """Future-only model-space actions, one per RGB frame on the same clock.

    For delta joints, q0_codec retains the supplied physical boundary and the
    reversible future normalizer. The clean q0 token is added only to a batch.
    """

    action: torch.Tensor
    action_contract_id: int
    action_valid_mask: torch.Tensor
    action_availability_mask: torch.Tensor
    action_layout_id: int | None = None
    raw_action_dim: int | None = None
    domain_id: int | None = None
    fps: float | None = None
    action_processing_record: Any = None
    q0_codec: Any = None
    q0_timestamp: float = 0.0
    target_timestamps: torch.Tensor | None = None

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any]) -> RolloutActionChunk:
        """Extract already model-space conditioning, excluding all GT RGB.

        A compiled training sample is already normalized. For physical reader
        targets and trained statistics use from_physical_sample explicitly.
        """
        encoding = sample.get("greenwam_joint_encoding", "absolute")
        if encoding not in {"absolute", "delta_from_q0"}:
            raise ValueError("unknown block rollout joint encoding")
        record = sample.get("action_processing_record")
        q0_codec = None
        action, valid = sample["action"], sample["action_valid_mask"]
        if encoding == "delta_from_q0":
            from greenwam.data.joint_deltas import Q0_CONDITIONING_SCHEMA, Q0_STATE_CODEC, Q0PrefixActionCodec

            if (
                sample.get("greenwam_action_representation") != "joint"
                or sample.get("greenwam_state_conditioning") != Q0_CONDITIONING_SCHEMA
                or sample.get("greenwam_q0_codec") != Q0_STATE_CODEC
            ):
                raise ValueError("delta block rollout requires the joint q0 conditioning contract")
            if record is None or not isinstance(record.action_codec, Q0PrefixActionCodec):
                raise ValueError("model-space delta actions require the explicit q0 prefix codec")
            q0_codec = record.action_codec
            if action.shape[0] != q0_codec.future_valid_mask.shape[0] + 1:
                raise ValueError("delta model sample must contain exactly one q0 prefix")
            prefix_valid = valid[0, : record.raw_action_dim]
            prefix = action[0, : record.raw_action_dim]
            if not torch.equal(prefix_valid, q0_codec.q0_valid_mask) or not torch.equal(
                prefix[prefix_valid], q0_codec.q0.to(prefix)[prefix_valid]
            ):
                raise ValueError("compiled q0 prefix differs from its physical state codec")
            action, valid = action[1:, : record.raw_action_dim], valid[1:, : record.raw_action_dim]
        chunk = cls(
            action=action,
            action_contract_id=int(sample["action_contract_id"]),
            action_layout_id=int(sample["action_layout_id"]) if "action_layout_id" in sample else None,
            action_valid_mask=valid,
            action_availability_mask=sample["action_availability_mask"],
            raw_action_dim=int(sample.get("raw_action_dim", sample["action"].shape[-1])),
            domain_id=int(sample["domain_id"]) if "domain_id" in sample else None,
            fps=float(sample["conditioning_fps_action"]) if "conditioning_fps_action" in sample else None,
            action_processing_record=record,
            q0_codec=q0_codec,
            q0_timestamp=float(sample.get("action_q0_timestamp", 0.0)),
            target_timestamps=sample.get("action_target_timestamps"),
        )
        if q0_codec is not None:
            _validate_delta_chunk(chunk, fps=chunk.fps)
        return chunk

    @classmethod
    def from_physical_sample(cls, sample: Mapping[str, Any], *, action_statistics: Any) -> RolloutActionChunk:
        """Normalize adapted physical actions, including unnormalized q0 deltas."""
        from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessor

        if sample.get("action_processing_record") is not None:
            raise ValueError("actions are already processed; use from_sample to avoid double normalization")
        if sample.get("greenwam_joint_encoding", "absolute") == "delta_from_q0":
            from greenwam.data.joint_deltas import prepare_q0_prefix

            raw, encoded, valid, codec = prepare_q0_prefix(sample, normalizer=action_statistics.for_sample(sample))
            processed = ActionProcessor(raw.shape[-1]).preprocess_action(
                dict(sample),
                raw,
                action_normalizer=None,
                action_valid_mask=valid,
                normalized_action=encoded,
                action_codec=codec,
            )
            return cls.from_sample(processed)
        if sample.get("greenwam_joint_encoding", "absolute") != "absolute":
            raise ValueError("unknown block rollout joint encoding")
        processed = ActionProcessor(sample["action"].shape[-1]).preprocess_action(
            dict(sample),
            sample["action"],
            action_normalizer=action_statistics.for_sample(sample),
            action_valid_mask=sample["action_valid_mask"],
        )
        return cls.from_sample(processed)


@dataclass(frozen=True)
class RolloutChunkRecord:
    memory_frame_indexes: list[int]
    history_length_before: int
    condition_frame_indexes: list[int]
    generated_frames: int
    history_length_after: int
    action_contract_id: int | None
    memory_frame_sha256: list[str] = field(default_factory=list)
    generated_frame_sha256: list[str] = field(default_factory=list)
    action_q0_frame_index: int | None = None
    action_q0_timestamp: float | None = None
    action_q0_source: str | None = None
    prompt_text: str = ""
    prompt_text_sha256: str = ""
    prompt_atom_id: str | None = None
    prompt_start_frame: int | None = None
    prompt_end_frame: int | None = None
    prompt_binding: str = "single_external_prompt"
    prompt_provenance: dict = field(default_factory=dict)
    prompt_token_count: int | None = None
    codec_padding_frames: int = 0


@dataclass
class BlockRollout:
    video: torch.Tensor  # [1,C,Tfuture,H,W], normalized float; excludes initial observation
    history_rgb: torch.Tensor  # [C,1+Tfuture,H,W], uint8; unique timeline including initial observation
    chunks: list[RolloutChunkRecord]


def rollout_memory_indexes(history_length: int) -> list[int]:
    """Exact floor(linspace(0, history_length-1, 4)), without float rounding."""
    if history_length < 1:
        raise ValueError("Block rollout memory requires at least the real initial observation")
    return [i * (history_length - 1) // 3 for i in range(4)]


def _rgb_frame_hashes(rgb: torch.Tensor) -> list[str]:
    """Hash the actual uint8 CTHW inputs as C-contiguous HWC frame bytes."""
    return [
        hashlib.sha256(frame.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).hexdigest()
        for frame in rgb.unbind(dim=1)
    ]


def _validate_rollout_fps(fps: float) -> None:
    if isinstance(fps, bool) or not math.isfinite(fps) or not 0 < fps <= 30:
        raise ValueError(
            "GreenWAM Edge FPS must be finite, positive and at most 30; resample faster sources before inference"
        )


def _validate_delta_chunk(chunk, *, fps):
    """Keep the q0 codec, temporal masks and physical clock bound together."""
    codec = chunk.q0_codec
    if codec is None:
        return
    record = chunk.action_processing_record
    if record is None or record.action_codec is not codec:
        raise ValueError("delta q0 codec differs from its inverse processing record")
    if (
        chunk.action.ndim != 2
        or chunk.action.shape != codec.future_valid_mask.shape
        or codec.q0.shape != (chunk.action.shape[-1],)
        or not torch.equal(chunk.action_valid_mask, codec.future_valid_mask)
    ):
        raise ValueError("delta future shape/mask differs from q0 codec")
    expected_valid = torch.cat((codec.q0_valid_mask.unsqueeze(0), codec.future_valid_mask))
    if record.raw_action_dim != codec.q0.numel() or not torch.equal(record.action_valid_mask, expected_valid):
        raise ValueError("delta inverse record must contain exactly q0 and future validity")
    if not math.isfinite(chunk.q0_timestamp):
        raise ValueError("delta q0 timestamp must be finite")
    if fps is not None:
        _validate_rollout_fps(fps)
        expected = chunk.q0_timestamp + torch.arange(1, chunk.action.shape[0] + 1, dtype=torch.float64) / fps
        if chunk.target_timestamps is not None:
            actual = torch.as_tensor(chunk.target_timestamps).detach().cpu().double()
            # Match the reader's 1 ms timestamp tolerance, including float32
            # episode timestamps whose absolute origin is far from zero.
            if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=0, atol=1e-3):
                raise ValueError("delta target timestamps must follow q0 on the future RGB clock")
    elif chunk.target_timestamps is not None:
        raise ValueError("delta target timestamps require explicit action FPS")
    # The same inverse used for model outputs validates q0 masks and geometry.
    _delta_physical_targets(chunk)


def _delta_physical_targets(chunk):
    codec = chunk.q0_codec
    return codec.decode_to_unified(torch.cat((codec.q0.unsqueeze(0).to(chunk.action), chunk.action), dim=0))[1:]


def _reanchor_delta_piece(chunk, physical, *, start, end, previous=None):
    from greenwam.data.joint_deltas import encode_joint_deltas

    codec = chunk.q0_codec
    q0 = codec.q0 if start == 0 else physical[start - 1]
    q0_valid = codec.q0_valid_mask if start == 0 else codec.future_valid_mask[start - 1]
    if previous is not None:
        # A missing boundary joint is not a later measured correction. Keep it
        # masked through this open-loop rollout instead of fabricating q0.
        q0_valid = q0_valid & (~codec.joint_mask | previous.action_valid_mask[-1])
    result = encode_joint_deltas(
        physical[start:end],
        q0,
        future_valid_mask=codec.future_valid_mask[start:end],
        q0_valid_mask=q0_valid,
        joint_mask=codec.joint_mask,
    )
    encoded = result.action
    if codec.future_normalizer is not None:
        encoded = codec.future_normalizer.normalize_action(encoded)
    encoded = torch.where(result.valid_mask, encoded, 0.0)
    sliced_codec = replace(
        codec,
        q0=result.conditioning_state[0],
        q0_valid_mask=result.conditioning_valid_mask[0],
        future_valid_mask=result.valid_mask,
    )
    record = replace(
        chunk.action_processing_record,
        action_codec=sliced_codec,
        action_valid_mask=torch.cat((result.conditioning_valid_mask, result.valid_mask)),
    )
    timestamp = chunk.q0_timestamp + start / chunk.fps
    return replace(
        chunk,
        action=encoded,
        action_valid_mask=result.valid_mask,
        q0_codec=sliced_codec,
        action_processing_record=record,
        q0_timestamp=timestamp,
        target_timestamps=(timestamp + torch.arange(1, end - start + 1, dtype=torch.float64) / chunk.fps),
    )


def chunk_rollout_actions(
    chunk: RolloutActionChunk, max_block_frames: int = 17, *, block_frames=None
) -> tuple[RolloutActionChunk, ...]:
    """Partition a flat future action horizon exactly, preserving order and masks.

    A 900-frame horizon becomes 52 blocks of 17, then 13, 1, 1, 1. No target
    is repeated or dropped. Every block fits the four-memory/five-future latent
    context in the legacy default. Explicit ``block_frames`` preserves whole
    source blocks of any positive length; availability and projector metadata
    remain unchanged and delta q0 is reanchored at each supplied boundary.
    """
    if not 1 <= max_block_frames <= 17:
        raise ValueError("Block rollout block budget must be between 1 and 17 future frames")
    if chunk.fps is not None:
        _validate_rollout_fps(chunk.fps)
    if chunk.action.ndim != 2 or chunk.action_valid_mask.shape != chunk.action.shape:
        raise ValueError("flat actions and their temporal validity mask must have identical [T,D] shapes")
    _validate_delta_chunk(chunk, fps=chunk.fps)
    if chunk.q0_codec is not None and chunk.fps is None:
        raise ValueError("delta horizon partitioning requires explicit action FPS")
    physical = _delta_physical_targets(chunk) if chunk.q0_codec is not None else None
    pieces = []
    start = 0
    lengths = (
        partition_block_rollout(chunk.action.shape[0], max_block_frames=max_block_frames)
        if block_frames is None
        else tuple(block_frames)
    )
    if (
        any(type(n) is not int or n < 1 for n in lengths)
        or (block_frames is None and any(n > max_block_frames or n % 4 != 1 for n in lengths))
        or sum(lengths) != chunk.action.shape[0]
    ):
        raise ValueError("Explicit action blocks must exactly partition the horizon with positive lengths")
    for frames in lengths:
        end = start + frames
        if physical is not None:
            pieces.append(
                _reanchor_delta_piece(chunk, physical, start=start, end=end, previous=pieces[-1] if pieces else None)
            )
            start = end
            continue
        record = chunk.action_processing_record
        if record is not None and record.action_valid_mask is not None and record.action_valid_mask.ndim == 2:
            if record.action_valid_mask.shape[0] != chunk.action.shape[0]:
                raise ValueError("inverse processing mask must match the complete action horizon")
            record = replace(record, action_valid_mask=record.action_valid_mask[start:end])
        pieces.append(
            replace(
                chunk,
                action=chunk.action[start:end],
                action_valid_mask=chunk.action_valid_mask[start:end],
                action_processing_record=record,
            )
        )
        start = end
    return tuple(pieces)


def _prepare_rollout_actions(chunk: RolloutActionChunk, *, fps: float, max_action_dim: int, whole_block=False):
    """Validate conditioning before generation and return padded projector data."""
    _validate_rollout_fps(fps)
    if chunk.fps is not None:
        _validate_rollout_fps(chunk.fps)
    if chunk.fps is not None and not math.isclose(chunk.fps, fps, rel_tol=1e-6):
        raise ValueError("Block rollout action FPS must equal future RGB FPS")
    _validate_delta_chunk(chunk, fps=fps)
    action = chunk.action
    if action.ndim != 2 or action.shape[0] < 1:
        raise ValueError("future actions must have nonempty [T,D] shape")
    if not whole_block and (action.shape[0] - 1) % 4:
        raise ValueError("future action/RGB length must equal 1 + 4*k")
    if not whole_block and action.shape[0] > 17:
        raise ValueError("Block rollout future blocks cannot exceed 17 frames; partition the flat action horizon first")
    frames, width = action.shape
    native_width = chunk.raw_action_dim if chunk.raw_action_dim is not None else width
    if not 0 < native_width <= width <= max_action_dim:
        raise ValueError("action width/native width must fit the model projector width")
    if chunk.action_contract_id < 0:
        raise ValueError("action_contract_id must be nonnegative")
    if chunk.action_layout_id is not None and chunk.action_layout_id < 0:
        raise ValueError("action_layout_id must be nonnegative")
    valid = chunk.action_valid_mask.to(device=action.device)
    available = chunk.action_availability_mask.to(device=action.device)
    if valid.dtype != torch.bool or available.dtype != torch.bool:
        raise ValueError("action masks must be boolean")
    if valid.shape != action.shape or available.shape not in ((width,), (native_width,)):
        raise ValueError("future action mask shapes must match action/native width")
    if available.numel() < width:
        available = torch.nn.functional.pad(available, (0, width - available.numel()))
    if valid[:, native_width:].any() or available[native_width:].any():
        raise ValueError("padded coordinates must be unavailable and invalid")
    if (valid & ~available.unsqueeze(0)).any():
        raise ValueError("valid mask contradicts action availability")
    if not torch.isfinite(action[valid]).all():
        raise ValueError("valid action coordinates must be finite")
    padding = max_action_dim - width
    action = torch.nn.functional.pad(torch.where(valid, action.float(), 0.0), (0, padding))
    valid = torch.nn.functional.pad(valid, (0, padding))
    available = torch.nn.functional.pad(available, (0, padding))
    return action, valid, available, native_width


def build_rollout_batch(
    *,
    memory_rgb: torch.Tensor,
    chunk: RolloutActionChunk,
    prompt: str,
    fps: float,
    max_action_dim: int,
    input_video_key: str = "video",
    input_caption_key: str = "ai_caption",
    whole_block: bool = False,
) -> dict[str, Any]:
    """Build a single-sample future-only batch; independent memory is a separate key."""
    # Packing types are shared runtime primitives; no training/config imports.
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan

    if memory_rgb.ndim != 4 or memory_rgb.shape[:2] != (3, 4) or memory_rgb.dtype != torch.uint8:
        raise ValueError("memory_rgb must have uint8 shape [3,4,H,W]")
    action, valid, available, native_width = _prepare_rollout_actions(
        chunk, fps=fps, max_action_dim=max_action_dim, whole_block=whole_block
    )
    frames = action.shape[0]
    if chunk.q0_codec is not None:
        codec = chunk.q0_codec
        q0_valid = codec.q0_valid_mask.to(device=action.device)
        if q0_valid.dtype != torch.bool or (q0_valid & ~available[:native_width]).any():
            raise ValueError("q0 validity contradicts action availability")
        q0 = torch.where(q0_valid, codec.q0.to(action), 0.0)
        if not torch.isfinite(q0).all():
            raise ValueError("valid q0 coordinates must be finite")
        action = torch.cat((torch.nn.functional.pad(q0.unsqueeze(0), (0, max_action_dim - native_width)), action))
        valid = torch.cat((torch.nn.functional.pad(q0_valid.unsqueeze(0), (0, max_action_dim - native_width)), valid))
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=[0, 1, 2, 3],
        vision_prefix_frame_count=4,
        condition_frame_indexes_action=list(range(action.shape[0])),
        action_start_frame_offset=-1 if chunk.q0_codec is not None else 0,
    )
    height, width_px = memory_rgb.shape[-2:]
    batch = {
        input_video_key: [torch.zeros((3, frames + (-(frames - 1)) % 4, height, width_px), dtype=torch.uint8)],
        "vision_pixel_prefixes": [memory_rgb],
        input_caption_key: [prompt],
        "prompt": [prompt],
        "mode": ["forward_dynamics"],
        "action": [action],
        "action_contract_id": [torch.tensor(chunk.action_contract_id, dtype=torch.long)],
        "raw_action_dim": [torch.tensor(native_width, dtype=torch.long)],
        "action_valid_mask": [valid],
        "action_availability_mask": [available],
        "conditioning_fps": [torch.tensor(fps)],
        "conditioning_fps_action": [torch.tensor(fps)],
        "sequence_plan": [plan],
    }
    if chunk.action_layout_id is not None:
        batch["action_layout_id"] = [torch.tensor(chunk.action_layout_id, dtype=torch.long)]
    if chunk.domain_id is not None:
        batch["domain_id"] = [torch.tensor(chunk.domain_id, dtype=torch.long)]
    if chunk.action_processing_record is not None:
        batch["action_processing_record"] = [chunk.action_processing_record]
    if chunk.q0_codec is not None:
        batch["action_q0_timestamp"] = [torch.tensor(chunk.q0_timestamp, dtype=torch.float64)]
        batch["action_target_timestamps"] = [
            chunk.q0_timestamp + torch.arange(1, frames + 1, dtype=torch.float64) / fps
        ]
    return batch


def _prepare_block_rollout(initial_image, *, fps, sampling_options, reference_frame_budget, source_resolution):
    """Shared image, sampling-option and native source budget validation."""
    _validate_rollout_fps(fps)
    if initial_image.ndim == 4 and initial_image.shape[1] == 1:
        initial_image = initial_image[:, 0]
    if initial_image.ndim != 3 or initial_image.shape[0] != 3 or initial_image.dtype != torch.uint8:
        raise ValueError("initial_image must be exactly one uint8 RGB image [3,H,W]")
    options = dict(sampling_options or {})
    forbidden = {"data_batch", "seed", "n_sample", "sampler", "velocity_postprocess_builder", "upsample_task"}
    if forbidden & options.keys():
        raise ValueError(f"sampling_options cannot override block rollout fields: {sorted(forbidden & options.keys())}")
    if (
        isinstance(reference_frame_budget, bool)
        or not isinstance(reference_frame_budget, int)
        or not 1 <= reference_frame_budget <= 900
    ):
        raise ValueError("reference_frame_budget must be an integer between 1 and 900")
    source_height, source_width = source_resolution if source_resolution is not None else initial_image.shape[-2:]
    frame_limit = scaled_rollout_frames(reference_frame_budget, fps=fps, height=source_height, width=source_width)
    return initial_image, options, frame_limit


@torch.no_grad()
def generate_block_rollout(
    model: Any,
    *,
    initial_image: torch.Tensor,
    action_chunks: Iterable[RolloutActionChunk] | RolloutActionChunk,
    prompt: str | Mapping[str, Any],
    fps: float,
    seed: int = 1,
    sampler: Any = None,
    sampling_options: Mapping[str, Any] | None = None,
    reference_frame_budget: int = 900,
    source_resolution: tuple[int, int] | None = None,
) -> BlockRollout:
    """Generate full new chunks with four clean independent RGB memory latents.

    ``prompt`` is a rendered external caption (repeated) or a user-supplied
    source-frame text schedule. Schedules split flat trajectories at admitted-interval
    boundaries; explicitly partitioned inputs crossing a boundary are rejected.
    Text is passed verbatim to the model tokenizer, with a 512 final-token cap.
    ``model`` is a block-rollout-enabled OmniMoTModel. Sampling options are forwarded to
    ``generate_samples_from_batch`` (e.g. num_steps/guidance/shift). No mask-control
    branch or causal KV cache is constructed. The model performs independent VAE
    encoding of each prefix frame; only its generated latent suffix is decoded.

    A single ``RolloutActionChunk`` represents a flat future horizon and is partitioned
    automatically using the legacy short partition; an iterable supplies whole
    blocks with arbitrary positive lengths and preserves their boundaries. The complete
    horizon is checked before any model call. Its cap is B/30 seconds, where
    B <= 900, converted to frames at effective ``fps``. ``source_resolution``
    retains native camera (height, width) provenance, including packed stereo,
    but does not affect duration. Actual canvas tokens determine compute cost. The initial real
    image is excluded from the budget. Only codec alignment RGB is padded, then
    trimmed from decode; real frames and action targets are never padded/dropped.

    Delta blocks anchor to the initial observed q0, then to the preceding target
    in the supplied physical trajectory. They do not infer states from generated
    RGB or read reference video. Each block reuses its pinned future normalizer.
    """
    initial_image, options, frame_limit = _prepare_block_rollout(
        initial_image,
        fps=fps,
        sampling_options=sampling_options,
        reference_frame_budget=reference_frame_budget,
        source_resolution=source_resolution,
    )
    if isinstance(action_chunks, RolloutActionChunk):
        if action_chunks.action.ndim != 2:
            raise ValueError("flat future actions must have shape [T,D]")
        if action_chunks.action.shape[0] > frame_limit:
            raise ValueError(f"future action horizon exceeds the scaled frame budget {frame_limit}")
        planned = plan_prompt_blocks(prompt, action_chunks.action.shape[0], mode="forward_ar", fps=fps)
        action_chunks = chunk_rollout_actions(action_chunks, block_frames=[b["frames"] for b in planned])
    prepared_chunks = []
    future_frames = 0
    contract_id = None
    for chunk in action_chunks:
        if chunk.action.ndim != 2:
            raise ValueError("future actions must have shape [T,D]")
        future_frames += chunk.action.shape[0]
        if future_frames > frame_limit:
            raise ValueError(f"future action horizon exceeds the scaled frame budget {frame_limit}")
        if contract_id is not None and chunk.action_contract_id != contract_id:
            raise ValueError("all chunks of one block rollout must use the same projector action contract")
        contract_id = chunk.action_contract_id
        _prepare_rollout_actions(chunk, fps=fps, max_action_dim=model.config.max_action_dim, whole_block=True)
        if prepared_chunks:
            previous = prepared_chunks[-1]
            if (previous.q0_codec is None) != (chunk.q0_codec is None):
                raise ValueError("all block rollout blocks must use the same joint encoding")
            if chunk.q0_codec is not None:
                expected_time = previous.q0_timestamp + previous.action.shape[0] / fps
                if not math.isclose(chunk.q0_timestamp, expected_time, rel_tol=0, abs_tol=1e-3):
                    raise ValueError("next block q0 timestamp differs from the preceding boundary")
                if not torch.equal(previous.q0_codec.joint_mask, chunk.q0_codec.joint_mask):
                    raise ValueError("all block rollout blocks must use identical delta joint semantics")
                if (chunk.q0_codec.q0_valid_mask & chunk.q0_codec.joint_mask & ~previous.action_valid_mask[-1]).any():
                    raise ValueError("next block q0 cannot invent an unknown supplied trajectory boundary joint")
                shared = previous.action_valid_mask[-1] & chunk.q0_codec.q0_valid_mask
                boundary = _delta_physical_targets(previous)[-1]
                if not torch.allclose(boundary[shared], chunk.q0_codec.q0[shared].to(boundary), rtol=1e-5, atol=1e-6):
                    raise ValueError("next block q0 differs from the supplied trajectory boundary")
        prepared_chunks.append(chunk)
    if not prepared_chunks:
        raise ValueError("action_chunks must contain at least one future chunk")
    return _generate_rollout_blocks(
        model,
        initial_image=initial_image,
        block_frames=tuple(chunk.action.shape[0] for chunk in prepared_chunks),
        prompt=prompt,
        fps=fps,
        seed=seed,
        sampler=sampler,
        options=options,
        action_chunks=prepared_chunks,
    )


def build_rollout_i2v_batch(
    *,
    memory_rgb: torch.Tensor,
    num_frames: int,
    prompt: str,
    fps: float,
    input_video_key: str = "video",
    input_caption_key: str = "ai_caption",
    whole_block: bool = False,
) -> dict[str, Any]:
    """Build a visual-only block rollout block without action tensors or projector routing."""
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan

    if memory_rgb.ndim != 4 or memory_rgb.shape[:2] != (3, 4) or memory_rgb.dtype != torch.uint8:
        raise ValueError("memory_rgb must have uint8 shape [3,4,H,W]")
    if (
        isinstance(num_frames, bool)
        or not isinstance(num_frames, int)
        or num_frames < 1
        or (not whole_block and (num_frames > 17 or (num_frames - 1) % 4))
    ):
        raise ValueError("Block rollout future frames must equal 1 + 4*k and fit the 17-frame block")
    _validate_rollout_fps(fps)
    height, width = memory_rgb.shape[-2:]
    return {
        input_video_key: [torch.zeros((3, num_frames + (-(num_frames - 1)) % 4, height, width), dtype=torch.uint8)],
        "vision_pixel_prefixes": [memory_rgb],
        input_caption_key: [prompt],
        "prompt": [prompt],
        "mode": ["i2v"],
        "conditioning_fps": [torch.tensor(fps)],
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=False,
                condition_frame_indexes_vision=[0, 1, 2, 3],
                vision_prefix_frame_count=4,
            )
        ],
    }


@torch.no_grad()
def generate_block_i2v_rollout(
    model: Any,
    *,
    initial_image: torch.Tensor,
    num_frames: int,
    prompt: str | Mapping[str, Any],
    fps: float,
    seed: int = 1,
    sampler: Any = None,
    sampling_options: Mapping[str, Any] | None = None,
    reference_frame_budget: int = 900,
    source_resolution: tuple[int, int] | None = None,
    block_frames: Iterable[int] | None = None,
) -> BlockRollout:
    """Generate exactly num_frames future RGB frames with no action conditioning.

    Each block uses four independently encoded frames from the full generated
    history; the first repeats the real initial observation. The initial image
    is excluded from num_frames. Physical-time budgeting, native source_resolution provenance, and sampling options match generate_block_rollout.
    Explicit ``block_frames`` preserves the supplied whole-block boundaries;
    omission retains the legacy short-block partition.
    """
    initial_image, options, frame_limit = _prepare_block_rollout(
        initial_image,
        fps=fps,
        sampling_options=sampling_options,
        reference_frame_budget=reference_frame_budget,
        source_resolution=source_resolution,
    )
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer future-frame count")
    if num_frames > frame_limit:
        raise ValueError(f"future video horizon exceeds the scaled frame budget {frame_limit}")
    planned = plan_prompt_blocks(prompt, num_frames, mode="i2v_ar", fps=fps, block_frames=block_frames)
    return _generate_rollout_blocks(
        model,
        initial_image=initial_image,
        block_frames=tuple(b["frames"] for b in planned),
        prompt=prompt,
        fps=fps,
        seed=seed,
        sampler=sampler,
        options=options,
    )


def _generate_rollout_blocks(
    model,
    *,
    initial_image,
    block_frames,
    prompt,
    fps,
    seed,
    sampler,
    options,
    action_chunks=None,
):
    """Shared generated-history loop; action-free blocks never construct actions."""
    texts = plan_prompt_blocks(
        prompt,
        sum(block_frames),
        fps=fps,
        mode="forward_ar" if action_chunks is not None else "i2v_ar",
        block_frames=block_frames,
    )
    # The public tokenizer uses the same chat-template helper as sampling.
    # Check every final rendered condition before starting an expensive rollout.
    token_counts = {}
    for item in texts:
        text = item["text"]
        if text not in token_counts:
            tokens = model.tokenize_text(text, is_video=False)
            count = tokens.shape[-1]
            if count > 512:
                raise ValueError(f"Block rollout prompt {item['atom_id']} has {count} final tokens; budget is 512")
            token_counts[text] = count
    history = initial_image.detach().cpu().unsqueeze(1).clone()
    outputs, records = [], []
    time_origin = (
        action_chunks[0].q0_timestamp if action_chunks is not None and action_chunks[0].q0_codec is not None else 0.0
    )
    model.eval()
    for index, frames in enumerate(block_frames):
        before = history.shape[1]
        indexes = rollout_memory_indexes(before)
        common = dict(
            memory_rgb=history[:, indexes].clone(),
            prompt=texts[index]["text"],
            fps=fps,
            input_video_key=model.input_video_key,
            input_caption_key=model.input_caption_key,
        )
        if action_chunks is None:
            batch = build_rollout_i2v_batch(num_frames=frames, whole_block=True, **common)
            contract_id = None
        else:
            chunk = action_chunks[index]
            batch = build_rollout_batch(
                chunk=chunk, max_action_dim=model.config.max_action_dim, whole_block=True, **common
            )
            contract_id = chunk.action_contract_id
        # Record actual unique-history positions, not source/GT frame indexes.
        batch["vision_prefix_frame_indexes"] = [torch.tensor(indexes)]
        batch["vision_target_frame_indexes"] = [torch.arange(before, before + frames)]
        batch["vision_prefix_timestamps"] = [time_origin + torch.tensor(indexes, dtype=torch.float64) / fps]
        batch["vision_target_timestamps"] = [time_origin + batch["vision_target_frame_indexes"][0].double() / fps]
        q0_audit = {}
        if action_chunks is not None and chunk.q0_codec is not None:
            batch["action_q0_frame_index"] = [torch.tensor(before - 1)]
            q0_audit = dict(
                action_q0_frame_index=before - 1,
                action_q0_timestamp=chunk.q0_timestamp,
                action_q0_source="initial_observed_state" if index == 0 else "supplied_trajectory_boundary",
            )
        memory_hashes = _rgb_frame_hashes(batch["vision_pixel_prefixes"][0])
        sampled = model.generate_samples_from_batch(batch, sampler=sampler, seed=[seed + index], n_sample=1, **options)
        latent = sampled["vision"]
        if len(latent) != 1 or latent[0].ndim != 5 or latent[0].shape[0] != 1 or latent[0].shape[2] <= 4:
            raise ValueError(
                "Block rollout sampler must return one BCTHW latent with four memory slots and a future suffix"
            )
        padding = (-(frames - 1)) % 4
        padded_frames = frames + padding
        expected_latents = 4 + 1 + (padded_frames - 1) // 4
        if latent[0].shape[2] != expected_latents:
            raise ValueError(
                f"Block rollout sampler latent length must equal {expected_latents} for the complete block"
            )
        decoded = model.decode(latent[0][:, :, 4:]).float().cpu()
        expected = (1, 3, padded_frames, *history.shape[-2:])
        if tuple(decoded.shape) != expected or not torch.isfinite(decoded).all():
            raise ValueError(f"decoded future must be finite and have shape {expected}; got {tuple(decoded.shape)}")
        # Only codec's synthetic tail is cropped; every real block frame enters
        # output and the next block's generated memory exactly once.
        decoded = decoded[:, :, :frames].clamp(-1, 1)
        outputs.append(decoded)
        generated_rgb = ((decoded[0] + 1) * 127.5).round().to(torch.uint8)
        history = torch.cat([history, generated_rgb], dim=1)
        records.append(
            RolloutChunkRecord(
                indexes,
                before,
                [0, 1, 2, 3],
                decoded.shape[2],
                history.shape[1],
                contract_id,
                memory_hashes,
                _rgb_frame_hashes(generated_rgb),
                prompt_text=texts[index]["text"],
                prompt_text_sha256=texts[index]["text_sha256"],
                prompt_atom_id=texts[index]["atom_id"],
                prompt_start_frame=texts[index]["start_frame"],
                prompt_end_frame=texts[index]["end_frame"],
                prompt_binding=texts[index]["binding"],
                prompt_provenance=texts[index]["provenance"],
                prompt_token_count=token_counts[texts[index]["text"]],
                codec_padding_frames=padding,
                **q0_audit,
            )
        )
    return BlockRollout(torch.cat(outputs, dim=2), history, records)
