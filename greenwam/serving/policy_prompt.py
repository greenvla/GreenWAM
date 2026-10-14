# SPDX-License-Identifier: OpenMDW-1.1
"""Compile one served GreenWAM policy observation into a model sample, for the single-arm Franka EEF servers.

Copy of the policy-inference path of the original action transform pipeline of the fixed-layout CALVIN and LIBERO
runs (``ActionTransformPipeline`` with ``skip_video_resize``, the GreenWAM JSON prompt formatter, text tokenization without
CFG dropout, ``build_sequence_plan_from_mode("policy")``, ``ActionProcessor.preprocess_action`` without a
normalizer) and of the batch assembly of their policy server. The statements keep their order; branches that are dead
on this path (video resize, the string prompt appenders, idle-frame text, state and history tokens, action padding
masks, memory clips, the other modes) are removed and raise if a sample would need them. The prediction horizon and
the conditioning fps come with the sample, so CALVIN (q0 + 20 rows) and LIBERO (q0 + 32 rows) share it.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch

from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

CANVAS_CAPTION = "Multi-camera canvas: main camera above; additional cameras below in the declared order."
WAM_MODE_DESCRIPTION = "Predict the future video and action given the image."
VIEWPOINT_TEMPLATES = {"concat_view": "This video contains concatenated views from multiple camera perspectives."}
# Embodiment domains of the original domain table for the served benchmarks.
EMBODIMENT_TO_DOMAIN_ID = {"libero": 5, "calvin": 21}
# The original prompt looked the canvas up in its resolution table before reducing the fraction; these canvases
# (width, height) are in neither revision of that table, so their aspect ratio is the reduced fraction.
_CANVAS_SIZES_WITHOUT_TABLE_ENTRY = frozenset({(288, 384)})
_MAX_NUM_TOKENS = 4096
_MULTI_ITEM_KEYS = frozenset({"text_token_ids", "images", "video", "action", "action_raw", "sound"})


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


class ActionPromptJsonFormatter:
    """Format action prompts into a structured JSON-compatible dictionary.

    JSON fields are emitted in this order: ``cinematography``, ``actions``, ``duration``, ``fps``, ``resolution``,
    then ``aspect_ratio``. ``cinematography`` is a dictionary, duration is truncated to an integer-second string such
    as ``"2s"``, and aspect ratio is stored as a comma-separated string such as ``"3,4"``.
    """

    def __init__(
        self,
        caption_key: str = "ai_caption",
        viewpoint_key: str = "viewpoint",
        video_key: str = "video",
        fps_key: str = "conditioning_fps",
        image_size_key: str = "image_size",
        idle_frames_key: str = "idle_frames",
        video_num_frames_key: str = "video_num_frames",
    ) -> None:
        self.caption_key: str = caption_key
        self.viewpoint_key: str = viewpoint_key
        self.video_key: str = video_key
        self.fps_key: str = fps_key
        self.image_size_key: str = image_size_key
        self.idle_frames_key: str = idle_frames_key
        self.video_num_frames_key: str = video_num_frames_key
        self.viewpoint_templates: dict[str, str] = VIEWPOINT_TEMPLATES

    def __call__(self, data_dict: dict) -> dict:
        """Replace the caption with the action JSON prompt structure."""
        additional_view_description = data_dict.pop("additional_view_description", None)
        caption = data_dict.get(self.caption_key)
        if not isinstance(caption, str) or caption == "":
            return data_dict

        height, width = self._get_resolution(data_dict)
        fps = self._get_scalar_float(data_dict.get(self.fps_key), self.fps_key)
        if fps <= 0:
            raise ValueError(f"ActionPromptJsonFormatter: '{self.fps_key}' must be positive, got {fps}")

        video = data_dict.get(self.video_key)
        if not isinstance(video, torch.Tensor) or video.ndim < 2:
            raise ValueError(
                f"ActionPromptJsonFormatter: expected '{self.video_key}' to be a video tensor with shape "
                f"(C, T, H, W), got {type(video).__name__}"
            )
        duration_seconds = int(data_dict.get(self.video_num_frames_key, video.shape[1])) / fps
        duration = self._truncate_seconds(duration_seconds)
        action_end_time = self._round_time_seconds(duration_seconds)

        prompt = {
            "cinematography": {
                "framing": self._get_viewpoint_caption(data_dict, additional_view_description),
            },
            "actions": [
                {
                    "time": f"0:00-{self._format_time_mss(action_end_time)}",
                    "description": self._ensure_sentence(caption),
                    "idle_frame": self._get_idle_frame_info(data_dict),
                }
            ],
            "duration": f"{duration}s",
            "fps": float(fps),
            "resolution": {"H": height, "W": width},
            "aspect_ratio": self._get_aspect_ratio(width, height),
        }
        cleaned_prompt = self._drop_empty_fields(prompt)
        self._raise_if_empty_fields(cleaned_prompt)
        data_dict[self.caption_key] = cleaned_prompt
        return data_dict

    def _truncate_seconds(self, seconds: float) -> int:
        """Truncate duration to integer seconds, matching video JSON-caption augmentors."""
        if seconds < 0 or not math.isfinite(seconds):
            return 0
        return int(seconds)

    def _round_time_seconds(self, seconds: float) -> int:
        """Round an action timestamp to integer seconds, matching video captioning."""
        if seconds < 0 or not math.isfinite(seconds):
            return 0
        return round(seconds)

    def _format_time_mss(self, seconds: int) -> str:
        """Format integer seconds as M:SS for JSON prompt time ranges."""
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes}:{remaining_seconds:02d}"

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Reduced width,height fraction of a served canvas."""
        if (width, height) not in _CANVAS_SIZES_WITHOUT_TABLE_ENTRY:
            raise ValueError(f"no serving aspect ratio for a {width}x{height} canvas (resolution table lookup removed)")
        divisor = math.gcd(width, height)
        return f"{width // divisor},{height // divisor}"

    def _get_viewpoint_caption(self, data_dict: dict, additional_view_description: object | None) -> str | None:
        """Resolve the viewpoint text used in the ``cinematography`` field."""
        viewpoint = data_dict.get(self.viewpoint_key)
        template = self.viewpoint_templates.get(viewpoint) if isinstance(viewpoint, str) else None
        if template is None:
            raise ValueError(f"unserved viewpoint {viewpoint!r}; the GreenWAM policy prompt uses concat_view")
        if additional_view_description:
            separator = " " if template.endswith(".") else ". "
            template = template + separator + str(additional_view_description).rstrip()
        return template

    def _get_resolution(self, data_dict: dict) -> tuple[int, int]:
        """Resolve ``(height, width)`` from the post-padding image size."""
        image_size = data_dict.get(self.image_size_key)
        if image_size is None:
            raise ValueError(f"ActionPromptJsonFormatter: missing '{self.image_size_key}' in data_dict.")

        if isinstance(image_size, torch.Tensor):
            if image_size.numel() < 2:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected '{self.image_size_key}' to contain at least "
                    f"height and width, got shape {tuple(image_size.shape)}"
                )
            return int(image_size[0].item()), int(image_size[1].item())

        try:
            return int(image_size[0]), int(image_size[1])
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(
                f"ActionPromptJsonFormatter: expected '{self.image_size_key}' to contain height and width."
            ) from e

    def _get_scalar_float(self, value: object, key: str) -> float:
        """Parse a required scalar float from a tensor or Python value."""
        if value is None:
            raise ValueError(f"ActionPromptJsonFormatter: missing '{key}' in data_dict.")

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected scalar tensor at '{key}', got shape {tuple(value.shape)}"
                )
            return float(value.item())

        if isinstance(value, (str, int, float)):
            try:
                return float(value)
            except ValueError as e:
                raise ValueError(
                    f"ActionPromptJsonFormatter: expected scalar float-compatible value at '{key}'."
                ) from e
        raise ValueError(f"ActionPromptJsonFormatter: expected scalar float-compatible value at '{key}'.")

    def _get_idle_frame_info(self, data_dict: dict) -> str | None:
        """Idle-frame text: served policy samples carry no idle-frame count, so there is none."""
        if data_dict.get(self.idle_frames_key) is not None:
            raise ValueError("idle-frame prompt text is not served")
        return None

    def _ensure_sentence(self, text: str) -> str:
        """Return text with terminal sentence punctuation."""
        text = text.strip()
        if text.endswith((".", "!", "?")):
            return text
        return f"{text}."

    def _drop_empty_fields(self, value: object) -> object:
        """Recursively remove empty strings, dictionaries, lists, and ``None`` values."""
        if isinstance(value, dict):
            return {
                key: cleaned
                for key, item in value.items()
                if not self._is_empty(cleaned := self._drop_empty_fields(item))
            }
        if isinstance(value, list):
            return [cleaned for item in value if not self._is_empty(cleaned := self._drop_empty_fields(item))]
        return value

    def _is_empty(self, value: object) -> bool:
        """Return whether a JSON field should be dropped."""
        return value is None or value == "" or value == [] or value == {}

    def _raise_if_empty_fields(self, value: object, path: str = "prompt") -> None:
        """Validate that no empty JSON fields remain after pruning."""
        if self._is_empty(value):
            raise ValueError(f"ActionPromptJsonFormatter: empty field remains at {path}.")

        if isinstance(value, dict):
            for key, item in value.items():
                self._raise_if_empty_fields(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._raise_if_empty_fields(item, f"{path}[{index}]")


class PolicyPromptJsonFormatter(ActionPromptJsonFormatter):
    """GreenWAM JSON prompt with the WAM mode phrase placed as GreenWAM's pipeline does."""

    def __call__(self, data_dict: dict) -> dict:
        robot = data_dict.pop("greenwam_robot", None)
        caption = data_dict.get(self.caption_key)
        if not isinstance(caption, str) or caption == "":
            return data_dict
        if not isinstance(robot, str) or not robot.strip():
            raise ValueError("GreenWAM JSON prompt requires greenwam_robot")
        data_dict[self.viewpoint_key] = "concat_view"
        data_dict = super().__call__(data_dict)
        prompt = data_dict[self.caption_key]
        if not isinstance(prompt, dict) or "cinematography" not in prompt:
            raise ValueError("Structured prompt missing after formatting")
        ordered: dict[str, Any] = {}
        for key, value in prompt.items():
            ordered[key] = value
            if key == "cinematography":
                ordered["robot"] = robot.strip()
        actions = ordered.pop("actions")
        data_dict[self.caption_key] = {"actions": actions, "mode_description": WAM_MODE_DESCRIPTION, **ordered}
        return data_dict


def build_policy_sequence_plan(
    video_length: int,
    action_length: int,
    has_text: bool = True,
    num_history_actions: int = 0,
    state_token_count: int = 0,
) -> SequencePlan:
    """``build_sequence_plan_from_mode("policy", ...)``: first vision frame clean, the action alignment cases as there.

    Case A (action = video_length - 1): only prepended tokens condition. Case B (action = video_length): one leading
    initial-state action conditions too (the q0 row). Case C (subsampled video): B when ``(base - 1)`` divides by
    ``(video_length - 1)``, else A.
    """
    condition_frame_indexes_vision = [0]
    prefix = num_history_actions + state_token_count
    base_action_length = action_length - prefix
    has_initial_state = video_length > 1 and (base_action_length - 1) % (video_length - 1) == 0
    if base_action_length == video_length - 1:
        condition_frame_indexes_action = list(range(prefix))
    elif base_action_length == video_length:
        condition_frame_indexes_action = list(range(prefix + 1))
    else:
        condition_frame_indexes_action = list(range(prefix + 1 if has_initial_state else prefix))

    if base_action_length == video_length - 1:
        action_start_frame_offset = 1 - prefix
    elif base_action_length == video_length:
        action_start_frame_offset = -prefix
    else:
        action_start_frame_offset = -prefix if has_initial_state else 1 - prefix

    return SequencePlan(
        has_text=has_text,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=condition_frame_indexes_vision,
        condition_frame_indexes_action=condition_frame_indexes_action,
        action_start_frame_offset=action_start_frame_offset,
    )


def pad_action_to_max_dim(
    action: torch.Tensor, max_action_dim: int
) -> torch.Tensor:  # action: [T,D], returns [T,D_model]
    """Pad action tensor to max_action_dim along the last dimension."""
    if action.shape[-1] > max_action_dim:
        raise ValueError(f"Action dimension {action.shape[-1]} is greater than max_action_dim {max_action_dim}")
    if action.shape[-1] == max_action_dim:
        return action  # [T,D_model]
    padding_size = max_action_dim - action.shape[-1]
    zero_padding = torch.zeros(*action.shape[:-1], padding_size, dtype=action.dtype, device=action.device)  # [T,D_pad]
    return torch.cat([action, zero_padding], dim=-1)  # [T,D_model]


def preprocess_action(
    data_dict: dict[str, Any], action: torch.Tensor, *, max_action_dim: int, action_channel_masking: bool = True
) -> dict[str, Any]:
    """``ActionProcessor.preprocess_action`` without a normalizer: the codec of the server normalized already.

    ``action_raw`` is the action as given, ``action`` the padded model-space action; the record that inverts it is
    the core ``ActionProcessingRecord`` (no normalizer, no codec), so the generated action leaves the model as is.
    """
    raw_action_dim = int(action.shape[-1])
    if "action_raw" in data_dict:
        raise ValueError("action_raw is reserved for the canonical ActionProcessor input")
    action_raw = action.clone()  # [T,D]

    processed_data_dict = dict(data_dict)
    processed_data_dict["action_raw"] = action_raw
    processed_data_dict["action"] = pad_action_to_max_dim(action, max_action_dim)  # [T,D_model]
    record = ActionProcessingRecord(raw_action_dim=raw_action_dim, action_normalizer=None)
    processed_data_dict["raw_action_dim"] = (
        torch.tensor(record.raw_action_dim, dtype=torch.long) if action_channel_masking else None
    )  # []
    processed_data_dict["action_processing_record"] = record
    mask = processed_data_dict.get("action_valid_mask")
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.ndim not in (1, 2) or int(mask.shape[-1]) != raw_action_dim:
            raise ValueError(
                f"action_valid_mask shape {tuple(mask.shape)} does not match action width {raw_action_dim}"
            )
        if mask.ndim == 2 and int(mask.shape[0]) != int(action.shape[0]):
            raise ValueError("temporal action_valid_mask length does not match action length")
        processed_data_dict["action_valid_mask"] = mask
    return processed_data_dict


class PolicySampleTransform:
    """Served GreenWAM policy sample -> model sample: canvas size, JSON prompt, token ids, sequence plan, model action."""

    def __init__(
        self,
        tokenizer_config: dict,
        max_action_dim: int,
        caption_key: str = "ai_caption",
        text_token_key: str = "text_token_ids",
        action_channel_masking: bool = True,
    ) -> None:
        self.caption_key = caption_key
        self.text_token_key = text_token_key
        self.max_action_dim = int(max_action_dim)
        self.action_channel_masking = bool(action_channel_masking)
        self.prompt_json_formatter = PolicyPromptJsonFormatter(caption_key=caption_key)
        self._processor = lazy_instantiate(tokenizer_config)

    def _tokenize(self, data_dict: dict) -> dict:
        input_caption = data_dict[self.caption_key]
        if isinstance(input_caption, dict):
            # Encode dict into a json string. This json string is then passed to the transformer tokenizer.
            input_caption = json.dumps(input_caption)
            data_dict[self.caption_key] = input_caption
        text_ids = self._processor.tokenize_text(input_caption, is_video=False, use_system_prompt=False)
        text_ids = text_ids[:_MAX_NUM_TOKENS]
        data_dict[self.text_token_key] = torch.tensor(text_ids)
        return data_dict

    def __call__(self, data_dict: dict) -> dict:
        mode = data_dict.get("mode")
        if mode != "policy":
            raise ValueError(f"the GreenWAM policy transform serves mode 'policy', got {mode!r}")
        for key in ("memory_video", "robot_state", "history_action", "actions_padding_mask"):
            if data_dict.get(key) is not None:
                raise ValueError(f"{key} is not served by the GreenWAM policy transform")

        # Canvas is already at its training size: the image size is its own [H, W, H, W].
        video = data_dict.get("video")
        assert isinstance(video, torch.Tensor), "video is required when skip_video_resize=True"
        h, w = video.shape[-2:]
        data_dict["image_size"] = torch.tensor([h, w, h, w], dtype=torch.float)

        data_dict = self.prompt_json_formatter(data_dict)

        if data_dict.get(self.text_token_key) is None:
            data_dict = self._tokenize(data_dict)

        video = data_dict.get("video")
        action = data_dict.get("action")
        assert video is not None, "video is required"
        video_length = video.shape[1]  # [C,T,H,W] -> T
        action_length = action.shape[0] if isinstance(action, torch.Tensor) else max(video_length - 1, 0)
        data_dict["sequence_plan"] = build_policy_sequence_plan(video_length=video_length, action_length=action_length)

        assert isinstance(action, torch.Tensor), "action tensor is required when sequence plan has action"
        data_dict = preprocess_action(
            data_dict, action, max_action_dim=self.max_action_dim, action_channel_masking=self.action_channel_masking
        )

        if data_dict.get("memory_elapsed_s") is None:
            data_dict["memory_elapsed_s"] = torch.tensor(0.0, dtype=torch.float32)
        return data_dict


def _build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "video" and isinstance(value, list):
            data_batch[key] = [value]
        elif key in _MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]
        else:
            data_batch[key] = [value]
    return data_batch


def build_data_batch_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Need at least one sample to build a data batch")
    merged: dict[str, Any] = {}
    for sample in samples:
        single = _build_data_batch_from_sample(sample)
        if not merged:
            merged = {key: list(value) for key, value in single.items()}
            continue
        for key, value in single.items():
            merged[key].extend(value)
    return merged


def policy_data_batch(
    samples: list[dict[str, Any]],
    *,
    action_contract_id: int,
    action_layout_id: int,
    action_conditioning_contract_id: int,
) -> dict[str, Any]:
    """Batch of served samples for ``generate_samples_from_batch``.

    One [T,128] valid mask per sample (as the training collate yields, not [1,T,128]) and the core routing of the
    layout: the future rows' contract, the semantic layout and the contract of the q0 conditioning row.
    """
    data_batch = build_data_batch_from_samples(samples)
    data_batch["action_valid_mask"] = [m[0] if m.ndim == 3 else m for m in data_batch["action_valid_mask"]]
    n = len(samples)
    data_batch["action_contract_id"] = [torch.tensor(int(action_contract_id))] * n
    data_batch["action_layout_id"] = [torch.tensor(int(action_layout_id))] * n
    data_batch["action_conditioning_contract_id"] = [torch.tensor(int(action_conditioning_contract_id))] * n
    return data_batch
