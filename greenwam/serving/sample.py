# SPDX-License-Identifier: OpenMDW-1.1
"""Compile one served observation into a model sample: copy of the WAM path of the training dataset, of
`wam_video_frames` (with `resolve_window`) and of `pad_full_subtask_video`.

`ServingDataset.__init__` / `__getitem__` keep the statements of the original in their order for a raw WAM sample
with joint `delta_from_q0` actions and no canvas release. Removed because they are dead there: prompt manifests,
colour augmentation, layout captions, the visual / AR / inverse modes and the EEF and absolute-joint action branches;
a sample that would need one of them raises instead of taking another path.
"""

from __future__ import annotations

import math
from pathlib import Path

from torch.utils.data import Dataset

from greenwam.configs.greenwam_modes import VISUAL_MODES, mode_spec, prepare_mode_sample
from greenwam.data.prompt_contract import DEFAULT_MAX_TEXT_TOKENS, validate_max_text_tokens

AUX_SHORT_MODES = frozenset({"t2v", "i2v", "forward"})  # auxiliaries of a WAM post-train; AR keeps whole atoms


def resolve_window(
    mode,
    *,
    atom_window_policy,
    wam_window_policy,
    aux_window_policy,
    n_frames,
    wam_n_frames,
    wam_horizon_s,
    effective_fps,
):
    """Window policy and RGB length of one reader.

    WAM is always a short window (tiles in midtrain, every admissible start when dense). Visual auxiliaries
    follow the atom by default; ``aux_window_policy='wam_window'`` gives them the WAM-sized window instead,
    which is what makes a post-training step cost the same for every mode (forward on a whole 10 s atom
    measured 8.6x a WAM step on one H100).
    """
    from greenwam.data.wam_temporal import wam_future_frames

    wam_policy = (
        "dense"
        if wam_window_policy == "dense"
        else "tile"
        if atom_window_policy == "full_subtask"
        else atom_window_policy
    )
    if mode == "wam":
        return wam_policy, int(wam_n_frames if atom_window_policy == "full_subtask" else n_frames)
    if aux_window_policy == "wam_window" and mode in AUX_SHORT_MODES:
        # The causal VAE only accepts 1+4k frames. WAM reaches that by padding after reading; an auxiliary
        # clip is read at that length directly, so round the physical window up the same way.
        return wam_policy, 1 + 4 * math.ceil(wam_future_frames(wam_horizon_s, effective_fps) / 4)
    return atom_window_policy, int(n_frames)


WINDOW_KEYS = (
    "atom_window_policy",
    "wam_window_policy",
    "aux_window_policy",
    "n_frames",
    "wam_n_frames",
    "wam_horizon_s",
)


def _sampling_blocks(node):
    if isinstance(node, dict):
        if "wam_n_frames" in node and "atom_window_policy" in node:
            yield node
        for value in node.values():
            yield from _sampling_blocks(value)
    elif isinstance(node, list):
        for value in node:
            yield from _sampling_blocks(value)


def wam_video_frames(config_file, effective_fps):
    """Unpadded RGB length produced by the WAM reader for this effective FPS.

    An explicit physical horizon overrides the nominal configured frame count in
    GreenWAMLeRobotDataset. Codec padding is applied later by MidtrainSFTDataset.
    """
    import yaml

    blocks = [
        {key: block.get(key) for key in WINDOW_KEYS}
        for block in _sampling_blocks(yaml.safe_load(Path(config_file).read_text()))
    ]
    if not blocks:
        raise ValueError(f"{config_file} holds no GreenWAM sampling block; the WAM window cannot be derived")
    if len({tuple(sorted(block.items())) for block in blocks}) != 1:
        raise ValueError(f"{config_file} disagrees on the WAM window across its dataloaders: {blocks}")
    block = blocks[0]
    if block["wam_horizon_s"] is not None:
        from greenwam.data.wam_temporal import wam_future_frames

        future_frames = wam_future_frames(block["wam_horizon_s"], effective_fps)
        if future_frames < 1:
            raise ValueError("WAM horizon contains no future frame at effective FPS")
        return 1 + future_frames
    _, frames = resolve_window(
        "wam",
        atom_window_policy=block["atom_window_policy"],
        wam_window_policy=block["wam_window_policy"],
        aux_window_policy=block["aux_window_policy"],
        n_frames=block["n_frames"],
        wam_n_frames=block["wam_n_frames"],
        wam_horizon_s=block["wam_horizon_s"],
        effective_fps=effective_fps,
    )
    return int(frames)


def pad_full_subtask_video(sample):
    """Replicate at most three terminal RGB frames only at the VAE boundary.

    Source frame indexes, captions, measured actions and masks remain unpadded.
    The last latent contains the real tail plus codec padding, not a new action.
    It retains normal mode-specific latent loss weight: masking it out would
    discard real tail supervision. No wholly synthetic latent is created.
    Pixel validity is provenance, not a conditioning mask (padding must never
    become clean future conditioning). The real-frame counts describe new RGB
    support per latent, not the causal encoder's full receptive field.
    """
    if "full_subtask_frames" not in sample:
        return sample
    import torch

    result = dict(sample)
    video = result["video"]
    if video.ndim != 4 or video.shape[1] < 1:
        raise ValueError("Full-subtask video must be nonempty [C,T,H,W]")
    count = video.shape[1]
    if count != result["full_subtask_frames"]:
        raise ValueError("Full-subtask RGB length differs from prepared target length")
    padding = (-(count - 1)) % 4
    if padding:
        result["video"] = torch.cat((video, video[:, -1:].expand(-1, padding, -1, -1)), dim=1)
    result["video_padding_frames"] = padding
    valid = torch.arange(count + padding, device=video.device) < count
    result["video_valid_mask"] = valid
    result["video_latent_real_frame_counts"] = torch.cat((valid[:1].long(), valid[1:].reshape(-1, 4).sum(dim=1)))
    result["video_padding_loss_policy"] = "retain_partial_latent_v1"
    return result


class ServingDataset(Dataset):
    """Compile text and visibility before either tokenizer sees a sample (WAM, joint delta_from_q0)."""

    def __init__(
        self,
        dataset,
        mode,
        *,
        resolution="480",
        max_action_dim=64,
        tokenizer_config=None,
        cfg_dropout_rate=0.1,
        allow_demo_derived=False,
        action_statistics=None,
        max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
        prompt_format="text",
    ):
        from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

        if mode != "wam":
            raise NotImplementedError(f"the serving compiler covers the WAM mode only, not {mode!r}")
        self.dataset, self.mode = dataset, mode
        self.resolution = resolution
        self.allow_demo_derived = allow_demo_derived
        self.action_statistics = action_statistics
        self.max_text_tokens = validate_max_text_tokens(max_text_tokens)
        from greenwam.data.prompt_json import PromptJsonFormatter, validate_prompt_format

        self.prompt_format = validate_prompt_format(prompt_format)
        structured = self.prompt_format == "json"
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            # prepare_mode_sample already replaced inverse targets with a safe
            # representation-specific instruction. Keep it and the layout.
            max_action_dim=max_action_dim,
            enable_mode_specific_prompt=mode != "inverse",
            max_text_tokens=self.max_text_tokens,
            text_overflow_policy="error",
            append_viewpoint_info=False,
            append_duration_fps_timestamps=structured,
            append_resolution_info=False,
            append_idle_frames=False,
            # "json": the Cosmos 3 structured prompt the Edge base was trained on (see prompt_json.py);
            # "text": the plain caption (canvas sentence + task + mode sentence).
            format_prompt_as_json=structured,
        )
        if structured:
            self.transform.prompt_json_formatter = PromptJsonFormatter(
                caption_key="ai_caption", append_duration_fps_timestamps=True, float_seconds=False
            )
        from greenwam.serving.canvas import CanvasVideoResize

        self.transform.video_resize = CanvasVideoResize(self.transform.video_resize)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from cosmos_framework.data.generator.sequence_packing import SequencePlan

        sample = dict(self.dataset[index])
        from greenwam.data.wam_outcomes import successful

        if not successful(sample.get("greenwam_wam_outcome"), sample.get("atom_id")):
            raise ValueError("WAM requires a verified successful demo outcome")
        sample = prepare_mode_sample(sample, self.mode, allow_demo_derived=self.allow_demo_derived)
        sample = pad_full_subtask_video(sample)
        layout_caption = sample.get("greenwam_canvas", {}).get("layout_caption")
        real_video_length = sample.get("full_subtask_frames", sample["video"].shape[1])
        if self.prompt_format == "json":
            from greenwam.data.prompt_json import robot_name

            # the canvas sentence becomes the framing (concat_view + layout caption), the robot its own field,
            # and duration/fps come from the measured frames, not the VAE-padded video
            sample["greenwam_robot"] = robot_name(sample["layout_id"])
            sample["video_num_frames"] = int(real_video_length)
        elif layout_caption:
            sample["ai_caption"] = f"{layout_caption} {sample['ai_caption']}"
        is_delta = (
            self.mode not in VISUAL_MODES
            and sample.get("greenwam_action_representation") == "joint"
            and sample.get("greenwam_joint_encoding", "absolute") == "delta_from_q0"
        )
        spec = mode_spec(
            self.mode,
            video_length=real_video_length,
            action_length=sample["action"].shape[0] if "action" in sample else 0,
        )
        if sample.get("action_target_scope") == "initial_and_future":
            if self.mode != "inverse" or is_delta:
                raise ValueError("Initial action target requires absolute inverse")
            spec["action_start_frame_offset"] = 0
        # The public ActionSFTDataset wrapper does not forward this mask.
        # It must reach ActionProcessor explicitly before noise/projection.
        normalizer = self.action_statistics.for_sample(sample) if self.action_statistics is not None else None
        if (
            self.mode == "wam"
            and sample.get("greenwam_action_representation") == "eef"
            and sample.get("eef_encoding") == "relative"
        ) or not is_delta:
            raise NotImplementedError("the serving compiler covers joint delta_from_q0 actions only")
        from greenwam.data.joint_deltas import prepare_q0_prefix, q0_conditioned_mode_spec

        spec = q0_conditioned_mode_spec(
            self.mode, video_length=real_video_length, action_length=sample["action"].shape[0]
        )
        raw, normalized, valid, codec = prepare_q0_prefix(sample, normalizer=normalizer)
        sample["action"] = raw
        sample = self.transform(
            sample,
            self.resolution,
            action_normalizer=None,
            action_valid_mask=valid,
            normalized_action=normalized,
            action_codec=codec,
        )
        sample["sequence_plan"] = SequencePlan(**spec)
        return sample
