# SPDX-License-Identifier: OpenMDW-1.1
"""ZMQ policy server for the RoboTwin 2.0 WAM post-train, on the serving copies (`greenwam/serving/`).

Speaks the msgpack REQ/REP protocol the RoboTwin eval client already uses (same shape as the Cosmos3-Edge
OBM server and as the dual-fps GreenWAM server) and answers with **absolute** qpos14 targets.

The serving contract mirrors the training recipe of the RoboTwin post-train exactly, because anything else
measures a different model than the one that was trained:

- canvas 640x640, head on top, Left | Right below, panels stretched (`robotwin_canvas`);
- physical clock 50/3 Hz, WAM horizon 1.0666667 s -> 17 future states; the RGB window is whatever the
  run's own config says the WAM reader produced (`wam_video_frames`), because the action head was trained
  beside a video of that length;
- `delta_from_q0` joints with the absolute gripper, q0 as prefix, statistics of the training release;
- no canvas release, so the compiled sample carries no `greenwam_canvas` block — exactly like the train reader.
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
from pathlib import Path
from typing import Any

from cosmos_framework.inference.common.init import init_script

init_script()

import numpy as np
import torch
import zmq

from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from greenwam.data.joint_deltas import (
    Q0_CONDITIONING_SCHEMA,
    Q0_STATE_CODEC,
    joint_mask_from_adapter,
)
from greenwam.data.prompt_json import robot_name
from greenwam.data.semantic_space import semantic_layout_id
from greenwam.data.shared_projectors import load_shared_projectors
from greenwam.data.wam_temporal import wam_future_frames
from greenwam.infer.robotwin_canvas import CANVAS_HEIGHT, CANVAS_WIDTH, compose_head_left_right
from greenwam.serving.action_statistics import ActionStatistics
from greenwam.serving.domains import resolve_domain_id
from greenwam.serving.sample import ServingDataset, wam_video_frames

_NDARRAY_KEY = "__ndarray__"
LAYOUT_ID = "robotwin2__aloha_agilex__14__14"
SOURCE = "RoboTwin2"
FPS = 50 / 3  # RoboTwin writes one frame per save_freq=15 simulation steps of 1/250 s
WAM_HORIZON_S = 1.0666666666666667
N_FUTURE = wam_future_frames(WAM_HORIZON_S, FPS)  # 17 future states at the physical clock
RAW_STATE_DIM = 14


def _pack_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_KEY: True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
    raise TypeError(f"Unsupported msgpack type: {type(obj)}")


def _unpack_ndarray(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_NDARRAY_KEY):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def pack_message(data: dict) -> bytes:
    import msgpack

    return msgpack.packb(data, default=_pack_ndarray, use_bin_type=True)


def unpack_message(payload: bytes) -> dict:
    import msgpack

    value = msgpack.unpackb(payload, raw=False, object_hook=_unpack_ndarray)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict message, got {type(value)}")
    return value


def _tokenizer_config(model) -> Any:
    vlm = getattr(getattr(model, "config", None), "vlm_config", None)
    tokenizer = getattr(vlm, "tokenizer", None) if vlm is not None else None
    return tokenizer.model_dump() if tokenizer is not None and hasattr(tokenizer, "model_dump") else tokenizer


class RobotwinPolicy:
    """One WAM draw per request: canvas + q0 + instruction -> 17 absolute qpos14 targets."""

    def __init__(self, args: argparse.Namespace) -> None:
        maybe_init_distributed()
        self.args = args
        self.n_video, self.n_action = wam_video_frames(args.config_file, FPS), int(args.pred_len)
        if self.n_action != N_FUTURE:
            raise ValueError(f"pred_len={self.n_action}; the train contract is {N_FUTURE} at {FPS:.4f} Hz")
        log.info(
            f"[robotwin] WAM window from {args.config_file}: {self.n_video} RGB frames, {self.n_action} action steps"
        )
        setup = OmniSetupOverrides.model_validate(
            {
                "checkpoint_path": args.checkpoint_path,
                "output_dir": args.output_dir,
                "config_file": args.config_file,
                "use_ema_weights": bool(args.use_ema_weights),
                "dp_shard_size": 1,
                "dp_replicate_size": 1,
                "use_torch_compile": False,
                "sampler": "unipc",
                "guardrails": False,
            }
        ).build_setup()
        init_output_dir(setup.output_dir)
        setup = disable_runtime_ema_for_frozen_config(setup)
        log.info(f"[robotwin] loading ckpt={args.checkpoint_path} config={args.config_file} ema={args.use_ema_weights}")
        self.model = OmniInference.create(setup).model
        self.model.eval()
        probe = self.model.net.get_parameter(
            "time_embedder.mlp.2.weight"
        )  # trained weight: tells net.* from net_ema.* in the log
        probe = probe.full_tensor() if hasattr(probe, "full_tensor") else probe
        log.info(f"[robotwin] loaded |net.time_embedder.mlp.2.weight| = {probe.float().norm().item():.6f}")
        self.adapter = load_shared_projectors().resolve(LAYOUT_ID)
        layout_int = semantic_layout_id(LAYOUT_ID)
        if layout_int is None:
            raise RuntimeError(f"{LAYOUT_ID} has no semantic layout id; the extension registry is not attached")
        self.layout_id_int = int(layout_int)
        self.stats = ActionStatistics(Path(args.stats))
        self.joint_mask = joint_mask_from_adapter(self.adapter)
        self.max_action_dim = int(getattr(self.model.config, "max_action_dim", 128) or 128)
        self.tokenizer_config = _tokenizer_config(self.model)
        self.domain_id = resolve_domain_id(LAYOUT_ID, SOURCE)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(int(args.seed))
        self._gate_normalization()
        log.info(
            f"[robotwin] ready layout={LAYOUT_ID} layout_int={self.layout_id_int} "
            f"contract={self.adapter.action_contract_id}/{self.adapter.action_contract} "
            f"family_dim={self.adapter.target_dim} canvas={CANVAS_HEIGHT}x{CANVAS_WIDTH} "
            f"frames={self.n_video} targets={self.n_action} fps={FPS:.4f}"
        )

    def _gate_normalization(self) -> None:
        """Refuse to serve unless the q0 codec and the release statistics are actually in the sample."""
        probe = {
            "prompt": "normalization gate",
            "state": np.zeros(RAW_STATE_DIM, dtype=np.float32),
            "rgb_head": np.zeros((240, 320, 3), dtype=np.uint8),
            "rgb_left_wrist": np.zeros((240, 320, 3), dtype=np.uint8),
            "rgb_right_wrist": np.zeros((240, 320, 3), dtype=np.uint8),
        }
        normalizer = self.stats.for_sample(self._raw_sample(probe))
        offset = torch.as_tensor(normalizer.offset, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(normalizer.scale, dtype=torch.float32).reshape(-1)
        if torch.allclose(offset, torch.zeros_like(offset)) and torch.allclose(scale, torch.ones_like(scale)):
            raise RuntimeError(f"action statistics resolved to identity 0/1 ({self.args.stats})")
        if int(offset.numel()) != int(self.adapter.target_dim):
            raise RuntimeError(f"stats dim {int(offset.numel())} != family dim {self.adapter.target_dim}")
        compiled = self._compile(self._raw_sample(probe))
        record = compiled.get("action_processing_record")
        if record is None or getattr(record, "action_codec", None) is None:
            raise RuntimeError("compiled WAM sample has no q0 action codec; futures would stay z-scored")
        if getattr(record, "action_normalizer", None) is not None:
            raise RuntimeError("compiled sample has both a normalizer and a codec; decode would run twice")
        if getattr(record.action_codec, "future_normalizer", None) is None:
            raise RuntimeError("q0 codec has no future normalizer")
        video = compiled["video"] if "video" in compiled else None
        log.info(
            f"[robotwin] norm_gate=OK codec={type(record.action_codec).__name__} "
            f"stats_dim={int(offset.numel())} video={tuple(video.shape) if video is not None else None} "
            f"plan={compiled['sequence_plan'].condition_frame_indexes_action}"
        )

    def _compile(self, raw: dict[str, Any]) -> dict[str, Any]:
        return ServingDataset(
            [raw],
            "wam",
            resolution="480",
            max_action_dim=self.max_action_dim,
            tokenizer_config=self.tokenizer_config,
            cfg_dropout_rate=0.0,
            action_statistics=self.stats,
            prompt_format="json",
            max_text_tokens=768,
        )[0]

    def _raw_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        canvas = compose_head_left_right(obs["rgb_head"], obs["rgb_left_wrist"], obs["rgb_right_wrist"])
        video = np.zeros((3, self.n_video, CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)
        video[:, 0] = np.transpose(canvas, (2, 0, 1))
        state = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
        if state.size < RAW_STATE_DIM or not np.isfinite(state[:RAW_STATE_DIM]).all():
            raise ValueError(f"q0 must be a finite {RAW_STATE_DIM}D state, got {state.shape}")
        adapted_q0 = self.adapter.adapt(
            torch.from_numpy(np.ascontiguousarray(state[:RAW_STATE_DIM])).reshape(1, RAW_STATE_DIM)
        )
        family_dim = int(self.adapter.target_dim)
        availability = adapted_q0.availability_mask.to(dtype=torch.bool)
        text = prompt.strip()
        sample = {
            "video": torch.from_numpy(video),
            "action": torch.zeros((self.n_action, family_dim), dtype=torch.float32),
            "action_valid_mask": availability.unsqueeze(0).expand(self.n_action, -1).clone(),
            "action_availability_mask": availability,
            "action_q0": adapted_q0.action,
            "action_q0_valid_mask": adapted_q0.valid_mask,
            "action_joint_mask": self.joint_mask,
            "action_contract": self.adapter.action_contract,
            "action_contract_id": torch.tensor(int(self.adapter.action_contract_id), dtype=torch.long),
            "action_layout_id": torch.tensor(int(self.layout_id_int), dtype=torch.long),
            "layout_id": LAYOUT_ID,
            "raw_action_dim": torch.tensor(family_dim, dtype=torch.long),
            "greenwam_action_representation": "joint",
            "greenwam_joint_encoding": "delta_from_q0",
            "greenwam_state_conditioning": Q0_CONDITIONING_SCHEMA,
            "greenwam_q0_codec": Q0_STATE_CODEC,
            "greenwam_robot": robot_name(LAYOUT_ID),
            "full_subtask_frames": self.n_video,
            "greenwam_prompts": {
                "visual": text,
                "forward": text,
                "wam": {"text": text, "available_before_rollout": True, "provenance": "source_task"},
                "policy": {"text": text, "available_before_rollout": True, "provenance": "source_task"},
            },
            "greenwam_wam_outcome": {
                "outcome": "success",
                "outcome_scope": "source_episode",
                "evidence_ref": "live_eval",
                "ledger_sha256": "0" * 64,
            },
            "atom_id": "live:robotwin_canvas",
            "conditioning_fps": torch.tensor(FPS, dtype=torch.float32),
            "conditioning_fps_action": torch.tensor(FPS, dtype=torch.float32),
            "domain_id": torch.tensor(int(self.domain_id), dtype=torch.long),
        }
        return sample  # ServingDataset pads to the VAE boundary itself (pad_full_subtask_video)

    def _next_seed(self) -> int:
        return int(self.args.seed) if self.args.deterministic_seed else int(self._rng.integers(0, 2**31))

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = {key: value for key, value in obs.items() if not str(key).startswith("_debug")}
        horizon = int(request.pop("prediction_horizon", self.n_action))
        if horizon != self.n_action:
            raise ValueError(f"prediction_horizon={horizon}; this server serves {self.n_action}")
        guidance = float(request.pop("guidance", 1.0))
        num_steps = int(request.pop("num_steps", 8))
        shift = float(request.pop("shift", 5.0))
        for ignored in ("decode_video", "video_text_cfg", "action_text_cfg"):
            request.pop(ignored, None)
        batch = custom_collate_fn([self._compile(self._raw_sample(request))])
        if getattr(self.model, "input_video_key", "video") != "video":
            batch[self.model.input_video_key] = batch.pop("video")
        if getattr(self.model, "input_caption_key", "ai_caption") != "ai_caption":
            batch[self.model.input_caption_key] = batch.pop("ai_caption")
        seed = self._next_seed()
        with self._lock, torch.inference_mode():
            outputs = self.model.generate_samples_from_batch(
                batch, seed=[seed], n_sample=1, num_steps=num_steps, guidance=guidance, shift=shift
            )
        physical = outputs["action"][0]
        compiled_plan = batch.get("sequence_plan")
        plan = compiled_plan[0] if isinstance(compiled_plan, (list, tuple)) else compiled_plan
        if getattr(plan, "condition_frame_indexes_action", None) == [0]:
            physical = physical[1:]  # drop the q0 prefix the codec conditioned on
        family_dim = int(self.adapter.target_dim)
        if physical.ndim != 2 or physical.shape[0] < self.n_action:
            raise ValueError(f"model returned {tuple(physical.shape)}, need at least {self.n_action} steps")
        source = self.adapter.unadapt(
            physical[: self.n_action, :family_dim], valid_mask=torch.ones((self.n_action, family_dim), dtype=torch.bool)
        ).action
        actions = source.detach().cpu().numpy().astype(np.float32)
        if actions.shape != (self.n_action, RAW_STATE_DIM):
            raise ValueError(f"unadapt produced {actions.shape}, expected {(self.n_action, RAW_STATE_DIM)}")
        log.info(
            f"[robotwin] seed={seed} steps={num_steps} guidance={guidance} shift={shift} "
            f"prompt={str(request.get('prompt'))[:60]!r}"
        )
        return {"actions": actions, "prediction_horizon": self.n_action, "action_space": "absolute_qpos14", "fps": FPS}


def serve(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    policy = RobotwinPolicy(args)
    address = f"tcp://{args.host}:{int(args.port)}"
    context = zmq.Context.instance()
    server = context.socket(zmq.REP)
    server.bind(address)
    log.info(f"[robotwin] serving at {address} host={socket.gethostname()}")
    while True:
        payload = server.recv()
        try:
            server.send(pack_message(policy.infer(unpack_message(payload))))
        except Exception as exc:  # one bad request must not take the server down mid-evaluation
            log.exception("[robotwin] infer failed")
            server.send(pack_message({"error": str(exc)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--stats", required=True, help="delta_from_q0/statistics.json of the training release")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--pred-len", type=int, default=N_FUTURE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--use-ema-weights", action="store_true", default=True)
    parser.add_argument("--no-ema-weights", action="store_false", dest="use_ema_weights")
    parser.add_argument("--output-dir", default=str(DEFAULT_FALLBACK_OUTPUT_DIR / "robotwin_canvas"))
    args = parser.parse_args()
    if "WAN_VAE_PATH" not in os.environ:
        raise SystemExit("set WAN_VAE_PATH to the Wan2.2 VAE used by the run")
    serve(args)


if __name__ == "__main__":
    main()
