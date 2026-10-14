# SPDX-License-Identifier: OpenMDW-1.1
"""ZMQ policy server for the CALVIN ABC->D GreenWAM policy, on the serving copies (`greenwam/serving/`).

Speaks the msgpack REQ/REP protocol of `benchmarks/calvin`: one request per action chunk with the instruction
(``prompt``), the static and gripper RGB frames as rendered (``rgb_static`` 200x200, ``rgb_gripper`` 84x84), the
15-D ``robot_obs`` (``state``) and the sampler settings (``prediction_horizon``, ``guidance``, ``num_steps``,
``shift``); the reply carries ``prediction_horizon`` absolute TCP targets (x, y, z, roll, pitch, yaw, gripper in
{-1, 1}) plus a zero progress column, ``[H, 8]`` float32.

The serving contract is the one the policy was trained and evaluated with:

- canvas 384x288: the static camera (stored size 192x192) stretched to 256x288 on top, the gripper camera to 128x288
  below, the GreenWAM JSON prompt with the robot and canvas descriptions;
- action layout ``franka_eef_v1`` (44): the q0 prefix row in the absolute contract 90, the H future rows framewise
  relative in contract 76, normalized with the statistics of the training data (``--stats``);
- clock 30 Hz; horizons 4..32 in steps of 4 (the VAE encodes H+1 frames);
- the numeric switches the run needs in the model config (``RECIPE_SWITCHES``); the server refuses to serve without
  them;
- the seed of every request is drawn from ``numpy.random.default_rng(--seed)``, or is ``--seed`` with
  ``--deterministic-seed``.

The text tokenizer is the Cosmos3-Edge processor of the config; ``COSMOS3_EDGE_DIR`` points it at a local snapshot.
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
import yaml
import zmq

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from greenwam.data.semantic_space import SEMANTIC_EXTENSIONS_ENV, SemanticSpace
from greenwam.serving.calvin import (
    CALVIN_ABS_CONTRACT_ID,
    CALVIN_LAYOUT_ID,
    CALVIN_REL_CONTRACT_ID,
    build_calvin_sample,
    load_calvin_codec,
)
from greenwam.serving.policy_prompt import PolicySampleTransform, get_domain_id, policy_data_batch

_NDARRAY_KEY = "__ndarray__"
LAYOUT_NAME = "franka_eef_v1"
FPS = 30.0
PREDICTION_HORIZONS = (4, 8, 12, 16, 20, 24, 28, 32)
RECIPE_SWITCHES = {
    "inference_noise_source": "torch_cuda",
    "inference_reuse_request_cache": False,
    "time_embedding_per_token": True,
    "pad_segment_only_when_padded": True,
    "normalize_video_in_model_dtype": True,
}


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


def served_config_file(config_file: str, output_dir: str) -> str:
    """The model config, with the text tokenizer pointed at ``COSMOS3_EDGE_DIR`` when it is set."""
    edge_dir = os.environ.get("COSMOS3_EDGE_DIR")
    if not edge_dir:
        return config_file
    doc = yaml.safe_load(Path(config_file).read_text())
    node = doc["model"]["config"]["vlm_config"]["tokenizer"]
    for key in ("repository", "revision", "subdir"):
        node.pop(key, None)
    node["tokenizer_type"] = str(edge_dir)
    out = Path(output_dir) / "served_config.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(out)


class CalvinPolicy:
    """One policy draw per request: canvas + robot_obs + instruction -> H cartesian_abs targets."""

    def __init__(self, args: argparse.Namespace) -> None:
        maybe_init_distributed()
        self.args = args
        layout = SemanticSpace.load().layouts.get(LAYOUT_NAME)
        if layout != CALVIN_LAYOUT_ID:
            raise RuntimeError(
                f"layout {LAYOUT_NAME} resolves to {layout}, not {CALVIN_LAYOUT_ID}: set "
                f"{SEMANTIC_EXTENSIONS_ENV} to benchmarks/semantic_extensions/franka_eef_v1.json"
            )
        self.codec = load_calvin_codec(args.stats)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        setup = OmniSetupOverrides.model_validate(
            {
                "checkpoint_path": args.checkpoint_path,
                "output_dir": args.output_dir,
                "config_file": served_config_file(args.config_file, args.output_dir),
                "use_ema_weights": True,  # the reported policy is the EMA of the run
                "dp_shard_size": 1,
                "dp_replicate_size": 1,
                "use_torch_compile": False,
                "sampler": "unipc",
                "guardrails": False,
            }
        ).build_setup()
        init_output_dir(setup.output_dir)
        setup = disable_runtime_ema_for_frozen_config(setup)
        log.info(f"[calvin] loading ckpt={args.checkpoint_path} config={args.config_file}")
        self.model = OmniInference.create(setup).model
        self.model.eval()
        self._gate_recipe()
        self.transform = PolicySampleTransform(
            tokenizer_config=_tokenizer_config(self.model),
            max_action_dim=int(getattr(self.model.config, "max_action_dim", 128) or 128),
        )
        self.domain_id = get_domain_id("calvin")
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(int(args.seed))
        log.info(
            f"[calvin] ready layout={LAYOUT_NAME}/{CALVIN_LAYOUT_ID} contracts="
            f"{CALVIN_REL_CONTRACT_ID}/{CALVIN_ABS_CONTRACT_ID} fps={FPS} seed={args.seed} "
            f"deterministic_seed={args.deterministic_seed}"
        )

    def _gate_recipe(self) -> None:
        """Refuse to serve unless the model config carries the run's numeric switches."""
        found = {key: getattr(self.model.config, key, None) for key in RECIPE_SWITCHES}
        wrong = {key: value for key, value in found.items() if value != RECIPE_SWITCHES[key]}
        if wrong:
            raise RuntimeError(
                f"model config {self.args.config_file} lacks the CALVIN recipe switches: {wrong}, "
                f"expected {RECIPE_SWITCHES}"
            )

    def _next_seed(self) -> int:
        return int(self.args.seed) if self.args.deterministic_seed else int(self._rng.integers(0, 2**31))

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = {key: value for key, value in obs.items() if not str(key).startswith("_debug_")}
        if request.pop("observations", None) is not None:
            raise ValueError("batched observations are not served")
        horizon = request.pop("prediction_horizon", None)
        if horizon is None or int(horizon) not in PREDICTION_HORIZONS:
            raise ValueError(f"prediction_horizon={horizon} not in the trained choices {list(PREDICTION_HORIZONS)}")
        horizon = int(horizon)
        guidance = float(request.pop("guidance"))
        for key in ("video_text_cfg", "action_text_cfg"):  # one text guidance scale for both streams
            value = request.pop(key, None)
            if value is not None and float(value) != guidance:
                raise ValueError(f"{key}={value}: this model takes one text guidance scale, guidance={guidance}")
        num_steps = int(request.pop("num_steps"))
        shift = float(request.pop("shift"))
        for key in ("decode_video", "reuse_overlap_noise", "reset_overlap_noise", "overlap_action_count"):
            if request.pop(key, None):
                raise ValueError(f"{key} is not served")
        if not isinstance(request.get("prompt"), str):
            raise ValueError("'prompt' must be a string")
        sample = build_calvin_sample(
            request, horizon, self.codec, conditioning_fps=FPS, conditioning_fps_action=FPS, domain_id=self.domain_id
        )
        batch = policy_data_batch(
            [self.transform(sample)],
            action_contract_id=CALVIN_REL_CONTRACT_ID,
            action_layout_id=CALVIN_LAYOUT_ID,
            action_conditioning_contract_id=CALVIN_ABS_CONTRACT_ID,
        )
        seed = self._next_seed()
        with self._lock, torch.inference_mode():
            generated = self.model.generate_samples_from_batch(
                batch, guidance=guidance, seed=[seed], num_steps=num_steps, shift=shift
            )
        future = generated["action"][0][1 : horizon + 1]  # row 0 is the q0 prefix
        state = torch.as_tensor(np.asarray(request["state"], dtype=np.float32))
        cartesian = self.codec.decode(future, state)
        actions = np.concatenate([cartesian, np.zeros((cartesian.shape[0], 1), np.float32)], axis=1)
        log.info(
            f"[calvin] seed={seed} H={horizon} steps={num_steps} guidance={guidance} shift={shift} "
            f"prompt={request['prompt'][:60]!r}"
        )
        return {"actions": actions, "prediction_horizon": horizon}


def serve(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    policy = CalvinPolicy(args)
    address = f"tcp://{args.host}:{int(args.port)}"
    server = zmq.Context.instance().socket(zmq.REP)
    server.bind(address)
    log.info(f"[calvin] serving at {address} host={socket.gethostname()}")
    while True:
        payload = server.recv()
        try:
            server.send(pack_message(policy.infer(unpack_message(payload))))
        except Exception as exc:  # one bad request must not take the server down mid-evaluation
            log.exception("[calvin] infer failed")
            server.send(pack_message({"error": str(exc)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--stats", required=True, help="calvin_eef_statistics JSON of the training data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_FALLBACK_OUTPUT_DIR / "calvin"))
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
