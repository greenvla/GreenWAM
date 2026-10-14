# SPDX-License-Identifier: OpenMDW-1.1
"""HTTP policy server for the released LIBERO checkpoint (GreenWAM + action expert), on the serving copies (`greenwam/serving/`).

Speaks the protocol of the upstream LIBERO client `cosmos_framework/simulation/libero/closed_loop_eval.py`: `GET /`
health, `GET /info`, `POST /predict` -> `{"action": [[...]]}`, `POST /predict_batch` -> `{"actions": [...]}`,
`POST /next_episode`. Each request carries the agentview | wrist image (one 256 x 512 PNG) and, with the client's
`--send_state`, the 8-D LIBERO state. The reply is the chunk of relative rows (delta position, rotation-6d, gripper
open fraction) the client executes with `--action_space frame_wise_relative --rotation_space 6d --action_dim 10
--gripper_mode zero_one`.

The served sample is the training sample of the LIBERO reader: the GreenWAM canvas in frame 0 of a (chunk + 1)-frame clip,
the q0 row from the state, the bare task in the GreenWAM JSON prompt, policy mode at 20 Hz, the core routing of the Franka
EEF layout (future contract 76, layout 44, q0 conditioning contract 90).

    PYTHONPATH=. python -m greenwam.infer.libero_policy_server --checkpoint-path <model> --config-file <config.yaml> \\
        --stats <libero_stats.json> --port 8300
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cosmos_framework.inference.common.init import init_script

init_script()

import numpy as np
import torch
from PIL import Image

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from greenwam.serving.franka_eef import CALVIN_WIDTH
from greenwam.serving.libero import LIBERO_ROBOT, LiberoEEFCodec, LiberoStats, canvas_video, task_only
from greenwam.serving.policy_prompt import CANVAS_CAPTION, PolicySampleTransform, get_domain_id, policy_data_batch

ACTION_CONTRACT_ID, ACTION_LAYOUT_ID, ACTION_CONDITIONING_CONTRACT_ID = 76, 44, 90
STATE_DIM = 8


def decode_png(image_b64: str) -> np.ndarray:
    if "," in image_b64[:64]:
        image_b64 = image_b64.split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB"), dtype=np.uint8)


def tokenizer_config_of(model: Any) -> Any:
    vlm = getattr(getattr(model, "config", None), "vlm_config", None)
    tokenizer = getattr(vlm, "tokenizer", None) if vlm is not None else None
    return tokenizer.model_dump() if tokenizer is not None and hasattr(tokenizer, "model_dump") else tokenizer


class LiberoRequestCompiler:
    """One client request -> one model sample (the LIBERO reader's training sample, without CFG dropout)."""

    def __init__(
        self, tokenizer_config: Any, stats_path: str, chunk: int = 32, fps: float = 20.0, max_action_dim: int = 128
    ) -> None:
        self.codec = LiberoEEFCodec(LiberoStats(stats_path))
        self.transform = PolicySampleTransform(tokenizer_config, max_action_dim)
        self.chunk, self.fps = int(chunk), float(fps)
        self.domain_id = get_domain_id("libero")

    def sample(self, req: dict[str, Any]) -> dict[str, Any]:
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("'prompt' must be a non-empty string")
        state = req.get("state")
        if state is None or len(state) != STATE_DIM:
            raise ValueError(f"'state' ({STATE_DIM} floats) is required: run the client with --send_state")
        image = decode_png(req["image"])  # [256, 512, 3]: agentview | wrist
        if image.shape[1] != 2 * image.shape[0]:
            raise ValueError(f"expected the agentview | wrist concat (H x 2H), got {image.shape}")
        action, valid = self.codec.encode_inference(torch.tensor(state, dtype=torch.float32), self.chunk)
        fps = torch.tensor(self.fps, dtype=torch.float32)
        return self.transform(
            {
                "ai_caption": task_only(prompt),
                "video": canvas_video(torch.from_numpy(image.copy()), self.chunk),
                "action": action,
                "action_valid_mask": valid,
                "conditioning_fps": fps,
                "conditioning_fps_action": fps,
                "mode": "policy",
                "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
                "viewpoint": "concat_view",
                "additional_view_description": CANVAS_CAPTION,
                "greenwam_robot": LIBERO_ROBOT,
            }
        )

    def batch(self, reqs: list[dict[str, Any]]) -> dict[str, Any]:
        return policy_data_batch(
            [self.sample(r) for r in reqs],
            action_contract_id=ACTION_CONTRACT_ID,
            action_layout_id=ACTION_LAYOUT_ID,
            action_conditioning_contract_id=ACTION_CONDITIONING_CONTRACT_ID,
        )


class LiberoPolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        maybe_init_distributed()
        self.args = args
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
        log.info(f"[libero] loading ckpt={args.checkpoint_path} config={args.config_file} ema={args.use_ema_weights}")
        self.model = OmniInference.create(setup).model
        self.model.eval()
        self.compiler = LiberoRequestCompiler(
            tokenizer_config_of(self.model),
            args.stats,
            chunk=args.chunk,
            fps=args.fps,
            max_action_dim=int(getattr(self.model.config, "max_action_dim", 128) or 128),
        )
        self._rng = np.random.default_rng(int(args.seed))
        log.info(
            f"[libero] ready guidance={args.guidance} steps={args.num_steps} shift={args.shift} "
            f"chunk={args.chunk} fps={args.fps} deterministic_seed={args.deterministic_seed} "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
        )

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.args.checkpoint_path,
            "action_space": "frame_wise_relative",
            "rotation_space": "6d",
            "action_dim": CALVIN_WIDTH,
            "gripper_mode": "zero_one",
            "needs_state": True,
            "action_chunk_size": self.args.chunk,
            "guidance": self.args.guidance,
            "num_steps": self.args.num_steps,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

    def _seed(self) -> int:
        return int(self.args.seed) if self.args.deterministic_seed else int(self._rng.integers(0, 2**31))

    def predict_batch(self, reqs: list[dict[str, Any]]) -> list[list[list[float]]]:
        batch = self.compiler.batch(reqs)
        with torch.inference_mode():
            seeds = [self._seed() for _ in reqs]
            samples = self.model.generate_samples_from_batch(
                batch, guidance=self.args.guidance, seed=seeds, num_steps=self.args.num_steps, shift=self.args.shift
            )
        # row 0 is the q0 prefix the codec conditioned on
        return [
            self.compiler.codec.relative_physical(samples["action"][i][1 : self.args.chunk + 1]).tolist()
            for i in range(len(reqs))
        ]


class InferenceThread:
    """Runs every model call on one thread. PyTorch keeps the cuDNN autotuning cache per thread
    (aten/src/ATen/native/cudnn/Conv_v8.cpp), so generating on the HTTP threads would re-tune, and possibly re-choose,
    the VAE's convolution algorithms on every request; the HTTP threads only parse, wait and answer, so health and
    info requests are served while a generation runs. Generation is serial either way."""

    def __init__(self, fn) -> None:
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn) -> None:
        while True:
            args, done, out = self._jobs.get()
            try:
                out.append(fn(*args))
            except Exception as exc:  # handed back to the waiting request
                out.append(exc)
            done.set()

    def __call__(self, *args):
        done, out = threading.Event(), []
        self._jobs.put((args, done, out))
        done.wait()
        if isinstance(out[0], Exception):
            raise out[0]
        return out[0]


class _Handler(BaseHTTPRequestHandler):
    policy: LiberoPolicy
    infer: InferenceThread

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path == "":
            self._send(200, {"status": "ok"})
        elif path == "/info":
            self._send(200, self.policy.info())
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            path = self.path.rstrip("/")
            if path == "/predict":
                self._send(200, {"action": self.infer([req])[0]})
            elif path == "/predict_batch":
                self._send(200, {"actions": self.infer(list(req["items"]))})
            elif path == "/next_episode":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except Exception as exc:  # one bad request must not take the server down mid-evaluation
            log.exception("[libero] request failed")
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--stats", required=True, help="libero_stats.json of the release")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--chunk", type=int, default=32, help="predicted rows per request")
    parser.add_argument("--fps", type=float, default=20.0, help="conditioning fps of the training data")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="seeds the generator that draws each request's seed")
    parser.add_argument("--deterministic-seed", action="store_true", help="sample every request with --seed itself")
    parser.add_argument("--use-ema-weights", action="store_true", default=True)
    parser.add_argument("--no-ema-weights", action="store_false", dest="use_ema_weights")
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cuDNN autotuning (default on): the reported evaluation's server loaded the model through its experiment "
        "config, whose trainer.cudnn.benchmark=True set it; the VAE encoding of the conditioning frame depends on it",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_FALLBACK_OUTPUT_DIR / "libero"))
    args = parser.parse_args()
    if "WAN_VAE_PATH" not in os.environ:
        raise SystemExit("set WAN_VAE_PATH to the Wan2.2 VAE used by the run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)  # before the model is built, as the original loader
    _Handler.policy = LiberoPolicy(args)
    _Handler.infer = InferenceThread(_Handler.policy.predict_batch)
    server = ThreadingHTTPServer((args.host, int(args.port)), _Handler)
    log.info(f"[libero] serving on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
