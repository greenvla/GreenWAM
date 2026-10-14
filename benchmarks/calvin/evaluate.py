# SPDX-License-Identifier: OpenMDW-1.1
"""CALVIN ABC->D evaluation of a GreenWAM policy server with the official multistep loop.

Runs the canonical evaluation sequences of ``calvin_agent.evaluation.multistep_sequences.get_sequences(1000)`` (or
the slice ``[--seq-start, --seq-end)`` of them, for sharding) through the official ``evaluate_sequence`` / ``rollout``:
5 chained subtasks, at most 360 steps each, success decided by the CALVIN task oracle. Writes ``result.json``
(chain length, per-level and per-subtask success) and ``sequences.json`` (one record per sequence) to
``--output-dir``; ``aggregate.py`` merges shards.

Example (one shard of 42 sequences, Mesa llvmpipe as EGL device 16):

    python -m benchmarks.calvin.evaluate --dataset-path <calvin>/task_D_D/validation --server-port 8200 \
        --seq-start 0 --seq-end 42 --egl-device 16 --output-dir out/shard00
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-path", required=True, help="CALVIN split directory with .hydra/merged_config.yaml")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, required=True)
    p.add_argument("--num-sequences", type=int, default=1000, help="size of the canonical sequence list")
    p.add_argument("--seq-start", type=int, default=0)
    p.add_argument("--seq-end", type=int, default=None)
    p.add_argument("--prediction-horizon", type=int, default=20)
    p.add_argument("--execution-horizon", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--video-text-cfg", type=float, default=1.5)
    p.add_argument("--action-text-cfg", type=float, default=1.5)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--egl-device", default=None, help="EGL device index for rendering (sets EGL_VISIBLE_DEVICES)")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _subtask_outcomes(sequence: tuple[str, ...], completed: int) -> dict[str, bool]:
    """Attempted subtasks of one chain: the first ``completed`` succeeded, the next one (if any) failed.

    Keyed by subtask name: none of the 1000 evaluation chains of ``get_sequences`` repeats a subtask.
    """
    outcomes: dict[str, bool] = {}
    for index, subtask in enumerate(sequence[: min(completed + 1, len(sequence))]):
        outcomes[subtask] = index < completed
    return outcomes


def main() -> None:
    # Seed before anything touches an RNG, as the official evaluation does.
    from pytorch_lightning import seed_everything

    seed_everything(0, workers=True)
    args = _parse_args()
    if args.egl_device is not None:
        os.environ["EGL_VISIBLE_DEVICES"] = str(args.egl_device)

    from benchmarks.calvin import pyhash_compat

    pyhash_compat.install()

    from collections import defaultdict

    import calvin_agent
    import hydra
    from calvin_agent.evaluation.evaluate_policy import evaluate_sequence
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import count_success
    from calvin_env.envs.play_table_env import get_env
    from omegaconf import OmegaConf

    from benchmarks.calvin.policy import PolicyClient, ServerPolicy

    model = ServerPolicy(
        PolicyClient(args.server_host, args.server_port),
        prediction_horizon=args.prediction_horizon,
        execution_horizon=args.execution_horizon,
        guidance=args.guidance,
        video_text_cfg=args.video_text_cfg,
        action_text_cfg=args.action_text_cfg,
        num_steps=args.num_steps,
        shift=args.shift,
    )
    env = get_env(
        Path(args.dataset_path),
        show_gui=False,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
    )
    conf_dir = Path(calvin_agent.__file__).resolve().parents[1] / "conf"
    task_oracle = hydra.utils.instantiate(OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    sequences = list(get_sequences(args.num_sequences))
    seq_end = len(sequences) if args.seq_end is None else args.seq_end
    plans: dict = defaultdict(list)
    records, results = [], []
    subtasks: dict[str, list[int]] = {}
    for index in range(args.seq_start, seq_end):
        initial_state, sequence = sequences[index]
        completed = evaluate_sequence(env, model, task_oracle, initial_state, sequence, val_annotations, plans, False)
        results.append(completed)
        for subtask, success in _subtask_outcomes(sequence, completed).items():
            counts = subtasks.setdefault(subtask, [0, 0])
            counts[0] += int(success)
            counts[1] += 1
        records.append({"index": index, "sequence": list(sequence), "completed": completed})

    n = len(results)
    levels = count_success(results)
    result = {
        "benchmark": "calvin",
        "total_sequences": n,
        "completed_subtasks": sum(results),
        "average_chain_length": round(sum(results) / n, 2),
        "average_chain_length_exact": sum(results) / n,
        "average_success_rate_percent": f"{100 * sum(results) / model.num_rollouts:.2f}%",
        "average_episode_length": f"{model.num_steps_taken / model.num_rollouts:.2f}",
        "levels": {f"{i + 1}": f"{rate * 100:.2f}%" for i, rate in enumerate(levels)},
        "subtasks": {k: f"{100 * s / t:.2f}% ({s}/{t})" for k, (s, t) in subtasks.items()},
        "run_config": {k: v for k, v in vars(args).items() if k not in ("output_dir",)},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "sequences.json").write_text(json.dumps(records))
    print(json.dumps({k: result[k] for k in ("total_sequences", "average_chain_length_exact", "levels")}))


if __name__ == "__main__":
    main()
