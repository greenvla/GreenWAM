#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Closed-loop RoboTwin 2.0 evaluation of the GreenWAM RoboTwin policy server.

Follows the official evaluation path (``scripts/eval_policy_xpolicylab.py`` of RoboTwin 2.0): evaluation seeds start
at ``100000 * (1 + seed)``, each candidate seed is first checked with the expert (a seed the expert cannot solve is
skipped, not counted as a policy failure), the instruction comes from the episode's generated descriptions, and the
episode ends on success or at the task's step limit.

One request per replan: the server gets the three D435 views and the current drive targets (the quantity the
training data recorded as ``state``) and answers with absolute qpos14 targets; the first ``--chunk`` of them are
executed through ``take_action``.

Every evaluated episode and every skipped seed is printed as one JSON line and appended to
``<out>/records.jsonl``; ``tally.py`` reads these records. Run inside the simulator environment with
``ROBOTWIN_ROOT`` set to the RoboTwin 2.0 checkout.

Example (one task, the reported policy settings):

    python benchmarks/robotwin/evaluate.py --tasks adjust_bottle --config demo_clean --episodes 100 --chunk 16 \
        --skip-seeds benchmarks/robotwin/skip_seeds.txt --server tcp://127.0.0.1:8101 --out out/clean/adjust_bottle
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np

CAMERAS = {"rgb_head": "head_camera", "rgb_left_wrist": "left_camera", "rgb_right_wrist": "right_camera"}


def env_args(base: dict, task_name: str, out: Path, collect_module, embodiments: dict) -> dict:
    robot_file = embodiments[base["embodiment"][0]]["file_path"]
    args = dict(base)
    # eval_mode=True is what the official evaluator sets: it arms the per-task step limit
    # (env_cfg/task_config/_eval_step_limit.yml) and switches randomized textures to the unseen pool.
    args.update(
        task_name=task_name,
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=collect_module.get_embodiment_config(robot_file),
        right_embodiment_config=collect_module.get_embodiment_config(robot_file),
        collect_data=False,
        eval_video_log=False,
        render_freq=0,
        eval_mode=True,
        save_path=str(out),
    )
    return args


_NDARRAY_KEY = "__ndarray__"  # same msgpack encoding as the policy server; kept local so the client
# runs in the simulator environment, which has no cosmos_framework.


def _pack_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_KEY: True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
    raise TypeError(f"Unsupported msgpack type: {type(obj)}")


def _unpack_ndarray(obj):
    if isinstance(obj, dict) and obj.get(_NDARRAY_KEY):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def pack_message(data: dict) -> bytes:
    import msgpack

    return msgpack.packb(data, default=_pack_ndarray, use_bin_type=True)


def unpack_message(payload: bytes) -> dict:
    import msgpack

    return msgpack.unpackb(payload, raw=False, object_hook=_unpack_ndarray)


def request_actions(socket, message: dict) -> np.ndarray:
    socket.send(pack_message(message))
    reply = unpack_message(socket.recv())
    if "error" in reply:
        raise RuntimeError(f"policy server: {reply['error']}")
    actions = np.asarray(reply["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise RuntimeError(f"policy returned {actions.shape}, expected [horizon, 14]")
    return actions


def training_colours(rgb: np.ndarray) -> np.ndarray:
    """The live render in the channel order the policy was trained on.

    RoboTwin's collector hands its RGB camera arrays to OpenCV, which treats them as BGR, so every recorded JPEG, and
    the training data decoded from it with a standard decoder, has red and blue swapped relative to the live render.
    On frame 0 of a recorded seed the live head camera matches the recording at 21.9-35.4 dB PSNR as rendered and at
    43.6-50.2 dB with red and blue swapped (wrist cameras ~30 dB and ~51.5 dB), i.e. identical up to JPEG. The policy
    learned the swapped colours, so it is served them.
    """
    return np.ascontiguousarray(np.asarray(rgb)[..., 2::-1])


def run_episode(task_env, socket, instruction: str, chunk: int, sampler: dict) -> dict:
    steps, requests = 0, 0
    while not (task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim):
        observation = task_env.get_obs()
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        message = {"prompt": instruction, "state": state, **sampler}
        for key, camera in CAMERAS.items():
            message[key] = training_colours(observation["observation"][camera]["rgb"])
        actions = request_actions(socket, message)
        requests += 1
        for action in actions[:chunk]:
            task_env.take_action(np.asarray(action, dtype=np.float64), action_type="qpos")
            steps += 1
            if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                break
    return dict(success=bool(task_env.eval_success), steps=steps, requests=requests, step_lim=int(task_env.step_lim))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--config", default="demo_clean", help="demo_clean or demo_randomized")
    parser.add_argument(
        "--episodes", type=int, default=100, help="solved seeds to evaluate per task (the official test_num)"
    )
    parser.add_argument("--seed", type=int, default=0, help="official seed index: start = 100000 * (1 + seed)")
    parser.add_argument("--max-seed-attempts", type=int, default=0, help="default: 50 per requested episode")
    parser.add_argument("--server", default="tcp://127.0.0.1:8101")
    parser.add_argument("--chunk", type=int, default=16, help="how many of the predicted targets to execute")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument(
        "--skip-seeds",
        type=Path,
        default=None,
        help='lines "<config> <task> <seed>": seeds whose setup or expert check never returns in the simulator',
    )
    parser.add_argument(
        "--setup-timeout",
        type=int,
        default=900,
        help="seconds; a seed whose setup / expert check does not return in time is skipped (0: off)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="resume a task at this seed (after a seed that hung or crashed the simulator)",
    )
    parser.add_argument(
        "--resume-done",
        type=int,
        default=0,
        help="episodes the resumed task already has; --episodes stays the official test_num",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = os.environ["ROBOTWIN_ROOT"]
    os.chdir(root)
    sys.path[:0] = [root, os.path.join(root, "scripts")]
    import collect_data as collect_module
    import yaml
    import zmq
    from envs import CONFIGS_PATH
    from envs._base_task import Base_Task
    from envs.utils import UnStableError
    from eval_policy_xpolicylab import build_instruction

    base = yaml.safe_load(Path(CONFIGS_PATH, f"{args.config}.yml").read_text())
    embodiments = yaml.safe_load(Path(CONFIGS_PATH, "_embodiment_config.yml").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    records = (args.out / "records.jsonl").open("a")

    def emit(row: dict) -> None:
        line = json.dumps(row)
        print(line, flush=True)
        records.write(line + "\n")
        records.flush()

    sampler = dict(num_steps=args.num_steps, guidance=args.guidance, shift=args.shift)
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(args.server)
    attempts_budget = args.max_seed_attempts or args.episodes * 50
    skip_seeds = set()
    if args.skip_seeds is not None:
        for line in args.skip_seeds.read_text().splitlines():
            parts = line.split("#")[0].split()
            if len(parts) == 3:
                skip_seeds.add((parts[0], parts[1], int(parts[2])))
    # Base_Task.get_cluttered_table shifts its default xlim / ylim lists in place by table_xy_bias (dump_bin_bigbin,
    # put_bottles_dustbin: +0.3 m per call), so every later task in the same process would sample clutter off the
    # table. The official harness runs one task per process; the pristine defaults are restored at each task start
    # to match it (accumulation within a task stays as the official harness has it).
    clutter_defaults = copy.deepcopy(Base_Task.get_cluttered_table.__defaults__)

    class SetupTimeout(Exception):
        pass

    def on_alarm(signum, frame):
        raise SetupTimeout()

    signal.signal(signal.SIGALRM, on_alarm)
    rows = []
    for task_name in args.tasks:
        Base_Task.get_cluttered_table.__defaults__ = copy.deepcopy(clutter_defaults)
        env = collect_module.class_decorator(task_name)
        task_args = env_args(base, task_name, args.out / task_name, collect_module, embodiments)
        seed_value = args.seed_start if args.seed_start is not None else 100000 * (1 + int(args.seed))
        done, attempts = int(args.resume_done), 0
        while done < args.episodes and attempts < attempts_budget:
            attempts += 1
            if (args.config, task_name, seed_value) in skip_seeds:
                emit(dict(task=task_name, seed=seed_value, config=args.config, skipped="hang_blacklist"))
                seed_value += 1
                continue
            episode_info, skip = {"info": {}}, None
            try:
                signal.alarm(args.setup_timeout)
                env.setup_demo(now_ep_num=done, seed=seed_value, is_test=True, **task_args)
                episode_info = env.play_once()
                solved = bool(env.plan_success and env.check_success())
            except SetupTimeout:
                solved, skip = False, f"setup_timeout_{args.setup_timeout}s"
            except UnStableError:
                solved, skip = False, "unstable"
            except Exception as exc:
                solved, skip = False, f"expert_error:{type(exc).__name__}"
            finally:
                signal.alarm(0)
                try:
                    env.close_env()
                except Exception:
                    pass
            if not solved:
                emit(dict(task=task_name, seed=seed_value, config=args.config, skipped=skip or "expert_failed"))
                seed_value += 1
                continue
            try:
                signal.alarm(args.setup_timeout)
                env.setup_demo(now_ep_num=done, seed=seed_value, is_test=True, **task_args)
            except (UnStableError, SetupTimeout) as exc:
                kind = "unstable" if isinstance(exc, UnStableError) else f"setup_timeout_{args.setup_timeout}s"
                emit(dict(task=task_name, seed=seed_value, config=args.config, skipped=kind))
                if isinstance(exc, SetupTimeout):
                    try:
                        env.close_env()
                    except Exception:
                        pass
                seed_value += 1
                continue
            finally:
                signal.alarm(0)
            instruction = build_instruction(
                task_args, episode_info, base.get("eval_instruction", "seen"), args.episodes
            )
            env.set_instruction(instruction=instruction)
            try:
                row = run_episode(env, socket, instruction, args.chunk, sampler)
            finally:
                try:
                    env.close_env()
                except Exception:
                    pass
            row.update(
                task=task_name,
                seed=seed_value,
                episode=done,
                instruction=instruction,
                config=args.config,
                chunk=args.chunk,
                **sampler,
            )
            emit(row)
            rows.append(row)
            done += 1
            seed_value += 1
    records.close()
    by_task = {task: [r for r in rows if r["task"] == task] for task in args.tasks}
    summary = {
        task: (round(100 * float(np.mean([r["success"] for r in values])), 1) if values else None)
        for task, values in by_task.items()
    }
    print(
        json.dumps(
            dict(
                config=args.config,
                episodes=len(rows),
                summary=summary,
                success_rate=round(100 * float(np.mean([r["success"] for r in rows])), 1) if rows else None,
            )
        )
    )


if __name__ == "__main__":
    main()
