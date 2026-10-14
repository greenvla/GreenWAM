# LIBERO and LIBERO-plus evaluation of the released checkpoint

How to check the released LIBERO checkpoint on the reference protocol. The client is the
upstream LIBERO client `cosmos_framework/simulation/libero/closed_loop_eval.py`; `evaluate.sh` runs it with the flags
of the reported evaluation against the GreenWAM LIBERO policy server (`greenwam/infer/libero_policy_server.py`).
Which benchmark runs, LIBERO or LIBERO-plus, is decided by the LIBERO package on `PYTHONPATH`.

## Protocol

- LIBERO: 4 suites (spatial, object, goal, libero_10) x 10 tasks x 50 episodes. Episode i of a task starts from entry
  i of the task's LIBERO init-state file. Success rate: successes over the 2000 episodes.
- Step budget: the suite's `max_steps` (220 / 280 / 300 / 520) plus the 10 warm-up no-op steps the client runs first,
  as in the LIBERO evaluation of OpenVLA-OFT (`run_libero_eval.py`: `t < max_steps + num_steps_wait`).
  `--reference_step_budget` turns this on; without it the upstream client counts the warm-up inside `max_steps`.
- Observation: agentview and wrist frames at 256 x 256, sent side by side as one PNG, and with `--send_state` the
  8-D robot state (end-effector position, axis-angle, the two finger joints: `observation.state` of the training
  data), from which the server builds the policy's current pose. The upstream client sends no state by default.
- Policy: a chunk of 32 relative end-effector actions per request (position, rotation 6D, gripper opening), of which
  the client executes the first 16 (`--action_horizon 16`). 4 UniPC steps, shift 5, guidance 1, EMA weights. The
  server draws the diffusion seed of every request from `numpy.random.default_rng(--seed)` (`--deterministic-seed`
  uses `--seed` itself for every request), answers one request at a time and keeps cuDNN autotuning on, as the
  reported evaluation's server did.
- LIBERO-plus (zero-shot, no LIBERO-plus data in training): all 10 030 perturbed tasks of the four suites, 1 episode
  each. Total: successes over all tasks; per perturbation axis by LIBERO-plus's `task_classification.json`.
- Simulator seed 0 (`--seed 0`).

Reported result:

| Benchmark | Score | Detail |
|---|---|---|
| LIBERO | 97.40 (1948/2000) | spatial 493/500, object 497/500, goal 481/500, libero_10 477/500 |
| LIBERO-plus | 86.78 (8704/10030) | Camera 81.74, Robot 67.55, Language 94.27, Light 97.37, Background 90.33, Noise 95.57, Layout 84.39 |

LIBERO-plus by suite: spatial 93.51, object 91.10, goal 78.23, libero_10 84.84.

## Requirements

- LIBERO at commit `8f1084e` (github.com/Lifelong-Robot-Learning/LIBERO) or LIBERO-plus at commit `4976dc3`
  (github.com/sylvestf/LIBERO-plus) with its `assets.zip`, on `PYTHONPATH`, and a `LIBERO_CONFIG_PATH` directory whose
  `config.yaml` points at that checkout's `bddl_files`, `init_files` and `assets`.
- robosuite 1.4.0, mujoco 2.3.7, PyTorch (the client imports `cosmos_framework` helpers), `requests`.
- EGL for offscreen rendering. The reported LIBERO run rendered on NVIDIA EGL, LIBERO-plus on Mesa llvmpipe
  (`EGL_DEVICE=-1`, CPU only); on 553 paired LIBERO-plus tasks the two gave 84.4 and 85.0.

## Run

Start a policy server with the released weights (`--checkpoint-path`, the model `--config-file` and the action
statistics `--stats` ship together), then run clients against it:

```bash
export GREENWAM_SEMANTIC_EXTENSIONS=benchmarks/semantic_extensions/franka_eef_v1.json WAN_VAE_PATH=<Wan2.2_VAE.pth> \
       COSMOS3_EDGE_DIR=<Cosmos3-Edge download at revision a9d944e2>
PYTHONPATH=. python -m greenwam.infer.libero_policy_server \
    --checkpoint-path <released model> --config-file <released config.yaml> --stats <libero_stats.json> --port 8300

# in the simulator environment
# LIBERO: one client per suite, 50 episodes per task, 10 environments in parallel
for suite in libero_spatial libero_object libero_goal libero_10; do
  SERVER=http://127.0.0.1:8300 OUT="$GREENWAM_OUT/libero/$suite" bash benchmarks/libero/evaluate.sh $suite
done
python benchmarks/libero/tally.py libero "$GREENWAM_OUT/libero"

# LIBERO-plus: 1 episode per task, in batches of task ids
SERVER=http://127.0.0.1:8300 OUT="$GREENWAM_OUT/libero_plus/libero_10/0000" \
  bash benchmarks/libero/evaluate.sh libero_10 "$(seq -s, 0 99)" 1 1
python benchmarks/libero/tally.py plus "$GREENWAM_OUT/libero_plus" \
  --classification "$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json"
```

A request that fails on the server side (a timeout of a cold server, say) ends the episode, and the upstream client
counts it as a failure. `tally.py` leaves such episodes out, lists them and exits with status 1. Rerun them into
another directory and pass both: a valid record replaces a failed one.

Compare two LIBERO evaluations on the same episodes (B minus A, 95 % interval from a paired hierarchical bootstrap
over tasks, then episodes):

```bash
python benchmarks/libero/tally.py paired --a <outputs of A> --b <outputs of B>
```
