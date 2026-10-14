# RoboTwin 2.0 evaluation of the released checkpoint

How to check the released RoboTwin 2.0 checkpoint on the board protocol. A client of the
GreenWAM RoboTwin policy server (`greenwam/infer/robotwin_policy_server.py`) on top of the official RoboTwin 2.0
evaluation path: the task configs, seeds, expert check, instruction generation, step limits and success checks are
RoboTwin's own; the client only turns the server's action chunks into `take_action` calls.

## Protocol

- Training sees only `demo_clean` demonstrations (aloha-agilex, 50 tasks x 50 episodes). Evaluation runs every task
  in `demo_clean` (seen instructions) and `demo_randomized` (unseen instructions), both upstream task configs, with
  `eval_mode=True`: the per-task step limit and, for randomized, the unseen texture pool.
- 100 episodes per task and setting on the official seed path: seeds from 100000 upward, the first 100 recorded
  episodes per task in seed order. A seed is skipped when the expert check rejects its scene (`expert_failed`,
  `unstable`, `expert_error:*`) or when its setup never returns in the simulator (`skip_seeds.txt` lists the seeds
  that hung in the reported evaluation).
- The success rate of a setting is the mean over its 50 tasks of per-task success; the headline is the mean of clean
  and randomized.
- Policy settings: 17 predicted steps, the first 16 executed (`--chunk 16`), 8 sampler steps, guidance 1.0, shift
  5.0, EMA weights. Each server draws the diffusion seed of every request from `numpy.random.default_rng(--seed)`.
- Episodes do not repeat bit for bit between evaluations. RoboTwin's instruction generator
  (`description/utils/generate_episode_instructions.py`) picks each episode's instruction with Python's global
  `random`, which the evaluation path does not seed (`setup_demo` seeds only NumPy). The expert check is not
  repeatable either: in the 100-episode evaluations of two checkpoints of the run (the expert does not involve the policy),
  1095 of the 16495 seeds both reached (6.6 %) got a different verdict (an evaluated episode in one, `expert_failed`
  or `expert_error` in the other), so the counted seeds differ slightly between evaluations. `paired_bootstrap.py`
  therefore pairs only the seeds both evaluations recorded.
- Camera frames are served with red and blue swapped (`training_colours`): the recorded RoboTwin demonstrations, and
  the training data built from them, carry that channel order.

Reported result: clean 86.98, randomized 48.10, mean 67.54 (5000 + 5000 episodes).

## Requirements

- RoboTwin 2.0 at commit `6dde571` (github.com/RoboTwin-Platform/RoboTwin) with the `XPolicyLab` submodule checked out
  at `bb9a0b5`, its assets (`_download_assets.sh`) and its simulator environment (SAPIEN 3.0.0b1); task configs and
  `scripts/eval_policy_xpolicylab.py` unmodified.
- `msgpack` and `pyzmq` in that environment. The client imports nothing from this repository.
- A GPU for the policy server (12.9 GB per server on an H100; the simulator renders on the same or another GPU).

## Run

Start a policy server with the released weights (`--checkpoint-path`, the model `--config-file` and the action
statistics `--stats` ship together), then evaluate tasks against it:

```bash
export GREENWAM_LAYOUT_EXTENSIONS=benchmarks/robotwin/layout_extensions.json \
       WAN_VAE_PATH=<Wan2.2_VAE.pth> \
       COSMOS3_EDGE_DIR=<Cosmos3-Edge download at revision a9d944e2>
PYTHONPATH=. python -m greenwam.infer.robotwin_policy_server \
    --checkpoint-path <released model> --config-file <released config.yaml> \
    --stats <released delta_from_q0 statistics.json> --port 8101 --seed 0

# in the simulator environment
export ROBOTWIN_ROOT=<RoboTwin checkout>
python benchmarks/robotwin/evaluate.py --tasks adjust_bottle beat_block_hammer --config demo_clean --episodes 100 \
    --chunk 16 --skip-seeds benchmarks/robotwin/skip_seeds.txt --server tcp://127.0.0.1:8101 --out "$GREENWAM_OUT/robotwin/main"
```

The reported evaluation ran one server per worker and one task at a time per worker. A seed that hangs the simulator
anyway (the client then prints nothing for minutes; a rollout prints every episode and an expert check takes minutes
at most) is skipped by stopping the client and resuming the task one seed later with the episodes it already has:
`--seed-start <hung seed + 1> --resume-done <episodes so far>` into a second output directory. A simulator crash
(SAPIEN `cannot create buffer` after many episodes in one process) is resumed at the seed it died on. `--episodes`
stays 100 in both cases: it is the official `test_num` the instruction generator receives.

Tally, with the outputs in priority order (the first output that has a record of a seed decides it):

```bash
python benchmarks/robotwin/tally.py 100 "$GREENWAM_OUT/robotwin/main" "$GREENWAM_OUT/robotwin/resumed" --tsv tally.tsv
python benchmarks/robotwin/paired_bootstrap.py 100 --a <outputs of policy A> --b <outputs of policy B>
```

`paired_bootstrap.py` pairs the two evaluations by (setting, task, seed) and gives the B − A difference with a 95 %
interval from a hierarchical bootstrap (tasks, then paired episodes).
