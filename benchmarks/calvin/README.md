# CALVIN ABC→D evaluation

A thin client of the GreenWAM policy server on top of the official CALVIN evaluation code: the canonical sequences
(`get_sequences(1000)`), the official `evaluate_sequence` / `rollout` loop, the CALVIN task oracle. The client only
turns the server's action chunks into simulator actions.

## Protocol

- 1000 canonical evaluation sequences of 5 chained subtasks in scene D; at most 360 environment steps per subtask; the
  chain stops at the first failed subtask. Score: average chain length (ACL) over the sequences, and the fraction of
  sequences that complete 1 … 5 subtasks.
- Seed: `seed_everything(0, workers=True)` before anything else. Initial scenes are seeded per sequence by the FNV-1
  hash of the initial condition (`pyhash_compat.py` supplies the hash).
- One request per chunk: the server receives the instruction, the static and gripper RGB frames as rendered
  (200×200 and 84×84) and the 15-D robot observation, and returns `prediction_horizon` actions; the client executes the
  first `execution_horizon` (absolute TCP targets, gripper thresholded at 0) and then asks again.
- Reported setting: prediction horizon 20, execution horizon 10, 4 UniPC steps with shift 3, text guidance 1.5
  (`guidance`, `video_text_cfg`, `action_text_cfg`), EMA weights on the server.
- Rendering: Mesa llvmpipe through EGL (`--egl-device <index of the llvmpipe EGL device>`).

Reported result: average length 4.64 over the 1000 sequences.

## Requirements

- `calvin` at commit `d677d38` (github.com/mees/calvin) with `calvin_models` on `PYTHONPATH`.
- `calvin_env` at commit `797142c` (github.com/mees/calvin_env) with its `tacto` submodule; later commits change the
  robot controller.
- `msgpack`, `pyzmq`, `pytorch_lightning`, `hydra-core`, `omegaconf` and the `calvin_env` requirements.
- A CALVIN split directory with `.hydra/merged_config.yaml` for scene D (for example `task_D_D/validation` or
  `task_ABC_D/validation` of the CALVIN download); only the environment configuration is read from it.

## Run

Start one policy server per shard, then evaluate a slice of the canonical list against it and merge:

```bash
# policy server, in the GreenWAM environment
export GREENWAM_SEMANTIC_EXTENSIONS=benchmarks/semantic_extensions/franka_eef_v1.json WAN_VAE_PATH=<Wan2.2_VAE.pth> \
       COSMOS3_EDGE_DIR=<Cosmos3-Edge download at revision a9d944e2>
PYTHONPATH=. python -m greenwam.infer.calvin_policy_server --checkpoint-path <released model> \
    --config-file <released config.yaml> --stats <released calvin_stats.json> --port 8200 --seed 0

# in the CALVIN environment
python -m benchmarks.calvin.evaluate --dataset-path "$CALVIN_ROOT/task_D_D/validation" \
    --server-port 8200 --seq-start 0 --seq-end 42 --egl-device "$EGL_LLVMPIPE" --output-dir "$GREENWAM_OUT/calvin/shard00"
python -m benchmarks.calvin.aggregate "$GREENWAM_OUT"/calvin/shard* --output "$GREENWAM_OUT/calvin/result.json"
```

24 shards of 42 consecutive sequences cover the 1000. Each shard needs a freshly started server: the server draws
the seed of every request from a NumPy generator seeded once at start-up (`--seed`, default 0), so a shard repeats
bit for bit only when its requests are the first ones the server sees. The released config carries the numeric
switches the run needs (`RECIPE_SWITCHES` in `greenwam/infer/calvin_policy_server.py`); the server checks them at
start-up and refuses to serve without them.
