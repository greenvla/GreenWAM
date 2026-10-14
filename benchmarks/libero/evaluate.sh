#!/usr/bin/env bash
# One client of the LIBERO / LIBERO-plus evaluation of the released checkpoint against a running GreenWAM LIBERO policy
# server: the upstream client cosmos_framework/simulation/libero/closed_loop_eval.py with the flags of the reported
# evaluation. The LIBERO package on PYTHONPATH (LIBERO or LIBERO-plus) and LIBERO_CONFIG_PATH decide the benchmark.
#   SERVER=http://<host>:<port> OUT=<dir> bash benchmarks/libero/evaluate.sh <suite> [task ids] [trials] [envs]
# task ids: comma-separated, all tasks of the suite when empty; trials per task (default 50); parallel environments of
# this client (default 10). EGL_DEVICE: EGL device to render on (default -1, the default device, e.g. Mesa llvmpipe).
set -euo pipefail
SUITE=${1:?suite}; TASKS=${2:-}; TRIALS=${3:-50}; ENVS=${4:-10}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
exec python "$ROOT/cosmos_framework/simulation/libero/closed_loop_eval.py" --server_url "${SERVER:?SERVER}" \
  --task_suite "$SUITE" ${TASKS:+--task_ids "$TASKS"} --num_trials_per_task "$TRIALS" --num_envs "$ENVS" \
  --camera agentview,wrist --image_size 256 --action_space frame_wise_relative --rotation_space 6d --action_dim 10 \
  --gripper_mode zero_one --action_horizon 16 --send_state --reference_step_budget --seed 0 \
  --wait_timeout 900 --timeout 300 --render_gpu_device_id "${EGL_DEVICE:--1}" --output_dir "${OUT:?OUT}"
