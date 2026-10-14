#!/usr/bin/env bash
# Generates the WorldArena test videos of one checkpoint, one process per visible GPU (episodes sharded by rank).
# usage: benchmarks/worldarena/rollout.sh <checkpoint_dir> <config.yaml> <episodes.json> <out_dir> [extra rollout args]
#   <config.yaml>    the training config published with the checkpoint
#   <episodes.json>  the WorldArena test summary (list of {gt_path, ...}) or a list of episode ids
# The reported numbers use the defaults below: no text, segments of at most 15 s (the cap of the training release),
# 35 sampling steps, guidance 1.0, seed 0. Videos are written to <out_dir>/flat/*.mp4.
# Environment: WORLDARENA_TEST_ROOT (WorldArena 1.0 test_dataset), GREENWAM_STATS (as for training). One process per GPU
# of CUDA_VISIBLE_DEVICES when it is set, else per GPU nvidia-smi lists. Paths may be relative to the calling directory.
set -uo pipefail
absp() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
CKPT=$(absp "$1") CFG=$(absp "$2") EPS=$(absp "$3") OUT=$(absp "$4"); shift 4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
for v in WORLDARENA_TEST_ROOT GREENWAM_STATS; do [ -n "${!v:-}" ] || { echo "set $v" >&2; exit 2; }; done
WORLDARENA_TEST_ROOT=$(absp "$WORLDARENA_TEST_ROOT") GREENWAM_STATS=$(absp "$GREENWAM_STATS")
export PYTHONPATH=$REPO:$REPO/greenwam/data:$REPO/greenwam/proj${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1
mkdir -p "$OUT/logs"; cd "$REPO"
GPUS=()
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then IFS=, read -r -a GPUS <<<"$CUDA_VISIBLE_DEVICES"
else for ((i = 0; i < $(nvidia-smi -L | wc -l); i++)); do GPUS+=("$i"); done; fi
NG=${#GPUS[@]}; pids=()
[ "$NG" -ge 1 ] || { echo "no visible GPU" >&2; exit 2; }
for r in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python -m greenwam.infer.worldarena_world_model --checkpoint-path "$CKPT" --config-file "$CFG" \
    --stats "$GREENWAM_STATS/delta_from_q0/statistics.json" --test-root "$WORLDARENA_TEST_ROOT" --episodes "$EPS" \
    --out "$OUT" --rank "$r" --world-size "$NG" --no-text --cap-seconds 15 "$@" > "$OUT/logs/rank$r.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "rollout rc=$rc videos=$(ls "$OUT"/flat/*.mp4 2>/dev/null | wc -l)"; exit $rc
