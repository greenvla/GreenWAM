#!/usr/bin/env bash
# WorldArena 1.0 evaluation (the 15 EWMScore metrics) of one rollout directory with the patched WorldArena 2.0 evaluator.
# usage: benchmarks/worldarena/evaluate.sh <model_name> <video_dir> <summary.json> <run_dir> [nproc=8]
#   <video_dir>     flat directory of generated videos (<task>_episode_<n>.mp4, as rollout.sh writes them)
#   <summary.json>  the WorldArena test summary used for the rollout (one entry per episode)
#   <run_dir>       evaluation output; csv_results/aggregated_results.csv is the input of scoring/ewmscore.py
# Environment (see README.md): WA_EVAL_DIR (WorldArena-2.0 checkout with evaluator/patches applied), WA_MODELS (judge
# weights), WA_PYTHON, WA_VLM_PYTHON, WA_JEPA_PYTHON (the evaluator's three environments), optional
# WORLDARENA_CACHE_DIR, GOFF (offset into the visible GPUs: CUDA_VISIBLE_DEVICES if set, else all) and PORT_BASE.
# Paths may be relative to the calling directory. Finished steps are skipped on a rerun; depth and dynamic
# degree can fail at 8 ranks on network file systems, so the reported runs ran a second pass with nproc 1.
set -euo pipefail
export PYTHONUNBUFFERED=1 LC_ALL=C
HERE=$(cd "$(dirname "$0")" && pwd)
absp() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
MODEL_NAME=$1 FLAT0=$(absp "$2") SUMMARY=$(absp "$3") RUN=$(absp "$4") NPROC=${5:-8}
GOFF=${GOFF:-0}
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$(( ${PORT_BASE:-29511} + 50 ))   # evaluate.py initializes torch.distributed from env
for v in WA_EVAL_DIR WA_MODELS WA_PYTHON WA_VLM_PYTHON WA_JEPA_PYTHON; do [ -n "${!v:-}" ] || { echo "set $v" >&2; exit 2; }; done
WA_EVAL_DIR=$(absp "$WA_EVAL_DIR") WA_MODELS=$(absp "$WA_MODELS")
for v in WA_PYTHON WA_VLM_PYTHON WA_JEPA_PYTHON; do case "${!v}" in /* | "") ;; */*) printf -v "$v" '%s' "$(absp "${!v}")" ;; esac; done
[ -z "${WORLDARENA_CACHE_DIR:-}" ] || WORLDARENA_CACHE_DIR=$(absp "$WORLDARENA_CACHE_DIR")
export VQ=$WA_EVAL_DIR/video_quality_ood
export PYTHONPATH=$VQ${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
export WORLDARENA_CACHE_DIR=${WORLDARENA_CACHE_DIR:-$RUN/cache}
export PATH="$(dirname "$WA_PYTHON"):${PATH:-}"
NEXP=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$SUMMARY")
DATA_DIR=$RUN/data
OUT=$RUN/output
OUT_VLM=$RUN/output_VLM
OUT_JEPA=$RUN/output_JEDi
CONFIG_PATH=$RUN/config.yaml
JEPA_GT=$RUN/jepa_gt
FAILS=$RUN/failures.txt
STATUS=$RUN/status.log
NPROC=${NPROC:-8}

mkdir -p "$DATA_DIR" "$OUT" "$OUT_VLM" "$OUT_JEPA" "$JEPA_GT" "$RUN/tmp_VLM" "$RUN/logs"
touch "$STATUS"
: > "$FAILS"
rm -f "$OUT"/generated_results.json.tmp "$OUT"/generated_results.json.tmp.*
if [ -e "$OUT/generated_results.json" ] && [ ! -s "$OUT/generated_results.json" ]; then
  rm -f "$OUT/generated_results.json"
fi

python3 "$HERE/evaluator/write_config.py" "$MODEL_NAME" "$RUN" "$VQ" "$WA_MODELS" "$CONFIG_PATH"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$STATUS"; }
mark_done() { touch "$RUN/done_$1"; }
is_done() { [ -f "$RUN/done_$1" ]; }
soft() {
  local name="$1"; shift
  if is_done "$name"; then log "skip $name"; return 0; fi
  log "start $name"
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then mark_done "$name"; log "ok $name"; return 0; fi
  echo "$name rc=$rc" >>"$FAILS"; log "FAIL $name rc=$rc"; return 0
}
hard() {
  local name="$1"; shift
  if is_done "$name"; then log "skip $name"; return 0; fi
  log "start $name"
  "$@"
  mark_done "$name"; log "ok $name"
}
wait_ranks() {
  local name="$1"; shift
  local pids=("$@") rc=0 i=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then echo "$name rank$i pid=$pid" >>"$FAILS"; rc=1; fi
    i=$((i+1))
  done
  return "$rc"
}

# Visible GPUs: the CUDA_VISIBLE_DEVICES list when it is set, else every GPU nvidia-smi lists; GOFF skips the first ones.
GPUS=()
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then IFS=, read -r -a GPUS <<<"$CUDA_VISIBLE_DEVICES"
else for ((i = 0; i < $(nvidia-smi -L 2>/dev/null | wc -l); i++)); do GPUS+=("$i"); done; fi
GPUS=("${GPUS[@]:$GOFF}")
gpu_list() { local IFS=,; echo "${GPUS[*]:0:$1}"; }

cd "$VQ"
ngpu=${#GPUS[@]}
log "MODEL_NAME=$MODEL_NAME NPROC=$NPROC ngpu=$ngpu"
test "${ngpu:-0}" -ge 1
if [ "$ngpu" -lt "$NPROC" ]; then NPROC=$ngpu; log "WARN clamp NPROC=$NPROC"; fi

n0=$(find "$FLAT0" -maxdepth 1 -name '*.mp4' | wc -l)
ns=$(python3 -c "import json; print(len(json.load(open('$SUMMARY'))))")
log "flatten v0=$n0 summary=$ns"
NEXP=${NEXP:-1000}
test "$n0" -eq "$NEXP"
test "$ns" -eq "$NEXP"
test -x "$WA_PYTHON"
test -f "$CONFIG_PATH"

n_std=$(find "$DATA_DIR/generated_dataset" -path '*/1/video/frame_00000.jpg' 2>/dev/null | wc -l || true)
if [ "${n_std:-0}" -ge "${NEXP:-1000}" ]; then
  log "preprocess standard already $n_std"
  mark_done preprocess_standard
  mark_done resize
else
  hard preprocess_standard \
    "$WA_PYTHON" preprocess_datasets.py \
      --summary_json "$SUMMARY" \
      --gen_video_dir "$FLAT0" \
      --output_base "$DATA_DIR"
  hard resize \
    "$WA_PYTHON" ./processing/video_resize.py --config_path "$CONFIG_PATH"
fi

run_detection_8() {
  local pids=() i
  for i in $(seq 0 $((NPROC - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$WA_PYTHON" ./processing/detection_tracking.py \
      --config_path "$CONFIG_PATH" --rank "$i" --world-size "$NPROC" \
      >"$RUN/logs/detect_rank${i}.log" 2>&1 &
    pids+=($!)
  done
  wait_ranks detection "${pids[@]}"
}

n_gt_traj=$(python3 - <<PY
from pathlib import Path
gt=Path("$DATA_DIR/gt_dataset/fixed_scene_task")
print(sum(1 for ep in gt.iterdir() if ep.is_dir() and (ep/"traj"/"traj.npy").exists()) if gt.exists() else 0)
PY
)
n_gen_traj=$(python3 - <<PY
from pathlib import Path
gen=Path("$DATA_DIR/generated_dataset/fixed_scene_task")
n=0
if gen.exists():
    for ep in gen.iterdir():
        if not ep.is_dir():
            continue
        for gid in ep.iterdir():
            if (gid/"traj"/"traj.npy").exists():
                n+=1
print(n)
PY
)
log "traj gt=$n_gt_traj gen=$n_gen_traj"
if [ "${n_gt_traj:-0}" -ge "${NEXP:-1000}" ] && [ "${n_gen_traj:-0}" -ge "${NEXP:-1000}" ]; then
  mark_done detection
else
  hard detection run_detection_8
fi

ddp_eval() {
  local dim="$1" port="$2"
  CUDA_VISIBLE_DEVICES=$(gpu_list "$NPROC") \
    "$WA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --master_port="$port" \
      evaluate.py --dimension "$dim" --config "$CONFIG_PATH" \
      2>&1 | tee -a "$RUN/logs/eval_${dim}.log"
}

port=${PORT_BASE:-29511}
for dim in image_quality aesthetic_quality background_consistency dynamic_degree flow_score photometric_smoothness motion_smoothness subject_consistency depth_accuracy; do
  soft "eval_$dim" ddp_eval "$dim" "$port"
  port=$((port+1))
done

soft eval_single \
  env CUDA_VISIBLE_DEVICES=${GPUS[0]} \
  "$WA_PYTHON" evaluate.py --dimension trajectory_accuracy psnr ssim semantic_alignment --config "$CONFIG_PATH"

run_vlm_8() {
  local pids=() i
  rm -f "$OUT_VLM/$MODEL_NAME"/*_rank*.json   # parts of an earlier run must not be merged into this one
  for i in $(seq 0 $((NPROC - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$WA_VLM_PYTHON" VLM_judge.py \
      --model_name "$MODEL_NAME" --video_dir "$FLAT0" --summary_json "$SUMMARY" \
      --metrics all --num_frames 16 --output_root "$OUT_VLM" --tmp_root "$RUN/tmp_VLM" \
      --config_path "$CONFIG_PATH" --rank "$i" --world-size "$NPROC" \
      >"$RUN/logs/vlm_rank${i}.log" 2>&1 &
    pids+=($!)
  done
  wait_ranks vlm "${pids[@]}" || return 1   # a failed rank leaves the step unfinished
  "$WA_PYTHON" - "$OUT_VLM" "$MODEL_NAME" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / sys.argv[2]
parts = sorted(root.glob("*_rank*.json"))
if not parts:  # NPROC=1 writes the summary itself; merging zero rank parts would overwrite it with []
    print("no rank parts, keep", root); sys.exit(0)
merged = []
for p in parts:
    merged.extend(json.loads(p.read_text()))
out = root / f"{sys.argv[2]}_summary_val_all_intern.json"
out.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
print("merged", len(parts), "->", out, "n=", len(merged))
PY
}
soft eval_vlm run_vlm_8

if [ ! -f "$RUN/done_jepa_gt" ]; then
  "$WA_PYTHON" - "$SUMMARY" "$JEPA_GT" <<'PY'
import json, os, sys
from pathlib import Path
summary, dst = sys.argv[1], Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
n=0
for item in json.load(open(summary)):
    src=Path(item["gt_path"]); name=item.get("name") or f"fixed_scene_task_episode_{item['episode']}"
    if not name.endswith(".mp4"): name += ".mp4"
    link=dst/name
    if link.exists() or link.is_symlink():
        n+=1; continue
    if src.exists():
        os.symlink(src, link); n+=1
print("jepa_gt", n)
PY
  mark_done jepa_gt
fi

soft eval_jepa \
  env CUDA_VISIBLE_DEVICES=${GPUS[0]} \
  bash -lc "cd '$VQ/JEDi' && '$WA_JEPA_PYTHON' batch.py \
    --real_dir '$JEPA_GT' --gen_dir '$FLAT0' --model_dir pretrained_models \
    --config_path configs/vith16_ssv2_16x2x3.yaml \
    --output_root '$OUT_JEPA' --save_intersection '$OUT_JEPA/intersection_names.json'"

rm -f "$RUN/done_aggregate"   # aggregation is cheap and must see the steps a second pass recomputed
soft aggregate \
  "$WA_PYTHON" csv_results/aggregate_results.py \
    --base_dir "$RUN" --model_name "$MODEL_NAME" \
    --csv_name aggregated_results.csv --jepa_result_path "$OUT_JEPA/results.json"

log "done eval"
if [ -s "$FAILS" ]; then cat "$FAILS" | tee -a "$STATUS"; exit 1; fi
log "all steps ok"
echo ">>> done $RUN"
