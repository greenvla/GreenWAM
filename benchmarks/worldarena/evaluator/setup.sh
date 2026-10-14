#!/usr/bin/env bash
# Builds the evaluator of the reported numbers: WorldArena 2.0 at 5978ce5 (MIT) with patches/ applied in order.
# usage: benchmarks/worldarena/evaluator/setup.sh <dest_dir>      then: export WA_EVAL_DIR=<dest_dir>
# With WA_MODELS set, the V-JEPA weights of the JEPA metric are linked from $WA_MODELS/v-jepa into the checkout.
# Judge weights and the evaluator's three Python environments are described in benchmarks/worldarena/README.md.
set -euo pipefail
DEST=$1
HERE=$(cd "$(dirname "$0")" && pwd)
git clone -q https://github.com/WorldArena2/WorldArena-2.0 "$DEST"
git -C "$DEST" checkout -q 5978ce5c81e55b8c8358f4f5966a13ce385ff155
for p in "$HERE"/patches/*.patch; do
  git -C "$DEST" apply "$p"
  echo "applied $(basename "$p")"
done
if [ -n "${WA_MODELS:-}" ]; then
  mkdir -p "$DEST/video_quality_ood/JEDi/pretrained_models"
  for f in vith16.pth.tar ssv2-probe.pth.tar; do
    ln -sfn "$WA_MODELS/v-jepa/$f" "$DEST/video_quality_ood/JEDi/pretrained_models/$f"
  done
fi
