# SPDX-License-Identifier: OpenMDW-1.1
"""Places an evaluated run on the WorldArena 1.0 board, raw and under the motion rule of WorldArena 2.0.

usage: board.py <board_dir> <run_dir> <gt_run_dir> [--name NAME] [--csv out.csv]
<board_dir> holds the per-model result JSONs of the board, e.g.
  huggingface-cli download WorldArena/WorldArena --repo-type space \\
    --revision 483dfcc9a2d57974809c15d2a235c3e204f2b0f2 --include "worldarena-results/*" --local-dir board
  (then pass board/worldarena-results).
The board keeps only per-model means, so a board entry is capped on its means: min(model mean, ground-truth mean) for
Dynamic Degree, Flow Score and Motion Smoothness, which is an upper bound of the per-video cap. The evaluated run is
capped per video against its ground-truth video (exact), as in ewmscore.py.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from ewmscore import CAPPED, METRICS, load, score

CATEGORIES = {
    "Visual": ["Image Quality", "Aesthetic Quality", "JEPA Similarity"],
    "Motion": ["Dynamic Degree", "Flow Score", "Motion Smoothness"],
    "Consistency": ["Subject Consistency", "Background Consistency", "Photometric Consistency"],
    "Physics": ["Interaction Quality", "Trajectory Accuracy"],
    "3D": ["Depth Accuracy", "Perspectivity"],
    "Control": ["Instruction Following", "Semantic Alignment"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board_dir", type=Path)
    ap.add_argument("run_dir")
    ap.add_argument("gt_run_dir")
    ap.add_argument("--name", default="GreenWAM")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    run, gt = load(args.run_dir), load(args.gt_run_dir)
    _, gt_means = score(gt)
    _, raw_means = score(run)
    _, capped_means = score(run, cap_with=gt)
    rows = []  # (name, source, raw means, capped means), means as fractions
    for path in sorted(args.board_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        metrics = {k: v for k, v in (doc.get("Metrics") or {}).items() if k in METRICS and isinstance(v, (int, float))}
        if metrics:
            capped = {k: min(v, gt_means[k]) if k in CAPPED else v for k, v in metrics.items()}
            rows.append((doc.get("Model", path.stem), "board", metrics, capped))
    rows.append((args.name, "this run", raw_means, capped_means))
    table = []
    for name, source, raw, capped in rows:
        entry = dict(
            model=name,
            source=source,
            n_metrics=len(raw),
            ewm_raw=100 * statistics.fmean(raw.values()),
            ewm_capped=100 * statistics.fmean(capped.values()),
        )
        for cat, keys in CATEGORIES.items():
            entry[cat] = 100 * statistics.fmean(capped[k] for k in keys) if all(k in capped for k in keys) else None
        table.append(entry)
    for key in ("ewm_raw", "ewm_capped"):
        for i, entry in enumerate(sorted(table, key=lambda e: -e[key])):
            entry["rank_" + key[4:]] = i + 1
    table.sort(key=lambda e: e["rank_capped"])
    print(
        f"{'#cap':>4s}{'#raw':>5s}  {'model':36s}{'raw':>8s}{'capped':>8s}  " + "".join(f"{c:>12s}" for c in CATEGORIES)
    )
    for e in table:
        cats = "".join(f"{e[c]:12.2f}" if e[c] is not None else f"{'-':>12s}" for c in CATEGORIES)
        print(
            f"{e['rank_capped']:4d}{e['rank_raw']:5d}  {e['model'][:36]:36s}{e['ewm_raw']:8.2f}{e['ewm_capped']:8.2f}  {cats}"
        )
    print(f"board entries {len(table) - 1}, with all 15 metrics {sum(1 for e in table if e['n_metrics'] == 15) - 1}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main()
