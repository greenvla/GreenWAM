# SPDX-License-Identifier: OpenMDW-1.1
"""WorldArena 1.0 EWMScore of one evaluated run, capped and uncapped.

usage: ewmscore.py <run_dir> <gt_run_dir>
Both directories hold csv_results/aggregated_results.csv as written by the evaluator (one row per video; the GT run is
the test-set ground truth evaluated as a model with the same evaluator).

- uncapped: the board formula, 100 x the mean over the 15 metrics of each metric's mean over videos;
- capped: the same after Dynamic Degree, Flow Score and Motion Smoothness of every video are capped at the value of
  its ground-truth video (the motion rule of WorldArena 2.0), so motion beyond the real one is not rewarded.
Means skip videos where a metric is missing; nothing is rounded before the final print.
"""

import csv
import statistics
import sys

METRICS = [
    "Image Quality",
    "Aesthetic Quality",
    "JEPA Similarity",
    "Dynamic Degree",
    "Flow Score",
    "Motion Smoothness",
    "Subject Consistency",
    "Background Consistency",
    "Photometric Consistency",
    "Interaction Quality",
    "Trajectory Accuracy",
    "Depth Accuracy",
    "Perspectivity",
    "Instruction Following",
    "Semantic Alignment",
]
CAPPED = {"Dynamic Degree", "Flow Score", "Motion Smoothness"}


def load(run_dir):
    with open(f"{run_dir}/csv_results/aggregated_results.csv", newline="") as f:
        return {
            row["Video_ID"]: {m: float(row[m]) if row.get(m) not in (None, "") else None for m in METRICS}
            for row in csv.DictReader(f)
        }


def score(videos, cap_with=None):
    means = {}
    for m in METRICS:
        values = []
        for vid, row in videos.items():
            value = row[m]
            if value is None:
                continue
            if cap_with is not None and m in CAPPED and cap_with[vid][m] is not None:
                value = min(value, cap_with[vid][m])
            values.append(value)
        means[m] = statistics.fmean(values)
    return 100 * statistics.fmean(means.values()), means


def main():
    run, gt = load(sys.argv[1]), load(sys.argv[2])
    missing = sorted(set(run) - set(gt))
    if missing:
        raise SystemExit(f"{len(missing)} videos have no ground-truth row, e.g. {missing[:3]}")
    uncapped, means_u = score(run)
    capped, means_c = score(run, cap_with=gt)
    print(f"videos {len(run)}")
    print(f"{'metric':26s}{'uncapped':>10s}{'capped':>10s}")
    for m in METRICS:
        print(f"{m:26s}{100 * means_u[m]:10.4f}{100 * means_c[m]:10.4f}")
    print(f"EWMScore uncapped {uncapped:.4f}  capped {capped:.4f}")


if __name__ == "__main__":
    main()
