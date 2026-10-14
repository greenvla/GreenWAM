# SPDX-License-Identifier: OpenMDW-1.1
"""Paired comparison of two evaluated WorldArena runs on their common videos, both capped per video at the ground truth.

usage: compare.py <candidate_run> <base_run> <gt_run>   (each holds csv_results/aggregated_results.csv)
Prints the 15 metric means, the capped EWMScore of both runs, and the paired differences with a 95 % bootstrap interval
(2000 resamples over videos).
"""

import csv
import random
import statistics as st
import sys

M = [
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
CAP = {"Dynamic Degree", "Flow Score", "Motion Smoothness"}


def load(run):
    rows = csv.DictReader(open(f"{run}/csv_results/aggregated_results.csv"))
    return {r["Video_ID"]: {k: (float(r[k]) if r.get(k) not in (None, "") else None) for k in M} for r in rows}


cand, base, gt = (load(a) for a in sys.argv[1:4])
ids = sorted(i for i in cand if i in base and i in gt)


def capped(D):
    return {
        i: {
            k: (min(D[i][k], gt[i][k]) if k in CAP and D[i][k] is not None and gt[i][k] is not None else D[i][k])
            for k in M
        }
        for i in ids
    }


C, B = capped(cand), capped(base)


def paired(i, k):
    """A video counts for metric k only where both runs (and, for a capped metric, the ground truth) have a value."""
    return C[i][k] is not None and B[i][k] is not None and (k not in CAP or gt[i][k] is not None)


def means(D, sel):
    out = {}
    for k in M:
        vals = [D[i][k] for i in sel if paired(i, k)]
        out[k] = st.mean(vals) if vals else float("nan")
    return out


def ewm(D, sel):
    return 100 * st.mean(means(D, sel).values())


mc, mb = means(C, ids), means(B, ids)
print(f"ids {len(ids)} (candidate {len(cand)}, base {len(base)}, gt {len(gt)})")
for k in M:
    n = sum(paired(i, k) for i in ids)
    if n < len(ids):
        print(f"{k}: {n} of {len(ids)} videos have a value in both runs; both means use only these")
print(f"{'metric':26s}{'cand':>8s}{'base':>8s}{'delta':>8s}")
for k in M:
    print(f"{k:26s}{100 * mc[k]:8.2f}{100 * mb[k]:8.2f}{100 * (mc[k] - mb[k]):+8.2f}")
rng = random.Random(0)


def ci(f):
    d = [f(rng.choices(ids, k=len(ids))) for _ in range(2000)]
    d.sort()
    return d[50], d[1949]


for name, f in (
    ("EWM capped", lambda s: ewm(C, s) - ewm(B, s)),
    ("Image Quality", lambda s: 100 * (means(C, s)["Image Quality"] - means(B, s)["Image Quality"])),
    ("Trajectory", lambda s: 100 * (means(C, s)["Trajectory Accuracy"] - means(B, s)["Trajectory Accuracy"])),
):
    lo, hi = ci(f)
    print(f"GATE {name}: cand-base {f(ids):+.2f} [{lo:+.2f}, {hi:+.2f}]")
print(
    f"GATE EWM capped: cand {ewm(C, ids):.2f} base {ewm(B, ids):.2f}; Trajectory cand {100 * mc['Trajectory Accuracy']:.2f}"
)
