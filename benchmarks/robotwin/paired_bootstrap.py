#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Paired comparison of two policies on a full RoboTwin 2.0 evaluation: arm B minus arm A.

Episodes are selected exactly as ``tally.py`` counts them (same parsing, sources in priority order, the first
``episodes`` recorded episodes per (setting, task) in seed order) and paired by (setting, task, seed): the official
seeds and the expert's rejections are the same for both arms, so a pair differs only in the policy. Interval: paired
hierarchical bootstrap of the task-mean success difference (tasks resampled with replacement, then the paired
episodes within each resampled task) for clean, randomized and their mean. The interval reflects the evaluation of
these two policies, not the spread between independent trainings.

    python benchmarks/robotwin/paired_bootstrap.py 100 --a out/policy_a --b out/policy_b
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tally import SETTINGS, TASKS, load, select  # noqa: E402


def ci(xs: list[float]) -> tuple[float, float]:
    xs = sorted(xs)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", type=int)
    ap.add_argument("--a", nargs="+", type=Path, required=True, help="sources of arm A in priority order")
    ap.add_argument("--b", nargs="+", type=Path, required=True, help="sources of arm B in priority order")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    A, B = select(load(a.a), a.episodes), select(load(a.b), a.episodes)
    pairs, report = {}, {}
    for s in SETTINGS:
        pairs[s] = {}
        for t in TASKS:
            ra, rb = A[(s, t)]["episodes"], B[(s, t)]["episodes"]
            pairs[s][t] = [(ra[x], rb[x]) for x in sorted(set(ra) & set(rb))]
        have = [v for v in pairs[s].values() if v]
        m_a = sum(sum(x for x, _ in v) / len(v) for v in have) / max(1, len(have))
        m_b = sum(sum(y for _, y in v) / len(v) for v in have) / max(1, len(have))
        short = [t for t in TASKS if len(pairs[s][t]) < a.episodes]
        report[s] = dict(A=100 * m_a, B=100 * m_b, diff=100 * (m_b - m_a), pairs=sum(len(v) for v in have), short=short)
    rng, boot = random.Random(a.seed), collections.defaultdict(list)
    have = {s: [t for t in TASKS if pairs[s][t]] for s in SETTINGS}
    for _ in range(a.reps):
        d = {}
        for s in SETTINGS:
            acc = 0.0
            for t in (rng.choice(have[s]) for _ in have[s]):
                v = pairs[s][t]
                sample = [v[rng.randrange(len(v))] for _ in v]
                acc += sum(y - x for x, y in sample) / len(sample)
            d[s] = 100 * acc / len(have[s])
            boot[s].append(d[s])
        boot["mean"].append((d["clean"] + d["randomized"]) / 2)
    for s in SETTINGS:
        lo, hi = ci(boot[s])
        r = report[s]
        print(
            f"{s}: A {r['A']:.1f}%  B {r['B']:.1f}%  B-A {r['diff']:+.1f} pp  95% CI [{lo:+.1f}, {hi:+.1f}]  pairs {r['pairs']}"
            + (f"  SHORT {len(r['short'])} tasks: {', '.join(r['short'][:6])}" if r["short"] else "")
        )
    lo, hi = ci(boot["mean"])
    diff = (report["clean"]["diff"] + report["randomized"]["diff"]) / 2
    print(
        f"mean of clean and randomized: A {(report['clean']['A'] + report['randomized']['A']) / 2:.1f}%  "
        f"B {(report['clean']['B'] + report['randomized']['B']) / 2:.1f}%  B-A {diff:+.1f} pp  95% CI [{lo:+.1f}, {hi:+.1f}]  "
        f"(paired hierarchical bootstrap, {a.reps} reps: tasks, then paired episodes)"
    )


if __name__ == "__main__":
    main()
