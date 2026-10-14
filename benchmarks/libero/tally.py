#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Count a LIBERO or LIBERO-plus evaluation (the summary.json files of closed_loop_eval.py), or compare two LIBERO
evaluations paired on the same episodes.

    python benchmarks/libero/tally.py libero OUT [OUT ...]
    python benchmarks/libero/tally.py plus OUT [OUT ...] --classification <LIBERO-plus>/libero/libero/benchmark/task_classification.json
    python benchmarks/libero/tally.py paired --a OUT_A [...] --b OUT_B [...]

Every OUT is searched recursively for summary.json; an episode is keyed by (suite, task, episode). An episode whose
request to the server failed (``error`` set, or no more steps than the 10 warm-up steps) is a failure in the client's
own count. Here it is left out, listed, and the tally exits with status 1: rerun it into another OUT and pass both,
a valid record replaces a failed one. The reported numbers contain no such episode.

``paired``: B minus A over the episodes both evaluations have, with a 95 % interval from a paired hierarchical
bootstrap (tasks resampled with replacement, then the paired episodes within each resampled task).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
AXES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)
WARMUP_STEPS = 10


def failed(episode: dict) -> bool:
    return bool(episode.get("error")) or episode["steps"] <= WARMUP_STEPS


def load(outs: list[Path]) -> tuple[dict, list]:
    """(suite, task, episode) -> success, plus the keys whose only records are failed requests."""
    good, bad = {}, {}
    for out in outs:
        for path in sorted(out.rglob("summary.json")):
            summary = json.loads(path.read_text())
            for task in summary["task_results"]:
                for ep in task["episode_results"]:
                    key = (summary["task_suite"], task["task_id"], ep["episode"])
                    if failed(ep):
                        bad.setdefault(key, f"{path}: {ep.get('error') or 'steps ' + str(ep['steps'])}")
                    else:
                        good.setdefault(key, bool(ep["success"]))
    return good, sorted((k, v) for k, v in bad.items() if k not in good)


def report_failed(bad: list) -> int:
    for key, why in bad[:20]:
        print(f"failed request, not counted: {key} {why}")
    if bad:
        print(f"{len(bad)} episodes with a failed request: rerun them")
    return 1 if bad else 0


def rate(values) -> str:
    values = list(values)
    return f"{100 * sum(values) / max(1, len(values)):6.2f}  ({sum(values)}/{len(values)})"


def tally_libero(outs: list[Path]) -> int:
    good, bad = load(outs)
    for suite in SUITES:
        print(f"{suite:16s} {rate(v for k, v in good.items() if k[0] == suite)}")
    print(f"{'LIBERO':16s} {rate(good.values())}")
    return report_failed(bad)


def tally_plus(outs: list[Path], classification: Path) -> int:
    good, bad = load(outs)
    category = {
        (suite, t["id"] - 1): t["category"]  # task index i <-> classification id i + 1
        for suite, tasks in json.loads(classification.read_text()).items()
        for t in tasks
    }
    axis = collections.defaultdict(list)
    for (suite, task, _), ok in good.items():
        axis[suite].append(ok)
        axis[category[(suite, task)]].append(ok)
    for name in SUITES + AXES:
        print(f"{name:22s} {rate(axis[name])}")
    print(f"{'Total':22s} {rate(good.values())}")
    print(f"tasks not evaluated: {len(category) - len({k[:2] for k in good})}")
    return report_failed(bad)


def paired(outs_a: list[Path], outs_b: list[Path], reps: int, seed: int) -> int:
    (a, bad_a), (b, bad_b) = load(outs_a), load(outs_b)
    tasks = collections.defaultdict(list)
    for key in sorted(set(a) & set(b)):
        tasks[key[:2]].append((a[key], b[key]))
    groups = list(tasks.values())
    pairs = [p for g in groups for p in g]
    diff = sum(y - x for x, y in pairs) / len(pairs)
    rng, boot = random.Random(seed), []
    for _ in range(reps):
        total, n = 0, 0
        for g in (rng.choice(groups) for _ in groups):
            sample = [g[rng.randrange(len(g))] for _ in g]
            total += sum(y - x for x, y in sample)
            n += len(sample)
        boot.append(total / n)
    boot.sort()
    lo, hi = boot[int(0.025 * reps)], boot[int(0.975 * reps) - 1]
    print(
        f"A {rate(x for x, _ in pairs)}  B {rate(y for _, y in pairs)}  B-A {100 * diff:+.2f} pp  "
        f"95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  ({len(pairs)} paired episodes in {len(groups)} tasks; "
        f"only in A {len(set(a) - set(b))}, only in B {len(set(b) - set(a))})"
    )
    return report_failed(bad_a) | report_failed(bad_b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("libero", "plus", "paired"))
    ap.add_argument("outs", nargs="*", type=Path)
    ap.add_argument("--classification", type=Path, help="LIBERO-plus task_classification.json (plus)")
    ap.add_argument("--a", nargs="+", type=Path, help="outputs of evaluation A (paired)")
    ap.add_argument("--b", nargs="+", type=Path, help="outputs of evaluation B (paired)")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "libero":
        sys.exit(tally_libero(args.outs))
    if args.mode == "plus":
        sys.exit(tally_plus(args.outs, args.classification))
    sys.exit(paired(args.a, args.b, args.reps, args.seed))


if __name__ == "__main__":
    main()
