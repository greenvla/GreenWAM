# SPDX-License-Identifier: OpenMDW-1.1
"""Merge CALVIN evaluation shards written by ``evaluate.py`` into one result.

Counts are summed over shards (sequences, completed subtasks, per-subtask successes and attempts), so the merged
average chain length is exact rather than an average of rounded per-shard values.

    python -m benchmarks.calvin.aggregate out/shard* --output out/result.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def merge(shard_dirs: list[Path]) -> dict:
    records, subtasks = [], {}
    for shard in shard_dirs:
        records += json.loads((shard / "sequences.json").read_text())
        for name, text in json.loads((shard / "result.json").read_text())["subtasks"].items():
            success, total = map(int, re.search(r"\((\d+)/(\d+)\)", text).groups())
            counts = subtasks.setdefault(name, [0, 0])
            counts[0] += success
            counts[1] += total
    indices = [r["index"] for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError("a sequence index occurs in more than one shard")
    completed = [r["completed"] for r in records]
    n = len(completed)
    levels = [sum(c >= k for c in completed) / n for k in range(1, 6)]
    return {
        "benchmark": "calvin",
        "total_sequences": n,
        "completed_subtasks": sum(completed),
        "average_chain_length": sum(completed) / n,
        "levels": {f"{k + 1}": f"{100 * rate:.2f}%" for k, rate in enumerate(levels)},
        "subtasks": {k: f"{100 * s / t:.2f}% ({s}/{t})" for k, (s, t) in sorted(subtasks.items())},
        "shards": [str(s) for s in shard_dirs],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("shards", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    merged = merge(sorted(args.shards))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2))
    print(json.dumps({k: merged[k] for k in ("total_sequences", "average_chain_length", "levels")}))


if __name__ == "__main__":
    main()
