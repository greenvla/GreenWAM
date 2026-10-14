#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Board metric of a full RoboTwin 2.0 evaluation: mean over the 50 tasks of per-task success, clean and randomized
separately, and the mean of the two.

Inputs are the records of ``evaluate.py`` (``records.jsonl``, or its stdout log): one JSON object per evaluated
episode or skipped seed. A setting counts only when all 50 tasks have ``episodes`` recorded episodes; otherwise it is
printed as PARTIAL, and so is the mean. Several runs of the same evaluation (a continuation after a seed that hung,
a rerun) are merged by a fixed source priority, never by outcome: for each (setting, task) and seed the first source
in the given order that has a record of the seed decides (an episode, or a skip), and the first ``episodes`` seeds
in seed order are counted, which is the official seed sequence with skipped seeds left out. A seed that appears twice
within one file is an error. Records of a broken simulator are dropped rather than counted as expert rejections:
everything after the first CUDA out-of-memory or CUDA error in a file, and a trailing run of at least 20
``expert_error`` skips of a task.

    python benchmarks/robotwin/tally.py 100 out/main out/continued --tsv out/tally.tsv

A source is a file or a directory (all ``records.jsonl`` and ``*.log`` files below it); priority = order on the
command line.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

SETTINGS = ("clean", "randomized")
TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
BROKEN = re.compile(r"OutOfMemoryError|CUDA out of memory|CUDA error")
CASCADE = 20
_NAME_SETTING = re.compile(r"(?:^|[_/.-])(?:demo_)?(clean|randomized)(?=[_/.-]|$)")


def files_of(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted({*source.rglob("records.jsonl"), *source.rglob("*.log")})
    return [source]


def parse(path: Path) -> tuple[dict, list[str]]:
    """-> {(setting, task): {seed: 1 / 0 for an episode, None for a skipped seed}}, notes; raises on a duplicate."""
    raw = path.read_text(errors="ignore").replace("\r", "")
    broken = BROKEN.search(raw)
    cut = broken.start() if broken else len(raw)
    rows, dropped = [], 0
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if "task" not in d or "seed" not in d or not ("skipped" in d or "success" in d):
            continue
        if m.start() > cut:
            dropped += 1
            continue
        rows.append(d)
    configs = {d["config"] for d in rows if d.get("config")}
    named = _NAME_SETTING.search(str(path))
    fallback = configs.pop().replace("demo_", "") if len(configs) == 1 else (named.group(1) if named else None)
    rec, order, notes = collections.defaultdict(dict), collections.defaultdict(list), []
    for d in rows:
        setting = d["config"].replace("demo_", "") if d.get("config") else fallback
        if setting not in SETTINGS:
            raise SystemExit(f"ERROR {path}: cannot tell the setting of {d}")
        key = (setting, d["task"])
        if d["seed"] in rec[key]:
            raise SystemExit(f"ERROR duplicate seed {d['seed']} for {key} within {path}")
        rec[key][d["seed"]] = None if "skipped" in d else int(bool(d["success"]))
        order[key].append((d["seed"], str(d.get("skipped", ""))))
    if dropped:
        notes.append(f"{path}: {dropped} records after a CUDA error dropped")
    for key, seq in order.items():
        run = 0
        while run < len(seq) and seq[-1 - run][1].startswith("expert_error"):
            run += 1
        if run >= CASCADE:
            for seed, _ in seq[len(seq) - run :]:
                del rec[key][seed]
            notes.append(f"{path}: {key[1]}: trailing {run} expert_error skips dropped")
    return rec, notes


def load(sources: list[Path]) -> list[tuple[str, list[dict]]]:
    out = []
    for source in sources:
        parsed = []
        for f in files_of(source):
            rec, notes = parse(f)
            parsed.append(rec)
            for note in notes:
                print(f"  dropped: {note}")
        print(f"source {source}: {len(parsed)} files")
        out.append((str(source), parsed))
    return out


def select(sources, n_ep: int) -> dict:
    """Per (setting, task): {seed: chosen record} restricted to the first n_ep episodes in seed order, plus provenance."""
    out = {}
    for s in SETTINGS:
        for t in TASKS:
            chosen, conflicts = {}, 0
            for tag, recs in sources:
                for rec in recs:
                    for seed, ok in rec.get((s, t), {}).items():
                        if seed in chosen:
                            prev = chosen[seed]
                            conflicts += prev[1] != tag and prev[0] is not None and ok is not None and prev[0] != ok
                            continue
                        chosen[seed] = (ok, tag)
            seeds = [x for x in sorted(chosen) if chosen[x][0] is not None][:n_ep]
            out[(s, t)] = dict(
                episodes={x: chosen[x][0] for x in seeds},
                origin=dict(collections.Counter(chosen[x][1] for x in seeds)),
                conflicts=conflicts,
            )
    return out


def summary(sel: dict, n_ep: int) -> tuple[list[str], dict]:
    lines, means = [], {}
    for s in SETTINGS:
        k = {t: len(sel[(s, t)]["episodes"]) for t in TASKS}
        succ = {t: sum(sel[(s, t)]["episodes"].values()) for t in TASKS}
        have = [t for t in TASKS if k[t]]
        rates = {t: succ[t] / k[t] for t in have}
        m = sum(rates.values()) / max(1, len(rates))
        se = math.sqrt(sum(r * (1 - r) / max(1, k[t] - 1) for t, r in rates.items())) / max(1, len(rates))
        complete = all(k[t] >= n_ep for t in TASKS)
        means[s] = m if complete else None
        short = [f"{t} {succ[t]}/{k[t]}" for t in have if k[t] < n_ep]
        missing = [t for t in TASKS if not k[t]]
        conflicts = sum(sel[(s, t)]["conflicts"] for t in TASKS)
        lines.append(
            f"{'' if complete else 'PARTIAL '}{s}: {100 * m:.2f}% ± {100 * 1.96 * se:.1f} (95 %) over {len(have)}/{len(TASKS)} "
            f"tasks, {sum(k[t] >= n_ep for t in TASKS)} with {n_ep} eps, {sum(k.values())} eps | short: {', '.join(short) or '-'} "
            f"| missing: {', '.join(missing) or '-'} | cross-source outcome conflicts (resolved by priority): {conflicts}"
        )
    mean = 50 * (means["clean"] + means["randomized"]) if all(v is not None for v in means.values()) else None
    lines.append(
        f"mean of clean and randomized: {mean:.2f}%"
        if mean is not None
        else "mean of clean and randomized: n/a (PARTIAL)"
    )
    return lines, dict(clean=means["clean"], randomized=means["randomized"], mean=mean)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", type=int)
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--tsv", type=Path)
    a = ap.parse_args()
    sel = select(load(a.sources), a.episodes)
    lines, _ = summary(sel, a.episodes)
    print("\n".join(lines))
    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("task\tclean\tclean_from\trandomized\trandomized_from\n")
            for t in TASKS:
                c, r = sel[("clean", t)], sel[("randomized", t)]
                fh.write(
                    f"{t}\t{sum(c['episodes'].values())}/{len(c['episodes'])}\t{c['origin']}\t"
                    f"{sum(r['episodes'].values())}/{len(r['episodes'])}\t{r['origin']}\n"
                )


if __name__ == "__main__":
    main()
