# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks of the WorldArena launch and scoring scripts, with stand-in GPUs, judges and model runs."""

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLLOUT = HERE / "rollout.sh"
EVALUATE = HERE / "evaluate.sh"
COMPARE = HERE / "scoring" / "compare.py"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _rollout(tmp_path: Path, visible: str | None) -> list[list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "launched.txt"
    _stub(bin_dir / "nvidia-smi", 'printf "GPU 0: a\\nGPU 1: b\\nGPU 2: c\\nGPU 3: d\\n"\n')
    _stub(bin_dir / "python", 'printf "%s %s\\n" "$CUDA_VISIBLE_DEVICES" "$*" >> "$PROBE_LOG"\n')
    caller = tmp_path / "caller"
    caller.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", WORLDARENA_TEST_ROOT="test", GREENWAM_STATS="stats", PROBE_LOG=str(log))
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    result = subprocess.run(["bash", str(ROLLOUT), "ckpt", "config.yaml", "episodes.json", "out"], cwd=caller, env=env)
    assert result.returncode == 0
    return sorted(line.split(" ", 1) for line in log.read_text().splitlines())


def test_rollout_keeps_to_the_visible_gpus(tmp_path):
    launched = _rollout(tmp_path, "2,3")
    assert [gpu for gpu, _ in launched] == ["2", "3"]
    assert "--world-size 2" in launched[0][1]


def test_rollout_uses_every_gpu_without_a_restriction(tmp_path):
    assert [gpu for gpu, _ in _rollout(tmp_path, None)] == ["0", "1", "2", "3"]


def test_rollout_resolves_relative_paths_before_changing_directory(tmp_path):
    args = _rollout(tmp_path, "0")[0][1]
    caller = tmp_path / "caller"
    for flag, name in (("--checkpoint-path", "ckpt"), ("--config-file", "config.yaml"), ("--out", "out")):
        assert f"{flag} {caller / name} " in args + " "
    assert f"--test-root {caller / 'test'} " in args + " "


def _run_vlm_step(tmp_path: Path, fail: bool) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Runs ``soft eval_vlm run_vlm_8`` of evaluate.sh on one stand-in judge, after an earlier 8-rank run left a part."""
    src = EVALUATE.read_text()
    functions = "\n".join(
        re.search(rf"^{name}\(\) \{{\n.*?^\}}", src, re.M | re.S).group()
        for name in ("soft", "wait_ranks", "run_vlm_8")
    )
    judge = tmp_path / "judge.py"
    judge.write_text(
        f"#!{sys.executable}\nimport json, os, sys, pathlib\na = sys.argv\n"
        'p = pathlib.Path(a[a.index("--output_root") + 1]) / a[a.index("--model_name") + 1]\n'
        "p.mkdir(parents=True, exist_ok=True)\n"
        'if os.environ["FAIL_PROBE"] == "1": sys.exit(1)\n'
        '(p / "probe_summary_val_all_intern.json").write_text(json.dumps([{"episode": 0, "value": 0.9}]))\n'
    )
    judge.chmod(0o755)
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    out = run / "output_VLM"
    (out / "probe").mkdir(parents=True)
    (out / "probe" / "probe_summary_val_all_intern_rank0.json").write_text(json.dumps([{"episode": 0, "value": 0.1}]))
    env = dict(
        os.environ,
        RUN=str(run),
        OUT_VLM=str(out),
        NPROC="1",
        WA_VLM_PYTHON=str(judge),
        WA_PYTHON=sys.executable,
        MODEL_NAME="probe",
        FLAT0="unused",
        SUMMARY="unused",
        CONFIG_PATH="unused",
        FAILS=str(run / "failures.txt"),
        FAIL_PROBE="1" if fail else "0",
    )
    shell = (
        'set -euo pipefail\nGPUS=(0)\nlog() { :; }\nmark_done() { touch "$RUN/done_$1"; }\n'
        'is_done() { [ -f "$RUN/done_$1" ]; }\n' + functions + "\nsoft eval_vlm run_vlm_8\n"
    )
    return subprocess.run(["bash", "-c", shell], env=env, capture_output=True, text=True), run, out / "probe"


def test_failed_vlm_rank_leaves_the_step_unfinished(tmp_path):
    result, run, _ = _run_vlm_step(tmp_path, fail=True)
    assert result.returncode == 0  # soft steps record the failure and go on
    assert not (run / "done_eval_vlm").exists()
    assert "eval_vlm rc=" in (run / "failures.txt").read_text()


def test_rerun_does_not_merge_rank_parts_of_an_earlier_run(tmp_path):
    result, run, probe = _run_vlm_step(tmp_path, fail=False)
    assert result.returncode == 0 and (run / "done_eval_vlm").exists()
    assert json.loads((probe / "probe_summary_val_all_intern.json").read_text())[0]["value"] == 0.9
    assert not list(probe.glob("*_rank*.json"))


def test_evaluate_resolves_relative_paths_before_changing_directory(tmp_path):
    caller = tmp_path / "caller"
    (caller / "videos").mkdir(parents=True)
    (caller / "summary.json").write_text("[]")
    (tmp_path / "evaluator" / "video_quality_ood").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "nvidia-smi", 'printf "GPU 0: a\\n"\n')
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        WA_EVAL_DIR="../evaluator",
        WA_MODELS="weights",
        WA_PYTHON=sys.executable,
        WA_VLM_PYTHON=sys.executable,
        WA_JEPA_PYTHON=sys.executable,
    )
    result = subprocess.run(
        ["bash", str(EVALUATE), "probe", "videos", "summary.json", "run", "1"],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
    )
    # The stand-in evaluator has no scripts, so its steps fail; the checks that read the paths come before them.
    assert "status.log: No such file" not in result.stderr
    status = (caller / "run" / "status.log").read_text()
    assert re.search(r"flatten v0=\s*0 summary=0", status)
    assert (caller / "run" / "config.yaml").is_file()


def _write_run(path: Path, metrics: list[str], value_of) -> None:
    (path / "csv_results").mkdir(parents=True)
    with (path / "csv_results" / "aggregated_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["Video_ID", *metrics])
        writer.writeheader()
        for i in range(20):
            writer.writerow({"Video_ID": f"episode{i}", **{m: value_of(i, m) for m in metrics}})


def test_compare_scores_both_runs_on_the_same_videos(tmp_path):
    src = COMPARE.read_text()
    namespace: dict = {}
    exec(src[: src.index("cand, base, gt =")], namespace)
    metrics = namespace["M"]
    vlm = {"Interaction Quality", "Perspectivity", "Instruction Following"}

    def candidate(i, m):  # one video lacks the VLM metrics
        return "" if m in vlm and i == 19 else (1.0 if m in vlm else 0.5)

    def base(i, m):  # same values on every video both have, a low score on the video the candidate lacks
        return (0.0 if i == 19 else 1.0) if m in vlm else 0.5

    _write_run(tmp_path / "cand", metrics, candidate)
    _write_run(tmp_path / "base", metrics, base)
    _write_run(tmp_path / "gt", metrics, lambda i, m: 0.5)
    result = subprocess.run(
        [sys.executable, str(COMPARE), *(str(tmp_path / x) for x in ("cand", "base", "gt"))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "GATE EWM capped: cand-base +0.00" in result.stdout
    assert "Interaction Quality: 19 of 20 videos" in result.stdout
