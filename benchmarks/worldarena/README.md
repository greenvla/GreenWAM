# WorldArena 1.0, track 1 (action-driven video generation)

GreenWAM generates the test videos of WorldArena 1.0 track 1 from the first frame and the robot actions, without the
instruction text, autoregressively in blocks, with segments of at most 15 s. This page verifies the released
checkpoint.

## Result

EWMScore over the 1000 test episodes, from `scoring/ewmscore.py`:

| | reported | released evaluator |
|---|---|---|
| capped (motion rule of WorldArena 2.0) | **63.31** | 63.29 |
| uncapped (WorldArena 1.0 board formula) | **63.35** | 63.34 |

- **uncapped**: 100 × the mean over the 15 board metrics of each metric's mean over videos.
- **capped**: the same after Dynamic Degree, Flow Score and Motion Smoothness of every generated video are capped at the
  value of its ground-truth video, so motion beyond the real one earns nothing. The ground truth is evaluated as a model
  with the same evaluator (see Evaluation).

Both are computed from the per-video results without intermediate rounding; averaging the reported per-metric means
after rounding them to two decimals gives 63.36 instead of 63.35.

In the reported evaluation, a race of the VLM judge left 19 of the 1000 videos without their three VLM metrics
(Interaction Quality, Perspectivity, Instruction Following); the reported means of these metrics are over the other 981
videos. Patch 0006 of the released evaluator fixes the race: on the same videos it scores all 1000 and gives 63.29
capped and 63.34 uncapped. Rerunning the VLM step with the fix leaves every other per-video result unchanged.

| metric (reported) | uncapped | capped |
|---|---|---|
| Image Quality | 36.29 | 36.29 |
| Aesthetic Quality | 36.81 | 36.81 |
| JEPA Similarity | 98.41 | 98.41 |
| Dynamic Degree | 21.55 | 21.52 |
| Flow Score | 11.40 | 11.36 |
| Motion Smoothness | 69.45 | 68.84 |
| Subject Consistency | 74.36 | 74.36 |
| Background Consistency | 79.86 | 79.86 |
| Photometric Consistency | 42.70 | 42.70 |
| Interaction Quality ¹ | 68.46 | 68.46 |
| Trajectory Accuracy | 55.97 | 55.97 |
| Depth Accuracy | 98.16 | 98.16 |
| Perspectivity ¹ | 88.64 | 88.64 |
| Instruction Following ¹ | 77.96 | 77.96 |
| Semantic Alignment | 90.31 | 90.31 |

¹ Reported: mean over 981 videos. Released evaluator, all 1000 videos: Interaction Quality 68.32, Perspectivity
88.64, Instruction Following 77.82.

## Checkpoint

The released WorldArena checkpoint (EMA weights) comes with its `config.yaml` and the action
statistics (`absolute/` and `delta_from_q0/` `statistics.json`); the repository README gives the download location.
This page reproduces the numbers above from that checkpoint: rollout of the test set, evaluation, scoring.

## Environment variables

| variable | content |
|---|---|
| `WORLDARENA_TEST_ROOT` | `test_dataset` of the WorldArena 1.0 test set (Hugging Face dataset `WorldArena/WorldArena_Robotwin2.0`) |
| `GREENWAM_STATS` | directory with the released `absolute/` and `delta_from_q0/` `statistics.json` |
| `WAN_VAE_PATH` | `Wan2.2_VAE.pth` of `Wan-AI/Wan2.2-TI2V-5B` (sha256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`) |
| `COSMOS3_EDGE_DIR` | download of `nvidia/Cosmos3-Edge` at revision `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` (text tokenizer) |

## Rollout

```bash
benchmarks/worldarena/rollout.sh <checkpoint_dir> <config.yaml> <test_summary.json> <gen_dir>
```

One process per visible GPU; no text, segments of at most 15 s, 35 sampling steps, guidance 1.0, seed 0. The videos
are written to `<gen_dir>/flat/<task>_episode_<n>.mp4`.
On H100 GPUs the rollout is deterministic: 20 test episodes generated with this code matched the reported videos
frame for frame.

## Evaluation

The evaluator is WorldArena 2.0 at `5978ce5` (MIT; its `video_quality_ood/` tree scores WorldArena 1.0 track 1) with the
patches in `evaluator/patches/`. They change how the evaluator runs, not what it computes: per-rank GPUs and sharding,
resume, atomic result files, the WorldArena 1.0 test layout, the join of per-frame and VLM results into one CSV, an
optional import of the Action Following module (not part of EWMScore), and one temporary directory per rank of the VLM
judge.

```bash
benchmarks/worldarena/evaluator/setup.sh <eval_dir>          # clone, check out 5978ce5, apply the patches
export WA_EVAL_DIR=<eval_dir>
```

The evaluator needs three Python 3.10 environments built from its own requirement files:
`video_quality_ood/requirements.txt` (`WA_PYTHON`), `requirements_worldarena_vlm.txt` (`WA_VLM_PYTHON`) and
`requirements_jedi.txt` (`WA_JEPA_PYTHON`). The judge weights go under `WA_MODELS` with the names below; the sources are
those listed in the evaluator's `config/config.yaml`. `evaluator/judges.sha256` holds the checksums of the single-file
weights the reported evaluation used (`cd $WA_MODELS && sha256sum -c judges.sha256`).

| metric | `$WA_MODELS/` | source |
|---|---|---|
| Semantic Alignment | `Qwen2.5-VL-7B-Instruct/`, `clip-vit-base-patch16/` | `Qwen/Qwen2.5-VL-7B-Instruct`, `openai/clip-vit-base-patch16` |
| Depth Accuracy | `depth-anything/` | `depth-anything/Depth-Anything-V2-Small-hf` |
| Aesthetic Quality | `clip_model/ViT-L-14.pt`, `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth` | OpenAI CLIP ViT-L/14, LAION aesthetic predictor |
| Background Consistency, Action Following | `ViT-B-32.pt` | OpenAI CLIP ViT-B/32 |
| Background Consistency, Dynamic Degree, Flow Score, Subject Consistency | `raft_model/models/raft-things.pth` | RAFT (EvalCrafter models) |
| Photometric Consistency | `Tartan-C-T-TSKH-spring540x960-M.pth` | SEA-RAFT |
| Motion Smoothness | `VFIMamba.pkl` | `MCG-NJU/VFIMamba` |
| Image Quality | `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` | MUSIQ (IQA-PyTorch weights) |
| Subject Consistency | `dino_model/facebookresearch_dino_main/`, `dino_model/dino_vitbase16_pretrain.pth` | `facebookresearch/dino` (Apache-2.0) |
| Interaction Quality, Perspectivity, Instruction Following | `Qwen3-VL-8B-Instruct/` | `Qwen/Qwen3-VL-8B-Instruct` |
| Trajectory Accuracy (detection and tracking) | `sam3/` | `facebook/sam3`: gated, access is granted on request under the SAM license |
| JEPA Similarity | `v-jepa/vith16.pth.tar`, `v-jepa/ssv2-probe.pth.tar` | V-JEPA (`facebookresearch/jepa`): CC BY-NC 4.0, non-commercial use only |

Run `setup.sh` with `WA_MODELS` set, or link the two V-JEPA files into `video_quality_ood/JEDi/pretrained_models/`.

```bash
benchmarks/worldarena/evaluate.sh <model_name> <gen_dir>/flat <test_summary.json> <eval_run_dir> 8
benchmarks/worldarena/evaluate.sh <model_name> <gen_dir>/flat <test_summary.json> <eval_run_dir> 1   # second pass, see below
```

Depth Accuracy and Dynamic Degree can fail at 8 ranks on a network file system; the reported evaluation ran a second
pass with one process, which recomputes only the failed steps and re-aggregates the CSV. The capped score also needs the ground truth evaluated
as a model: run `evaluate.sh` once more on the test videos themselves, linked as `<task>_episode_<n>.mp4` into one
directory.

## Scoring

```bash
python benchmarks/worldarena/scoring/ewmscore.py <eval_run_dir> <gt_eval_run_dir>               # capped and uncapped EWMScore
python benchmarks/worldarena/scoring/compare.py <candidate_run> <base_run> <gt_eval_run_dir>    # paired, with bootstrap intervals
```

`scoring/board.py` places a run on the WorldArena 1.0 board (Hugging Face space `WorldArena/WorldArena`, results in
`worldarena-results/`) raw and capped, with the six category scores. Board entries keep only per-model means, so they
are capped on their means against the ground-truth means, an upper bound of the per-video cap applied to the run itself.
Download the board on a case-sensitive file system: two result files differ only in letter case.
