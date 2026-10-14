# WorldArena evaluator used for the reported numbers

Base: [WorldArena 2.0](https://github.com/WorldArena2/WorldArena-2.0) at commit
`5978ce5c81e55b8c8358f4f5966a13ce385ff155` (MIT), tree `video_quality_ood/`, which scores WorldArena 1.0 track 1.
The reported evaluation ran on the tree of the earlier upstream commit `222f7b9`, which is no longer reachable on the
remote. The two trees are the same: all 1096 files of `video_quality_ood/` at `5978ce5` have the git blob hashes
recorded for `222f7b9`:

```
recorded 1096 upstream 1096 same 1096 different 0 only recorded 0 only upstream 0
```

`setup.sh` clones the base and applies the patches in order; a clean clone plus `0001`–`0005` reproduces the evaluator
of the reported numbers file for file (judge weights and configuration paths aside). `0006` fixes a race of the sharded
VLM judge that left the VLM metrics of 19 of the 1000 test videos empty in the reported evaluation; the benchmark README
gives the result with and without it.

| patch | content |
|---|---|
| `0001-run-on-the-gpu-of-each-rank` | every metric model on the GPU of its rank instead of `cuda:0`; the device of `torch.distributed` ranks; detection/tracking and VLM judging sharded over `--rank`/`--world-size`; VLM judging matches `<task>_episode_<n>.mp4` |
| `0002-resume-and-atomic-results` | per-dimension results with resume (psnr and ssim included); atomic result files, which several ranks read while one writes; `WORLDARENA_CACHE_DIR` |
| `0003-robotwin-test-layout` | the WorldArena 1.0 test layout (`test_dataset/video/episodeN.mp4`, generated `<task>_episode_<n>.mp4`); `.jpg` and `.png` frames for PSNR/SSIM |
| `0004-aggregate-join` | per-frame and VLM results joined on one canonical video id; the Photometric Consistency column; the merged VLM summary; the JEPA Similarity column. Without it the CSV misses some of the 15 metrics |
| `0005-optional-action-following` | the module-level import of `action_following` removed: the base ships only its bytecode, so the package does not import. The Action Following dimension keeps its lazy import; it is not one of the 15 EWMScore metrics |
| `0006-vlm-temporary-directory-per-rank` | every rank of the sharded VLM judge writes and removes its own temporary directory; with one shared directory the first rank to finish removed it and the others lost their remaining videos |

None of the patches changes a metric computation, a model or a threshold; `config/config.yaml` of the base is not used
(`write_config.py` writes the configuration of each run).
