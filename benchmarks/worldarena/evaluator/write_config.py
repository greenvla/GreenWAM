# SPDX-License-Identifier: OpenMDW-1.1
"""Writes the WorldArena evaluator config of one run: data and output paths under the run directory, judge weights
under $WA_MODELS with the file names listed in benchmarks/worldarena/README.md.
usage: write_config.py <model_name> <run_dir> <evaluator video_quality_ood dir> <judge models dir> <out config.yaml>"""

import sys

name, run, vq, m, out = sys.argv[1:6]
raft = f"{m}/raft_model/models/raft-things.pth"
config = f"""model_name: {name}
data:
  gt_path: {run}/data/gt_dataset
  val_base: {run}/data/generated_dataset
data_action_following:
  gt_path: {run}/data_action_following/gt_dataset
  val_base: {run}/data_action_following/generated_dataset
save_path: {run}/output
save_path_action_following: {run}/output_action_following
ckpt:
  action_following: {m}/ViT-B-32.pt
  semantic_alignment:
    caption: {m}/Qwen2.5-VL-7B-Instruct
    CLIP: {m}/clip-vit-base-patch16
  depth_accuracy: {m}/depth-anything
  aesthetic_quality:
    clip: {m}/clip_model/ViT-L-14.pt
    aesthetic_head: {m}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
  background_consistency:
    clip: {m}/ViT-B-32.pt
    raft: {raft}
  dynamic_degree:
    raft: {raft}
  flow_score:
    raft: {raft}
  photometric_smoothness:
    cfg: {vq}/WorldArena/third_party/SEA-RAFT/config/eval/spring-M.json
    model: {m}/Tartan-C-T-TSKH-spring540x960-M.pth
  motion_smoothness:
    model: {m}/VFIMamba.pkl
  image_quality:
    musiq: {m}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
  subject_consistency:
    repo: {m}/dino_model/facebookresearch_dino_main
    weight: {m}/dino_model/dino_vitbase16_pretrain.pth
    model: dino_vitb16
    raft: {raft}
  sam3_model_ckpt: {m}/sam3
  vlm_model: {m}/Qwen3-VL-8B-Instruct
"""
open(out, "w").write(config)
print("wrote", out)
