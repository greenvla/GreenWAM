# SPDX-License-Identifier: OpenMDW-1.1
"""WorldArena track 1 (action-driven) rollouts with GreenWAM's own forward_ar contract.

One test episode = first frame + instruction + recorded joint commands. The episode is cut exactly like the training
release (equal chunks of at most ``--cap-seconds``, ``build_robotwin_release._chunks``); every chunk is one whole-block
``forward_ar`` call of ``generate_block_rollout``: memory = 4 frames spread over the generated history (the first
block bootstraps from the real first frame), q0 = the supplied trajectory boundary. When the scaled 30 s budget of a
segment is used up, memory restarts from the last generated frame, as the training planner does
(``plan_whole_atom_rollouts``). Canvas = GreenWAM single_view 736x544 of the 320x240 head camera; physical clock 50/3 Hz;
the prompt of each block is the training JSON prompt of that block (robot, instruction, its own duration).

Output: ``<out>/flat/fixed_scene_task_episode_<N>.mp4`` (640x480, as many frames as GT) for the WorldArena evaluator,
and ``<out>/meta/episode<N>.json`` with the chunk bounds (seam analysis: frames from a real start vs from own history).
"""

from __future__ import annotations

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cosmos_framework.data.generator.action.utils.transforms import add_action_mode_metadata
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.block_rollout import RolloutActionChunk, chunk_rollout_actions, generate_block_rollout
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import disable_runtime_ema_for_frozen_config
from cosmos_framework.utils import log
from cosmos_framework.utils.rollout_partition import scaled_rollout_frames
from greenwam.data.joint_deltas import Q0_CONDITIONING_SCHEMA, Q0_STATE_CODEC, joint_mask_from_adapter
from greenwam.data.prompt_json import prompt_text, render_structured_prompt, robot_name
from greenwam.data.semantic_space import semantic_layout_id
from greenwam.data.shared_projectors import load_shared_projectors
from greenwam.serving.action_statistics import ActionStatistics

LAYOUT_ID = "robotwin2__aloha_agilex__14__14"
FPS = 50 / 3  # RoboTwin writes one frame per save_freq=15 simulation steps of 1/250 s
HEIGHT, WIDTH = 544, 736  # GreenWAM single_view canvas of one 4:3 camera (greenwam_convert canvas_plan)
WRAPPER = "enters the frame to "  # WorldArena wraps the RoboTwin instruction in a fixed scene sentence


def chunks(length: int, max_frames: int) -> list[tuple[int, int]]:
    """Same cut as data_tools build_robotwin_release._chunks: equal consecutive chunks of at most max_frames."""
    count = -(-length // max_frames)
    bounds = [round(k * length / count) for k in range(count + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def instruction(path: Path) -> str:
    text = str(json.loads(path.read_text()).get("instruction") or "").strip()
    cut = text.lower().find(WRAPPER)
    text = text[cut + len(WRAPPER) :].strip() if cut >= 0 else text
    if not text:
        raise ValueError(f"{path}: empty instruction")
    return text[0].upper() + text[1:]


def load_episode(root: Path, eid: int, swap_rb: bool = False) -> dict:
    with h5py.File(root / "data" / "fixed_scene_task" / f"episode{eid}.hdf5", "r") as f:
        joints = np.asarray(f["joint_action/vector"], dtype=np.float32)
    frame = np.asarray(Image.open(root / "first_frame" / "fixed_scene_task" / f"episode{eid}.png").convert("RGB"))
    if swap_rb:  # into the colour layout of the RoboTwin training roots (see --swap-rb)
        frame = np.ascontiguousarray(frame[..., ::-1])
    return dict(
        eid=eid,
        joints=joints,
        first=torch.from_numpy(frame).permute(2, 0, 1).contiguous(),
        text=instruction(root / "instructions" / "fixed_scene_task" / f"episode{eid}.json"),
    )


def to_canvas(chw: torch.Tensor) -> torch.Tensor:
    """resize_fill of the head camera into the single_view canvas (the writer's ffmpeg scale=736:544)."""
    x = F.interpolate(chw.float().unsqueeze(0), size=(HEIGHT, WIDTH), mode="bicubic", align_corners=False)
    return x.squeeze(0).round().clamp(0, 255).to(torch.uint8)


def block_prompt(text: str, frames: int, layout_caption: str | None) -> str:
    caption = render_structured_prompt(
        layout_caption=layout_caption,
        robot=robot_name(LAYOUT_ID),
        text=text,
        frames=int(frames),
        fps=float(FPS),
        height=HEIGHT,
        width=WIDTH,
    )
    return prompt_text(add_action_mode_metadata(caption, "forward_dynamics"))


def action_chunk(
    adapter, stats, layout_int: int, joints: np.ndarray, targets: list[int], q0: int
) -> RolloutActionChunk:
    raw = torch.from_numpy(joints)
    tgt, start = adapter.adapt(raw[targets]), adapter.adapt(raw[q0 : q0 + 1])
    sample = {
        "action": tgt.action,
        "action_valid_mask": tgt.valid_mask,
        "action_availability_mask": tgt.availability_mask,
        "action_q0": start.action,
        "action_q0_valid_mask": start.valid_mask,
        "action_joint_mask": joint_mask_from_adapter(adapter),
        "action_contract": adapter.action_contract,
        "action_contract_id": int(adapter.action_contract_id),
        "action_layout_id": layout_int,
        "raw_action_dim": int(adapter.target_dim),
        "greenwam_joint_encoding": "delta_from_q0",
        "greenwam_action_representation": "joint",
        "greenwam_state_conditioning": Q0_CONDITIONING_SCHEMA,
        "greenwam_q0_codec": Q0_STATE_CODEC,
        "action_q0_timestamp": float(q0) / FPS,
        "action_target_timestamps": torch.tensor([i / FPS for i in targets], dtype=torch.float32),
        "conditioning_fps_action": torch.tensor(FPS),
    }
    chunk = RolloutActionChunk.from_physical_sample(sample, action_statistics=stats)
    chunk.fps = FPS
    return chunk


def segments(length: int, cap_frames: int, budget: int) -> list[list[tuple[int, int]]]:
    """Blocks as (first_target, stop); a new segment starts when the next block would exceed the memory budget."""
    out, used = [[]], 0
    for k, (a, b) in enumerate(chunks(length, cap_frames)):
        a = a + 1 if k == 0 else a  # frame 0 is the observed seed, never a target
        if b <= a:
            continue
        if out[-1] and used + (b - a) > budget:
            out.append([])
            used = 0
        out[-1].append((a, b))
        used += b - a
    return [s for s in out if s]


@torch.no_grad()
def rollout(model, adapter, stats, layout_int, ep, args) -> tuple[torch.Tensor, dict]:
    length = int(ep["joints"].shape[0])
    budget = scaled_rollout_frames(900, fps=FPS, height=240, width=320)
    image, video, plan = to_canvas(ep["first"]), [], []
    for seg_index, blocks in enumerate(segments(length, int(args.cap_seconds * FPS), budget)):
        first, stop = blocks[0][0], blocks[-1][1]
        flat = action_chunk(adapter, stats, layout_int, ep["joints"], list(range(first, stop)), first - 1)
        parts = list(chunk_rollout_actions(flat, block_frames=[b - a for a, b in blocks]))
        spans = [
            dict(
                atom_id=f"episode{ep['eid']}:s{seg_index}:b{k}",
                start_frame=a if k else first - 1,
                end_frame=b,
                text=block_prompt(ep["text"], b - a, args.layout_caption or None),
                text_provenance="source_task",
                provenance=dict(source="worldarena_test_instruction", episode=ep["eid"]),
            )
            for k, (a, b) in enumerate(blocks)
        ]
        schedule = dict(
            schema_version=1,
            mode="forward_ar",
            root="worldarena_test",
            episode_index=int(ep["eid"]),
            camera="head_camera",
            fps=float(FPS),
            start_frame=first,
            end_frame=stop,
            source_frame_stride=1,
            allow_demo_derived=False,
            spans=spans,
        )
        # action-driven track: first frame + actions only -> the empty prompt the model saw under text dropout
        result = generate_block_rollout(
            model,
            initial_image=image,
            action_chunks=parts,
            prompt="" if args.no_text else schedule,
            fps=FPS,
            seed=args.seed + 1000 * seg_index,
            sampling_options={"num_steps": args.num_steps, "guidance": args.guidance},
            reference_frame_budget=900,
            source_resolution=(240, 320),
        )
        future = result.history_rgb[:, 1:]
        video.append(future)
        image = future[:, -1].contiguous()
        plan.append(
            dict(
                segment=seg_index,
                blocks=[[a, b] for a, b in blocks],
                memory=[c.memory_frame_indexes for c in result.chunks],
            )
        )
    frames = torch.cat([to_canvas(ep["first"]).unsqueeze(1), *video], dim=1)
    if frames.shape[1] != length:
        raise RuntimeError(f"episode{ep['eid']}: generated {frames.shape[1]} frames, GT has {length}")
    return frames, dict(
        eid=ep["eid"],
        length=length,
        text=ep["text"],
        plan=plan,
        cap_seconds=args.cap_seconds,
        layout_caption=args.layout_caption,
        no_text=bool(args.no_text),
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
    )


def save(frames: torch.Tensor, path: Path, size: tuple[int, int]) -> None:
    """uint8 [C,T,H,W] -> mp4 at the evaluator size, near-lossless so the codec does not eat image quality."""
    x = F.interpolate(frames.permute(1, 0, 2, 3).float(), size=size[::-1], mode="bicubic", align_corners=False)
    x = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        "30",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, input=x.tobytes(), check=True)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True, help="config.yaml of a GreenWAM run with the RoboTwin layout")
    parser.add_argument("--stats", required=True, help="delta_from_q0 statistics.json of the training release")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument(
        "--episodes", type=Path, required=True, help="summary json (list with gt_path) or a list of ids"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=10.0, help="must equal the cap of the training release")
    parser.add_argument(
        "--layout-caption",
        default="Single camera view.",
        help="canvas caption of the training recipe (custom.greenwam.text.layout_caption); '' for none",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="WorldArena action-driven protocol: no instruction, empty prompt (as under training text dropout)",
    )
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ema-weights", action="store_false", dest="use_ema_weights")
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        help="the RoboTwin LeRobot roots (convert_robotwin_hdf5.py) carry R and B swapped: feed the first "
        "frame in that layout and swap the generated frames back to true RGB before writing",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    items = json.loads(args.episodes.read_text())
    ids = [int(Path(i["gt_path"]).stem[len("episode") :]) if isinstance(i, dict) else int(i) for i in items]
    ids = ids[args.rank :: args.world_size]
    setup = OmniSetupOverrides.model_validate(
        {
            "checkpoint_path": args.checkpoint_path,
            "output_dir": str(args.out / f"_setup_rank{args.rank}"),
            "config_file": args.config_file,
            "use_ema_weights": bool(args.use_ema_weights),
            "dp_shard_size": 1,
            "dp_replicate_size": 1,
            "use_torch_compile": False,
            "sampler": "unipc",
            "guardrails": False,
        }
    ).build_setup()
    init_output_dir(setup.output_dir)
    model = OmniInference.create(disable_runtime_ema_for_frozen_config(setup)).model
    model.eval()
    adapter, layout_int = load_shared_projectors().resolve(LAYOUT_ID), semantic_layout_id(LAYOUT_ID)
    if layout_int is None:
        raise RuntimeError(f"{LAYOUT_ID} has no semantic layout id; the extension registry is not attached")
    stats = ActionStatistics(Path(args.stats))
    log.info(
        f"[wa-rollout] rank {args.rank}/{args.world_size}: {len(ids)} episodes, cap {args.cap_seconds} s, "
        f"steps {args.num_steps}, guidance {args.guidance}, ckpt {args.checkpoint_path}"
    )
    for eid in ids:
        target = args.out / "flat" / f"fixed_scene_task_episode_{eid}.mp4"
        if target.exists():
            continue
        ep = load_episode(args.test_root, eid, args.swap_rb)
        frames, meta = rollout(model, adapter, stats, int(layout_int), ep, args)
        if args.swap_rb:
            frames = frames.flip(0)
        meta["swap_rb"] = bool(args.swap_rb)
        save(frames, target, (640, 480))
        (args.out / "meta").mkdir(parents=True, exist_ok=True)
        (args.out / "meta" / f"episode{eid}.json").write_text(json.dumps(meta) + "\n")
        log.info(
            f"[wa-rollout] episode{eid}: {meta['length']} frames, "
            f"{sum(len(s['blocks']) for s in meta['plan'])} blocks, {len(meta['plan'])} segments"
        )


if __name__ == "__main__":
    main()
