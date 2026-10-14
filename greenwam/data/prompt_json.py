# SPDX-License-Identifier: OpenMDW-1.1
"""GreenWAM structured (JSON) prompts: the Cosmos 3 action prompt format the Edge base was trained on, plus one field.

Layout of the rendered prompt (key order matters for the tokenizer, it serialises the dict as JSON):

  {"cinematography": {"framing": "<concat_view template>. <canvas layout caption>"},
   "robot": "<robot_names.json[layout_id]>",
   "actions": [{"time": "0:00-0:01", "description": "<task text>"}],
   "duration": "1s", "fps": 30.0, "resolution": {"H": 736, "W": 544}, "aspect_ratio": "...",
   "mode_description": "Predict the future video and action given the image."}    (added by the pipeline per mode)

Every mode renders this structure: visual modes t2v/i2v/i2v_ar carry it without a mode phrase (the base
has none for them); inverse follows the base and drops the task text, leading with the base phrase plus GreenWAM's
representation instruction (see ``inverse_structured_prompt``).

``robot`` is the only key the base never saw; the dataset name is deliberately absent (source identity reaches the
model through action_layout_id routing, and the name would be a shortcut the deployment never has). Duration uses
the measured frame count of the sample (not the VAE-padded video) and the sample's effective FPS.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.transforms import ACTION_MODE_DESCRIPTIONS

PROMPT_FORMATS = ("text", "json")
ROBOT_NAMES_PATH = Path(__file__).resolve().parents[1] / "registry" / "robot_names.json"
CANVAS_VIEWPOINT = "concat_view"  # DEFAULT_VIEWPOINT_TEMPLATES: "This video contains concatenated views from multiple camera perspectives."


@lru_cache(maxsize=1)
def load_robot_names(path=None):
    doc = json.loads(Path(path or ROBOT_NAMES_PATH).read_text())
    if doc.get("schema_version") != 1 or doc.get("field") != "robot" or not isinstance(doc.get("robot_names"), dict):
        raise ValueError("robot_names.json must be schema_version 1 with a robot_names mapping")
    names = doc["robot_names"]
    if any(not isinstance(v, str) or not v.strip() for v in names.values()):
        raise ValueError("Every robot name must be a nonempty string")
    return dict(names)


def robot_name(layout_id, *, names=None):
    names = load_robot_names() if names is None else names
    try:
        return names[layout_id]
    except KeyError as e:
        raise ValueError(f"No robot name recorded for layout {layout_id!r}; add it to robot_names.json") from e


def validate_prompt_format(value):
    if value not in PROMPT_FORMATS:
        raise ValueError(f"custom.greenwam.text.prompt_format must be one of {PROMPT_FORMATS}, got {value!r}")
    return value


class PromptJsonFormatter(ActionPromptJsonFormatter):
    """Cosmos 3 JSON prompt with the canvas as the framing and the robot as its own field."""

    def __call__(self, data_dict):
        caption = data_dict.get(self.caption_key)
        if not isinstance(caption, str) or caption == "":
            return data_dict
        robot = data_dict.pop("greenwam_robot", None)
        if not isinstance(robot, str) or not robot.strip():
            raise ValueError("GreenWAM JSON prompt requires greenwam_robot (see robot_names.json)")
        layout_caption = (
            data_dict.get("greenwam_canvas", {}).get("layout_caption")
            if isinstance(data_dict.get("greenwam_canvas"), dict)
            else None
        )
        if layout_caption and not data_dict.get("additional_view_description"):
            data_dict["additional_view_description"] = layout_caption
        data_dict[self.viewpoint_key] = (
            CANVAS_VIEWPOINT  # the reader defaults to third_person_view; a canvas is a concat view
        )
        data_dict = super().__call__(data_dict)
        prompt = data_dict[self.caption_key]
        if not isinstance(prompt, dict) or "cinematography" not in prompt:
            raise ValueError("Structured prompt missing after formatting")
        ordered = {}
        for key, value in prompt.items():
            ordered[key] = value
            if key == "cinematography":
                ordered["robot"] = robot.strip()
        data_dict[self.caption_key] = (
            inverse_structured_prompt(ordered) if data_dict.get("mode") == "inverse_dynamics" else ordered
        )
        return data_dict


def inverse_structured_prompt(prompt):
    """Inverse dynamics: the base omits the task text (it describes the target) and leads with its mode phrase.
    GreenWAM's representation instruction (INVERSE_PROMPT / INVERSE_EEF_PROMPT) follows that phrase instead of posing as
    the task, so the pipeline's own mode metadata stays disabled for inverse."""
    actions = prompt.get("actions") or []
    instruction = str(actions[0].get("description", "")).strip() if actions else ""
    description = " ".join(part for part in (ACTION_MODE_DESCRIPTIONS["inverse_dynamics"], instruction) if part)
    return {
        "mode_description": description,
        **{key: value for key, value in prompt.items() if key not in ("actions", "mode_description")},
    }


def prompt_text(caption):
    """Serialised prompt exactly as the tokenizer sees it (dict prompts are JSON-encoded there)."""
    return json.dumps(caption) if isinstance(caption, dict) else caption


_RENDERER = None


def render_structured_prompt(*, layout_caption, robot, text, frames, fps, height, width, float_seconds=False):
    """The same dict the training pipeline produces for a sample, without the mode entry (added by the caller
    through add_action_mode_metadata, exactly as ActionTransformPipeline does). Used for rollout schedules."""
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = ActionPromptJsonFormatter(caption_key="ai_caption", float_seconds=float_seconds)
    f = _RENDERER
    if not isinstance(robot, str) or not robot.strip():
        raise ValueError("Structured prompt requires a robot name")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Structured prompt requires task text")
    if type(frames) is not int or frames < 1 or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("Structured prompt requires positive frames and fps")
    duration = frames / float(fps)
    if float_seconds:
        time_str, duration_str = f"0.00-{duration:.2f}s", f"{duration:.2f}s"
    else:
        time_str, duration_str = (
            f"0:00-{f._format_time_mss(f._round_time_seconds(duration))}",
            f"{f._truncate_seconds(duration)}s",
        )
    framing = f._get_viewpoint_caption({f.viewpoint_key: CANVAS_VIEWPOINT}, layout_caption)
    prompt = {
        "cinematography": {"framing": framing},
        "robot": robot.strip(),
        "actions": [{"time": time_str, "description": f._ensure_sentence(text)}],
        "duration": duration_str,
        "fps": float(fps),
        "resolution": {"H": int(height), "W": int(width)},
        "aspect_ratio": f._get_aspect_ratio(int(width), int(height)),
    }
    cleaned = f._drop_empty_fields(prompt)
    f._raise_if_empty_fields(cleaned)
    return cleaned
