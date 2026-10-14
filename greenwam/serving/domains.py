# SPDX-License-Identifier: OpenMDW-1.1
"""Embodiment-domain slot of a layout: copy of `resolve_domain_id` of the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

DOMAIN_PROPOSAL = Path(__file__).resolve().parent.parent / "registry" / "domain_proposal.json"


def resolve_domain_id(layout_id: str, source: str | None = None) -> int:
    """Domain of a layout: the upstream slot if the source matches, else the only
    free one (31).

    `num_embodiment_domains = 32` (`configs/base/defaults/model_config.py:318`),
    and the table is taken up to 30 (`domain_utils.py:6-58`), so there is exactly one
    free slot. This is not a domain allocation but a stub that lets the config
    build; the allocation is in `registry/domain_proposal.json`.
    """
    doc = json.loads(Path(DOMAIN_PROPOSAL).read_text(encoding="utf-8"))
    if source:
        match = (doc.get("release_match") or {}).get(source)
        if match:
            return int(match[1])
    free = doc.get("free_slots") or [31]
    return int(free[0])
