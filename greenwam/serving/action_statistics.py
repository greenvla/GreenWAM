# SPDX-License-Identifier: OpenMDW-1.1
"""Action statistics of a checkpoint: copy of `ActionStatistics` of the training pipeline and the key helpers it uses.
Loading and per-sample lookup only; fitting the statistics is part of training and not of this release."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _layout_id(sample):
    """Source layout identity (semantic schema v2) when the reader emitted it, else None."""
    value = sample.get("action_layout_id")
    if value is None:
        return None
    value = int(torch.as_tensor(value).reshape(-1)[0]) if not isinstance(value, int) else value
    if value <= 0:
        raise ValueError("action_layout_id must be a positive layout identity")
    return value


class MissingStatisticsError(ValueError):
    """A sample's (contract/layout) cell has no fitted statistics."""


def statistics_key(sample):
    """Per-layout statistics key: 'contract/layout' when a layout identity is present, else 'contract'.

    One contract (family) can hold several source layouts whose channels sit in the same shared slots with
    different units, offsets and coverage; pooling them hides those shifts, so statistics are fitted per layout.
    """
    contract = str(int(sample["action_contract_id"]))
    layout = _layout_id(sample)
    return contract if layout is None else f"{contract}/{layout}"


class ActionStatistics:
    def __init__(
        self,
        document,
        *,
        expected_schema=None,
        expected_train=None,
        expected_representation=None,
        expected_sampling=None,
    ):
        doc = json.loads(Path(document).read_text()) if isinstance(document, (str, Path)) else document
        if (
            doc.get("schema_name") != "greenwam_action_statistics"
            or doc.get("schema_version") != 1
            or doc.get("split") != "train"
            or doc.get("kind") != "mean_std"
        ):
            raise ValueError("Expected versioned train-only action statistics")
        for key in ("train_manifest_sha256", "contract_schema_sha256"):
            _hash(doc.get(key), key)
        if expected_schema is not None and doc["contract_schema_sha256"] != expected_schema:
            raise ValueError("Action statistics contract schema mismatch")
        if expected_train is not None and doc["train_manifest_sha256"] != expected_train:
            raise ValueError("Action statistics training manifest mismatch")
        if expected_representation is not None and doc.get("representation") != expected_representation:
            raise ValueError("Action statistics representation mismatch")
        if expected_sampling is not None and doc.get("sampling") != expected_sampling:
            raise ValueError("Action statistics sampling policy mismatch")
        if not isinstance(doc.get("contracts"), Mapping) or not doc["contracts"]:
            raise ValueError("Missing statistics contracts")
        self.document, self.contracts = doc, {}
        self.per_layout = any("/" in str(key) for key in doc["contracts"])
        from cosmos_framework.data.generator.action.utils.action_processing import ActionAffineNormalization

        for key, row in doc["contracts"].items():
            offset, scale = torch.tensor(row["offset"]), torch.tensor(row["scale"])
            counts = torch.tensor(row["count"])
            if offset.ndim != 1 or not offset.numel() or scale.shape != offset.shape or counts.shape != offset.shape:
                raise ValueError(f"Invalid statistics shape for contract {key}")
            if not torch.isfinite(offset).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"Invalid statistics offset/scale for contract {key}")
            if counts.dtype != torch.int64 or (counts < 0).any() or not row.get("contract"):
                raise ValueError(f"Invalid statistics counts/contract {key}")
            self.contracts[key] = (row["contract"], counts, ActionAffineNormalization(offset, scale))

    def for_sample(self, sample):
        if (
            sample.get("greenwam_action_representation") == "joint"
            and sample.get("greenwam_joint_encoding", "absolute") != self.document["representation"]
        ):
            raise ValueError("Action statistics representation differs from joint sample encoding")
        key = statistics_key(sample)
        if key not in self.contracts:
            if self.per_layout and "/" not in key:
                raise ValueError(f"Per-layout action statistics require action_layout_id for contract {key}")
            if not self.per_layout and "/" in key and key.split("/")[0] in self.contracts:
                key = key.split("/")[0]  # legacy per-contract document
            else:
                raise MissingStatisticsError(f"No training statistics for action contract {key}")
        name, counts, normalizer = self.contracts[key]
        action, mask = sample["action"], sample["action_valid_mask"]
        if name != sample["action_contract"] or action.shape[-1] != counts.numel():
            raise ValueError(f"Action statistics contract semantics mismatch: {key}")
        active = torch.as_tensor(mask, dtype=torch.bool).any(dim=0)
        if (active.cpu() & (counts == 0)).any():
            raise ValueError(f"Active unobserved action channel in contract {key}")
        return normalizer
