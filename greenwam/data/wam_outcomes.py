# SPDX-License-Identifier: OpenMDW-1.1
"""Positive WAM-only admission from a reviewed, complete source-outcome ledger."""

from __future__ import annotations

import re

POLICY = "verified_successful_demo_only_v1"
OUTCOMES = {"success", "failure", "unknown"}
SCOPES = {"emitted_segment", "source_episode"}


def validate_annotation(value, atom_id=None):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"outcome", "outcome_scope", "evidence_ref", "ledger_sha256"}
        or value.get("outcome") not in OUTCOMES
        or value.get("outcome_scope") not in SCOPES
        or not isinstance(value.get("ledger_sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", value["ledger_sha256"]) is None
    ):
        raise ValueError(f"Invalid WAM outcome qualification: {atom_id}")
    ref = value.get("evidence_ref")
    if ref is not None and (not isinstance(ref, str) or not ref.strip()):
        raise ValueError(f"Invalid WAM outcome evidence reference: {atom_id}")
    if value["outcome"] != "unknown" and ref is None:
        raise ValueError(f"Known WAM outcome requires source evidence: {atom_id}")
    return value


def successful(value, atom_id=None):
    value = validate_annotation(value, atom_id)
    return value is not None and value["outcome"] == "success"
