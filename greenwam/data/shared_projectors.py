# SPDX-License-Identifier: OpenMDW-1.1
"""Structured family adapters (family schema v2), with explicit compatibility with the compact v1 schema.

Each family owns fixed legs/body/arms/hands slots and one projector pair.
Layouts share established coordinates and use typed slots for native additions.
Schema hashes distinguish old compact targets from reordered full vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SHARED_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "registry" / "family_projectors_v2.json"
# Post-training layouts added on top of the frozen GreenWAM registry, which stays byte-identical (so its
# checkpoint-schema digest does too); see SharedProjectorRegistry.add_layout_extensions.
# A benchmark release may carry its own layout extensions (its layout and admitted atom count); its launcher points here.
DEFAULT_LAYOUT_EXTENSIONS_PATH = Path(
    os.environ.get("GREENWAM_LAYOUT_EXTENSIONS") or DEFAULT_SHARED_REGISTRY_PATH.with_name("layout_extensions.json")
)
LEGACY_ACTION_BANK_SIZE = 45


class SharedProjectorError(ValueError):
    """Invalid family registry or unavailable source semantic adapter."""


@dataclass(frozen=True)
class AdaptedState:
    action: Any
    availability_mask: Any
    valid_mask: Any


@dataclass(frozen=True)
class SourceStateAdapter:
    layout_id: str
    family_id: str
    action_contract_id: int
    family_key: str
    roots: int
    recipe: Mapping[str, Any]

    @property
    def target_dim(self) -> int:
        return int(self.recipe["target_dim"])

    @property
    def native_width(self) -> int:
        return self.target_dim

    @property
    def action_contract(self) -> str:
        prefix = "bodyparts_v2/" if self.recipe.get("contract_version") == 2 else "state_future_v1/"
        return prefix + self.family_id

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(self.recipe["source_columns"])

    @property
    def source_column_shapes(self) -> Mapping[str, Any]:
        return self.recipe["source_column_shapes"]

    @property
    def ready(self) -> bool:
        return bool(self.recipe["runtime_semantic_gather_available"] and self.available_target_slots)

    @property
    def available_target_slots(self) -> tuple[int, ...]:
        """Statically usable targets; pending calibration cannot count as supervision."""
        grip = self.recipe["source_adapter_conditions"].get("gripper") or {}
        masked = set(grip.get("source_indices") or ()) if grip.get("runtime_pending") else set()
        return tuple(
            slot
            for index, slot in zip(self.recipe["source_indices"], self.recipe["destination_slots"])
            if self.recipe["availability_mask"][slot] and index not in masked
        )

    @property
    def source_local_fallback(self) -> Any:
        return self.recipe.get("source_local_fallback")

    def adapt_columns(self, columns: Mapping[str, Any], *, pair_valid: Any = None) -> AdaptedState:
        import torch

        values = []
        for key in self.source_columns:
            value = torch.as_tensor(columns[key])
            width = math.prod(self.source_column_shapes[key])
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            if value.ndim != 2 or value.shape[-1] != width:
                raise SharedProjectorError(f"{self.layout_id}: invalid source column shape for {key}")
            values.append(value)
        return self.adapt(torch.cat(values, dim=-1), pair_valid=pair_valid)

    def adapt(self, state: Any, *, pair_valid: Any = None) -> AdaptedState:
        """Adapt [T,D_source], zeroing unavailable/invalid values without clipping."""
        import torch

        if not self.ready:
            raise SharedProjectorError(f"{self.layout_id}: pending semantic map in {self.family_id}")
        state = torch.as_tensor(state).to(dtype=torch.float32)
        if state.ndim != 2 or state.shape[-1] != self.recipe["raw_state_width"]:
            raise SharedProjectorError(f"{self.layout_id}: expected [T,{self.recipe['raw_state_width']}] state")
        indices = torch.as_tensor(self.recipe["source_indices"], device=state.device, dtype=torch.long)
        slots = torch.as_tensor(self.recipe["destination_slots"], device=state.device, dtype=torch.long)
        selected = state.index_select(-1, indices)
        scale = state.new_tensor(self.recipe["scale"])
        offset = state.new_tensor(self.recipe["offset"])
        values = selected * scale + offset
        selected_valid = torch.isfinite(selected) & torch.isfinite(values)

        grip = self.recipe["source_adapter_conditions"].get("gripper") or {}
        grip_indices = set(grip.get("source_indices") or ())
        for j, source_index in enumerate(self.recipe["source_indices"]):
            if source_index not in grip_indices:
                continue
            if grip.get("runtime_pending"):
                selected_valid[:, j] = False
            elif "closed_value" in grip and "open_value" in grip:
                lo, hi = sorted((float(grip["closed_value"]), float(grip["open_value"])))
                selected_valid[:, j] &= (selected[:, j] >= lo) & (selected[:, j] <= hi)

        available = torch.tensor(self.recipe["availability_mask"], device=state.device, dtype=torch.bool)
        action = state.new_zeros((state.shape[0], self.target_dim))
        valid = torch.zeros_like(action, dtype=torch.bool)
        action[:, slots] = values
        valid[:, slots] = selected_valid
        valid &= available.unsqueeze(0)
        if pair_valid is not None:
            pair_valid = torch.as_tensor(pair_valid, device=state.device)
            if pair_valid.dtype != torch.bool or pair_valid.shape != (state.shape[0],):
                raise SharedProjectorError("pair_valid must be bool [T]")
            valid &= pair_valid[:, None]
        action = torch.where(valid, action, torch.zeros_like(action))
        return AdaptedState(action, available, valid)

    def unadapt(self, action: Any, *, valid_mask: Any) -> AdaptedState:
        """Undo source gather/scale without inventing unmapped native values."""
        import torch

        action = torch.as_tensor(action).to(dtype=torch.float32)
        valid_mask = torch.as_tensor(valid_mask, device=action.device)
        if action.ndim != 2 or action.shape[-1] != self.target_dim:
            raise SharedProjectorError(f"{self.layout_id}: expected [T,{self.target_dim}] action")
        if valid_mask.dtype != torch.bool or valid_mask.shape != action.shape:
            raise SharedProjectorError("valid_mask must be bool with the action shape")
        source = torch.tensor(self.recipe["source_indices"], device=action.device)
        slots = torch.tensor(self.recipe["destination_slots"], device=action.device)
        scale = action.new_tensor(self.recipe["scale"])
        if (scale == 0).any() or len(set(self.recipe["source_indices"])) != len(source):
            raise SharedProjectorError("source mapping is not invertible")
        selected = (action[:, slots] - action.new_tensor(self.recipe["offset"])) / scale
        available_slots = torch.zeros(self.target_dim, device=action.device, dtype=torch.bool)
        available_slots[list(self.available_target_slots)] = True
        valid = valid_mask[:, slots] & available_slots[slots] & torch.isfinite(selected)
        raw = action.new_zeros((action.shape[0], self.recipe["raw_state_width"]))
        raw_valid = torch.zeros_like(raw, dtype=torch.bool)
        raw_available = torch.zeros(raw.shape[-1], device=raw.device, dtype=torch.bool)
        raw[:, source] = torch.where(valid, selected, 0.0)
        raw_valid[:, source] = valid
        raw_available[source] = available_slots[slots]
        return AdaptedState(raw, raw_available, raw_valid)


class SharedProjectorRegistry:
    def __init__(self, doc: Mapping[str, Any]) -> None:
        if (doc.get("schema_version"), doc.get("accepted_decision")) not in {(1, "schema_v1"), (2, "schema_v2")}:
            raise SharedProjectorError("expected family schema v1 or v2 with its accepted_decision token")
        self.doc = doc
        self.bank_offset = LEGACY_ACTION_BANK_SIZE
        groups = doc.get("groups") or []
        ids = [g["cluster_id"] for g in groups]
        expected = [f"C{i:02d}" for i in range(1, 22)]
        if sorted(ids) != expected:
            raise SharedProjectorError("expected the 21 unique accepted families C01..C21")
        self.groups = {g["cluster_id"]: g for g in groups}
        if doc["schema_version"] == 2:
            for group in groups:
                self._validate_body_parts(group)
        self.family_count = len(groups)
        self.bank_size = self.bank_offset + self.family_count
        self.layouts: dict[str, SourceStateAdapter] = {}
        self.extensions = None
        self.extensions_sha256 = None
        self.extension_semantic_layout_ids: dict[str, int] = {}
        for layout in doc.get("layouts") or []:
            self._add_layout(layout)
        if len(self.layouts) != 43 or sum(a.roots for a in self.layouts.values()) != 1217:
            raise SharedProjectorError("accepted registry must cover 43 layouts and 1217 roots")
        # Readiness may improve when a source bridge is evidenced. Check the
        # declared accounting against actual recipes instead of requiring four
        # roots to remain pending forever.
        counts = doc.get("counts") or {}
        actual = self.coverage_report()["counts"]
        for key, value in {
            "families": actual["families"],
            "layouts": actual["layouts"],
            "sources": actual["sources"],
            "roots": actual["assigned_roots"],
            "admitted_atoms": actual["assigned_admitted_atoms"],
            "roots_with_shared_semantic_gather": actual["executable_roots"],
            "pending_semantic_adapter_roots": actual["pending_roots"],
        }.items():
            if counts.get(key) != value:
                raise SharedProjectorError(f"coverage count {key}: declared {counts.get(key)}, actual {value}")

    @staticmethod
    def _validate_body_parts(group: Mapping[str, Any]) -> None:
        parts = group.get("body_parts") or {}
        if set(parts) != {"legs", "body", "arms", "hands"}:
            raise SharedProjectorError("body blocks must declare legs/body/arms/hands")
        cursor = 0
        for name in ("legs", "body", "arms", "hands"):
            block = parts[name]
            if (
                set(block) != {"start", "stop"}
                or any(type(block[k]) is not int for k in block)
                or block["start"] != cursor
                or block["stop"] < cursor
            ):
                raise SharedProjectorError(f"{group['cluster_id']}: noncontiguous {name} block")
            cursor = block["stop"]
        names = group.get("slot_names") or []
        if (
            cursor != group["compact_target_dim"]
            or len(names) != cursor
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != cursor
        ):
            raise SharedProjectorError("body block width and unique slot names must match family width")

    @property
    def max_target_dim(self) -> int:
        return max(g["compact_target_dim"] for g in self.groups.values())

    def _add_layout(self, layout: Mapping[str, Any]) -> None:
        lid = layout["layout_id"]
        family_id = layout["cluster_id"]
        if lid in self.layouts or family_id not in self.groups:
            raise SharedProjectorError(f"duplicate layout or unknown family: {lid}")
        recipe = layout["recipe"]
        self._validate_recipe(lid, recipe, self.groups[family_id])
        if self.doc["schema_version"] == 2:
            names = self.groups[family_id]["slot_names"]
            expected_names = [names[slot] for slot in recipe["destination_slots"]]
            if recipe.get("canonical_names") != expected_names:
                raise SharedProjectorError(f"{lid}: source names differ from family slot names")
            if 0 in recipe["scale"] or len(set(recipe["source_indices"])) != len(recipe["source_indices"]):
                raise SharedProjectorError(f"{lid}: v2 mapping must be invertible")
        self.layouts[lid] = SourceStateAdapter(
            lid,
            family_id,
            self.bank_offset + int(family_id[1:]) - 1,
            self.groups[family_id]["key"],
            int(layout["roots"]),
            recipe,
        )
        if recipe["runtime_semantic_gather_available"] and not self.layouts[lid].ready:
            raise SharedProjectorError(f"{lid}: declared ready map has no active targets")

    def add_layout_extensions(self, extensions: Mapping[str, Any]) -> None:
        """Add post-training layouts to existing families without touching the frozen registry.

        The extension names the digest of the exact registry it extends; each layout reuses one frozen
        semantic layout id whose mapping is checked slot-for-slot by ``semantic_layout_id``.
        """
        if self.extensions is not None:
            raise SharedProjectorError("layout extensions are already attached")
        if (extensions.get("schema_name"), extensions.get("schema_version")) != ("layout_extensions", 1):
            raise SharedProjectorError("expected layout_extensions v1")
        if extensions.get("parent_shared_projectors_sha256") != canonical_sha256(self.doc):
            raise SharedProjectorError("layout extensions were written for a different frozen registry")
        rows = extensions.get("layouts") or []
        if not rows:
            raise SharedProjectorError("layout extensions must add at least one layout")
        for layout in rows:
            semantic = layout.get("semantic_layout_id")
            if isinstance(semantic, bool) or not isinstance(semantic, int) or semantic < 1:
                raise SharedProjectorError(f"{layout.get('layout_id')}: extension needs a frozen semantic_layout_id")
            self._add_layout({key: value for key, value in layout.items() if key != "semantic_layout_id"})
            self.extension_semantic_layout_ids[layout["layout_id"]] = semantic
        self.extensions = extensions
        self.extensions_sha256 = canonical_sha256(extensions)

    @staticmethod
    def _validate_recipe(lid: str, recipe: Mapping[str, Any], group: Mapping[str, Any]) -> None:
        width, dim = recipe["raw_state_width"], recipe["target_dim"]
        source, slots = recipe["source_indices"], recipe["destination_slots"]
        if dim != group["compact_target_dim"] or width <= 0:
            raise SharedProjectorError(f"{lid}: invalid family dimension")
        if any(not isinstance(i, int) or not 0 <= i < width for i in source):
            raise SharedProjectorError(f"{lid}: source index outside declared state width")
        if any(not isinstance(i, int) or not 0 <= i < dim for i in slots) or len(set(slots)) != len(slots):
            raise SharedProjectorError(f"{lid}: invalid destination slots")
        if not (len(source) == len(slots) == len(recipe["scale"]) == len(recipe["offset"])):
            raise SharedProjectorError(f"{lid}: inconsistent gather/scale/scatter lengths")
        availability = recipe["availability_mask"]
        if len(availability) != dim or set(availability) - {0, 1}:
            raise SharedProjectorError(f"{lid}: invalid availability mask")
        if {i for i, flag in enumerate(availability) if flag} - set(slots):
            raise SharedProjectorError(f"{lid}: available slot has no semantic gather")
        if not all(math.isfinite(v) for v in recipe["scale"] + recipe["offset"]):
            raise SharedProjectorError(f"{lid}: nonfinite adapter coefficients")
        columns = recipe["source_columns"]
        shapes = recipe["source_column_shapes"]
        if len(set(columns)) != len(columns) or sum(math.prod(shapes[k]) for k in columns) != width:
            raise SharedProjectorError(f"{lid}: source column widths/order do not match raw width")
        if recipe["native_action_used"] or recipe["validity_contract"].get("clip"):
            raise SharedProjectorError(f"{lid}: native action or clipping violates the family contract")

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> SharedProjectorRegistry:
        return cls(doc)

    def resolve(self, layout_id: str) -> SourceStateAdapter:
        try:
            return self.layouts[layout_id]
        except KeyError as exc:
            raise SharedProjectorError(f"unknown layout {layout_id!r}") from exc

    def coverage_report(self) -> dict[str, Any]:
        """Audit the accepted data snapshot without treating assignment as readiness.

        Executable means at least one semantically mapped state target; it does
        not certify all motors, per-frame validity, or cross-source calibration.
        Counts come from recipes, not the registry's cached summary.
        """
        layouts = []
        for layout in self.doc["layouts"]:
            adapter = self.layouts[layout["layout_id"]]
            recipe = adapter.recipe
            conditions = recipe["source_adapter_conditions"]
            grip = conditions.get("gripper") or {}
            available = sorted(adapter.available_target_slots)
            layouts.append(
                dict(
                    layout_id=adapter.layout_id,
                    family_id=adapter.family_id,
                    action_contract_id=adapter.action_contract_id,
                    source=layout["sources"][0],
                    roots=adapter.roots,
                    admitted_atoms=int(layout["atoms"]),
                    executable=adapter.ready,
                    target_dim=adapter.target_dim,
                    available_target_slots=available,
                    masked_target_slots=sorted(set(range(adapter.target_dim)) - set(available)),
                    gripper_codec_pending=bool(grip.get("runtime_pending")),
                    pending_gripper_source_indices=list(grip.get("source_indices") or [])
                    if grip.get("runtime_pending")
                    else [],
                    pending_semantic_map=conditions.get("pending_semantic_map"),
                    calibration_fully_validated=bool(recipe.get("runtime_calibration_fully_validated")),
                    dropped_source_indices=sorted(
                        set(range(recipe["raw_state_width"])) - set(recipe["source_indices"])
                    ),
                    dropped_groups=recipe.get("dropped_groups", []),
                )
            )
        sources = []
        for source in sorted({row["source"] for row in layouts}):
            rows = [row for row in layouts if row["source"] == source]
            sources.append(
                dict(
                    source=source,
                    family_ids=sorted({row["family_id"] for row in rows}),
                    roots=sum(row["roots"] for row in rows),
                    admitted_atoms=sum(row["admitted_atoms"] for row in rows),
                    executable_roots=sum(row["roots"] for row in rows if row["executable"]),
                    executable_admitted_atoms=sum(row["admitted_atoms"] for row in rows if row["executable"]),
                    all_layouts_executable=all(row["executable"] for row in rows),
                    pending_layout_ids=[row["layout_id"] for row in rows if not row["executable"]],
                    masked_gripper_roots=sum(row["roots"] for row in rows if row["gripper_codec_pending"]),
                )
            )
        ready = [row for row in layouts if row["executable"]]
        pending = [row for row in layouts if not row["executable"]]
        grippers = [row for row in layouts if row["gripper_codec_pending"]]
        counts = dict(
            families=self.family_count,
            layouts=len(layouts),
            sources=len(sources),
            assigned_roots=sum(row["roots"] for row in layouts),
            assigned_admitted_atoms=sum(row["admitted_atoms"] for row in layouts),
            executable_layouts=len(ready),
            executable_roots=sum(row["roots"] for row in ready),
            executable_admitted_atoms=sum(row["admitted_atoms"] for row in ready),
            pending_layouts=len(pending),
            pending_roots=sum(row["roots"] for row in pending),
            pending_admitted_atoms=sum(row["admitted_atoms"] for row in pending),
            masked_gripper_layouts=len(grippers),
            masked_gripper_roots=sum(row["roots"] for row in grippers),
            masked_gripper_admitted_atoms=sum(row["admitted_atoms"] for row in grippers),
        )
        return dict(
            schema_version=1,
            accepted_decision=self.doc["accepted_decision"],
            admitted_manifest_sha256=self.doc.get("admitted_manifest_sha256"),
            all_admitted_sources_executable=not pending,
            scope="accepted data snapshot; executable core coverage, not full motor/calibration coverage",
            counts=counts,
            sources=sources,
            layouts=layouts,
        )

    def require_complete_source_coverage(self) -> None:
        """Fail explicitly when all-source action training is requested."""
        report = self.coverage_report()
        if not report["all_admitted_sources_executable"]:
            pending = [row["layout_id"] for row in report["layouts"] if not row["executable"]]
            raise SharedProjectorError(
                f"Shared action coverage unresolved: {report['counts']['pending_roots']} roots / "
                f"{report['counts']['pending_admitted_atoms']} admitted atoms: {', '.join(pending)}"
            )


def canonical_sha256(doc: Mapping[str, Any]) -> str:
    """Digest used by the checkpoint schema for registry documents."""
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_layout_extensions(path: str | Path = DEFAULT_SHARED_REGISTRY_PATH) -> Mapping[str, Any] | None:
    """Extensions attach only to the default frozen registry; explicit historical registries stay as they were."""
    if Path(path).resolve() != DEFAULT_SHARED_REGISTRY_PATH.resolve() or not DEFAULT_LAYOUT_EXTENSIONS_PATH.is_file():
        return None
    return json.loads(DEFAULT_LAYOUT_EXTENSIONS_PATH.read_text(encoding="utf-8"))


def load_shared_projectors(path: str | Path = DEFAULT_SHARED_REGISTRY_PATH) -> SharedProjectorRegistry:
    registry = SharedProjectorRegistry.from_doc(json.loads(Path(path).read_text(encoding="utf-8")))
    extensions = default_layout_extensions(path)
    if extensions is not None:
        registry.add_layout_extensions(extensions)
    return registry
