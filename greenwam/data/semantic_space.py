# SPDX-License-Identifier: OpenMDW-1.1
"""Lossless semantic gather/scatter for fixed transport128 actions."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SEMANTIC_SPACE_PATH = Path(__file__).resolve().parent.parent / "registry" / "semantic_space_v1.json"
# A benchmark that brings a new body with its own action contracts (a new source of an existing family is a layout
# extension instead, see shared_projectors.py) ships a semantic extension file; its launcher points this variable at
# it. The frozen space stays byte-identical: the extension merges in only on top of the default space.
SEMANTIC_EXTENSIONS_ENV = "GREENWAM_SEMANTIC_EXTENSIONS"


def semantic_layout_id(layout_id, registry=None):
    """Frozen semantic id of a layout, or the id a post-training layout extension reuses.

    An extension declares an existing id; it is accepted only if that id's mapping for the
    layout's family contract covers exactly the same family slots. None = unmapped layout.
    """
    space = SemanticSpace.load()
    found = space.layouts.get(layout_id)
    if found is not None:
        return found
    if registry is None:
        from greenwam.data.shared_projectors import load_shared_projectors

        registry = load_shared_projectors()
    declared = registry.extension_semantic_layout_ids.get(layout_id)
    if declared is None:
        return None
    adapter = registry.resolve(layout_id)
    row = space.mapping(adapter.action_contract_id, declared)
    if sorted(row["source_slots"]) != sorted(adapter.recipe["destination_slots"]):
        raise ValueError(f"{layout_id}: semantic layout {declared} maps other family slots")
    return declared


class SemanticSpace:
    def __init__(self, doc: Mapping[str, Any]) -> None:
        version = doc.get("schema_version")
        if version not in (1, 2) or doc.get("transport_dim") != 128:
            raise ValueError("expected semantic schema transport_dim=128")
        self.doc = dict(doc)
        self.version = version
        self.semantic_dim = int(doc["semantic_dim"])
        self.transport_dim = 128
        names, parts, raw = doc.get("slot_names"), doc.get("slot_body_part_ids"), doc.get("contracts")
        if (
            not isinstance(names, list)
            or len(names) != self.semantic_dim
            or len(set(names)) != len(names)
            or not isinstance(parts, list)
            or len(parts) != self.semantic_dim
            or set(parts) - {0, 1, 2, 3}
            or not isinstance(raw, Mapping)
        ):
            raise ValueError("invalid semantic slot registry")
        self._parts = tuple(parts)
        self.contracts = {}
        for key, row in raw.items():
            try:
                cid = int(key)
            except (TypeError, ValueError) as e:
                raise ValueError("contract IDs must be integers") from e
            if version == 1:
                self._validate_row(row, f"{key}: contract")
            self.contracts[cid] = row
        from greenwam.data.state_eef import STATE_EEF_ABSOLUTE_BANK_OFFSET, STATE_EEF_BANK_OFFSET

        eef_families = STATE_EEF_ABSOLUTE_BANK_OFFSET - STATE_EEF_BANK_OFFSET
        # joint IDs 45..65, relative EEF IDs [66, 80), absolute EEF IDs mirrored from 80; only declared EEF families exist
        relative = {cid for cid in self.contracts if STATE_EEF_BANK_OFFSET <= cid < STATE_EEF_ABSOLUTE_BANK_OFFSET}
        absolute = {cid for cid in self.contracts if cid >= STATE_EEF_ABSOLUTE_BANK_OFFSET}
        joints = {cid for cid in self.contracts if cid < STATE_EEF_BANK_OFFSET}
        if (
            joints != set(range(45, STATE_EEF_BANK_OFFSET))
            or not relative
            or absolute != {cid + eef_families for cid in relative}
            or max(relative) >= STATE_EEF_ABSOLUTE_BANK_OFFSET
        ):
            raise ValueError("semantic space must retain joint IDs 45..65 and paired relative/absolute EEF IDs from 66")
        cursor = 0
        for part_id, name in enumerate(("legs", "body", "arms", "hands")):
            block = doc.get("body_parts", {}).get(name, {})
            if block.get("start") != cursor or not isinstance(block.get("stop"), int) or block["stop"] < cursor:
                raise ValueError("body part blocks must be contiguous")
            if any(value != part_id for value in self._parts[cursor : block["stop"]]):
                raise ValueError("body part block disagrees with semantic slots")
            cursor = block["stop"]
        if cursor != self.semantic_dim:
            raise ValueError("body part blocks must cover semantic space")
        self.contract_ids = tuple(sorted(self.contracts))
        self.contract_id_to_index = {x: i for i, x in enumerate(self.contract_ids)}
        self.id_to_index = self.contract_id_to_index
        self.layouts = (
            {str(name): int(identity) for name, identity in doc.get("layouts", {}).items()} if version == 2 else {}
        )
        self._mappings = {}
        if version == 2:
            if not self.layouts or set(self.layouts.values()) != set(range(1, len(self.layouts) + 1)):
                raise ValueError("layouts must have stable contiguous IDs from 1")
            for row in doc.get("mappings", []):
                cid, lid = row.get("contract_id"), row.get("layout_id")
                if (
                    not isinstance(cid, int)
                    or not isinstance(lid, int)
                    or cid not in self.contracts
                    or lid not in self.layouts.values()
                ):
                    raise ValueError("invalid mapping pair")
                self._validate_row(row, f"{cid}/{lid}: mapping")
                if doc.get("space_contract") in ("typed_bodyparts69_v1", "typed_bodyparts79_v2"):
                    from greenwam.data.state_eef import STATE_EEF_ABSOLUTE_BANK_OFFSET, STATE_EEF_BANK_OFFSET

                    expected = 0 if cid < STATE_EEF_BANK_OFFSET else 1 if cid < STATE_EEF_ABSOLUTE_BANK_OFFSET else 2
                    if row.get("representation_id") != expected or row.get("context_id") != lid:
                        raise ValueError("invalid layout/representation context")
                    if len(row.get("source_roles", [])) != len(row["source_slots"]):
                        raise ValueError("missing source role provenance")
                if (cid, lid) in self._mappings:
                    raise ValueError("duplicate contract/layout mapping")
                shared = row.get("shared_mask")
                if shared is None:
                    row = dict(row, shared_mask=[True] * len(row["source_slots"]))
                elif len(shared) != len(row["source_slots"]) or any(not isinstance(x, bool) for x in shared):
                    raise ValueError(f"{cid}/{lid}: shared_mask must align with source_slots")
                self._mappings[cid, lid] = row
            if not self._mappings:
                raise ValueError("schema v2 requires mappings")
        self._part_tables = {}
        self._shared_tables = {}
        # dense (contract, layout) lookup: contracts 45..max, layouts 0..N (ids 1..N are real, 0 is reserved)
        self._n_contracts = max(self.contracts) - 45 + 1
        self._n_layouts = len(self.layouts) + 1
        # the frozen document and the extension merged on top of it (see merge_semantic_extension)
        self.frozen_doc = self.doc
        self.extension = None

    def _validate_row(self, row, label):
        source, semantic = row.get("source_slots"), row.get("semantic_slots")
        if (
            not isinstance(source, list)
            or not isinstance(semantic, list)
            or not source
            or len(source) != len(semantic)
            or len(set(source)) != len(source)
            or len(set(semantic)) != len(semantic)
            or any(not isinstance(x, int) or not 0 <= x < 128 for x in source)
            or any(not isinstance(x, int) or not 0 <= x < self.semantic_dim for x in semantic)
        ):
            raise ValueError(f"{label} source and semantic slots must be unique and in bounds")

    @classmethod
    def from_doc(cls, doc):
        return cls(doc)

    @classmethod
    def load(cls, path=None):
        resolved = (DEFAULT_SEMANTIC_SPACE_PATH if path is None else Path(path)).resolve()
        extension = (
            os.environ.get(SEMANTIC_EXTENSIONS_ENV) if resolved == DEFAULT_SEMANTIC_SPACE_PATH.resolve() else None
        )
        return _load_cached(str(resolved), str(Path(extension).resolve()) if extension else None)

    def layout_id(self, name):
        try:
            return self.layouts[name]
        except KeyError as e:
            raise ValueError(f"unknown layout: {name}") from e

    def first_layout_id(self, contract_id):
        ids = sorted(lid for cid, lid in self._mappings if cid == contract_id)
        if not ids:
            raise ValueError(f"no layout mapping for contract {contract_id}")
        return ids[0]

    def mapping(self, contract_id, layout_id):
        try:
            return self._mappings[contract_id, layout_id]
        except KeyError as e:
            raise ValueError(f"unmapped contract/layout pair: {contract_id}/{layout_id}") from e

    def validate_sources(self, shared_doc=None, eef_doc=None):
        anatomical_hash = self.doc.get("anatomical_map_sha256")
        if anatomical_hash is not None:
            name = self.doc.get("anatomical_map_file", "anatomical_maps_v1.json")
            path = DEFAULT_SEMANTIC_SPACE_PATH.parent / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != anatomical_hash:
                raise ValueError("anatomical map hash mismatch; rebuild semantic space")
        raw = self.doc.get("source_registry_hashes", {})
        canonical = self.doc.get("source_registry_canonical_hashes", {})
        if shared_doc is None and eef_doc is None:
            root = Path(__file__).resolve().parent.parent / "registry"
            sources = {
                "family_projectors_v2.json": root / "family_projectors_v2.json",
                "state_eef.json": root / "state_eef.json",
            }
            for name, path in sources.items():
                if hashlib.sha256(path.read_bytes()).hexdigest() != raw.get(name):
                    raise ValueError(f"semantic source hash mismatch: {name}")
            return
        if shared_doc is None or eef_doc is None:
            raise ValueError("shared_doc and eef_doc must be supplied together")
        for name, value in (("family_projectors_v2.json", shared_doc), ("state_eef.json", eef_doc)):
            if hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest() != canonical.get(name):
                raise ValueError(f"semantic source hash mismatch: {name}")

    def _rows(self, contract_ids, layout_ids, count):
        import torch

        ids = torch.as_tensor(contract_ids)
        integer = (torch.int8, torch.int16, torch.int32, torch.int64)
        if ids.dtype not in integer or ids.shape != (count,):
            raise ValueError("contract_ids must be integer [N]")
        if set(ids.detach().cpu().tolist()) - set(self.contracts):
            raise ValueError("unknown contract IDs")
        if self.version == 1:
            return ids, None
        if layout_ids is None:
            raise ValueError("layout_ids are required by semantic schema v2")
        lids = torch.as_tensor(layout_ids, device=ids.device)
        if lids.dtype not in integer or lids.shape != (count,):
            raise ValueError("layout_ids must be integer [N]")
        pairs = zip(ids.detach().cpu().tolist(), lids.detach().cpu().tolist())
        missing = [f"{cid}/{lid}" for cid, lid in pairs if (cid, lid) not in self._mappings]
        if missing:
            raise ValueError(f"unmapped contract/layout pair: {missing}")
        return ids, lids

    def _row_mapping(self, cid, lid):
        return self.contracts[cid] if self.version == 1 else self._mappings[cid, lid]

    def pack(self, values, contract_ids, valid_mask, layout_ids=None):
        import torch

        values = torch.as_tensor(values)
        valid = torch.as_tensor(valid_mask, device=values.device)
        if values.ndim != 2 or values.shape[1] != 128 or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("values and valid_mask must be [N,128], with a bool mask")
        ids, lids = self._rows(contract_ids, layout_ids, values.shape[0])
        ids = ids.to(values.device)
        lids = None if lids is None else lids.to(values.device)
        out = torch.zeros((len(values), self.semantic_dim), dtype=values.dtype, device=values.device)
        mask = torch.zeros_like(out, dtype=torch.bool)
        for row, cid in enumerate(ids.tolist()):
            spec = self._row_mapping(cid, None if lids is None else int(lids[row]))
            source, semantic = spec["source_slots"], spec["semantic_slots"]
            allowed = torch.zeros(128, dtype=torch.bool, device=values.device)
            allowed[source] = True
            if (valid[row] & ~allowed).any():
                raise ValueError(f"{cid}: unmapped active transport coordinate")
            gathered = values[row, source]
            present = valid[row, source] & torch.isfinite(gathered)
            shared = torch.tensor(spec.get("shared_mask", [True] * len(source)), device=values.device)
            present = present & shared
            out[row, semantic] = torch.where(present, gathered, torch.zeros_like(gathered))
            mask[row, semantic] = present
        return out, mask

    def shared_transport_mask(self, contract_ids, layout_ids=None):
        """[N,128] bool: transport coordinates that participate in shared semantic supervision."""
        import torch

        ids = torch.as_tensor(contract_ids)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("contract_ids must be integer [N]")
        if self.version == 1:
            return torch.ones((ids.shape[0], 128), dtype=torch.bool, device=ids.device)
        if layout_ids is None:
            raise ValueError("layout_ids are required by semantic schema v2")
        lids = torch.as_tensor(layout_ids, device=ids.device)
        if lids.shape != ids.shape:
            raise ValueError("layout_ids must be integer [N]")
        key = str(ids.device)
        if key not in self._shared_tables:
            table = torch.zeros((self._n_contracts * self._n_layouts, 128), dtype=torch.bool, device=ids.device)
            for (cid, lid), spec in self._mappings.items():
                slots = [s for s, keep in zip(spec["source_slots"], spec["shared_mask"]) if keep]
                table[(cid - 45) * self._n_layouts + lid, slots] = True
            self._shared_tables[key] = table
        if ((ids < 45) | (ids >= 45 + self._n_contracts) | (lids < 0) | (lids >= self._n_layouts)).any():
            raise ValueError("unknown contract IDs")
        return self._shared_tables[key][(ids.long() - 45) * self._n_layouts + lids.long()]

    def unpack(self, values, contract_ids, layout_ids=None):
        import torch

        values = torch.as_tensor(values)
        if values.ndim != 2 or values.shape[1] != self.semantic_dim:
            raise ValueError(f"semantic values must be [N,{self.semantic_dim}]")
        ids, lids = self._rows(contract_ids, layout_ids, values.shape[0])
        out = torch.zeros((len(values), 128), dtype=values.dtype, device=values.device)
        for row, cid in enumerate(ids.tolist()):
            spec = self._row_mapping(cid, None if lids is None else int(lids[row]))
            got = values[row, spec["semantic_slots"]]
            out[row, spec["source_slots"]] = torch.where(torch.isfinite(got), got, torch.zeros_like(got))
        return out

    def body_part_ids(self, contract_ids, layout_ids=None):
        import torch

        # This is used by every training loss. Keep per-token IDs on device.
        ids = torch.as_tensor(contract_ids)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("contract_ids must be integer [N]")
        if ((ids < 45) | (ids >= 45 + self._n_contracts)).any():
            raise ValueError("unknown contract IDs")
        lids = None
        if self.version == 2:
            if layout_ids is None:
                raise ValueError("layout_ids are required by semantic schema v2")
            lids = torch.as_tensor(layout_ids, device=ids.device)
            if lids.shape != ids.shape or lids.dtype not in (torch.int32, torch.int64):
                raise ValueError("layout_ids must be integer [N]")
            if ((lids < 1) | (lids >= self._n_layouts)).any():
                raise ValueError("unmapped contract/layout pair")
        key = str(ids.device)
        if key not in self._part_tables:
            table = torch.full((self._n_contracts * self._n_layouts, 128), -1, dtype=torch.int64, device=ids.device)
            for contract_id in self.contract_ids:
                for layout_id in range(1, self._n_layouts):
                    spec = (
                        self.contracts[contract_id]
                        if self.version == 1
                        else self._mappings.get((contract_id, layout_id))
                    )
                    if spec is not None:
                        table[(contract_id - 45) * self._n_layouts + layout_id, spec["source_slots"]] = torch.tensor(
                            [self._parts[slot] for slot in spec["semantic_slots"]], device=ids.device
                        )
            self._part_tables[key] = table
        index = (ids.long() - 45) * self._n_layouts + (1 if self.version == 1 else lids.long())
        result = self._part_tables[key][index]
        if (result < 0).all(dim=1).any():
            raise ValueError("unmapped contract/layout pair")
        return result


def restrict_action_valid_mask_to_shared(valid_mask, shared_mask, width):
    """AND a per-sample [D] or [T,D] action validity mask with the [1,128] shared-supervision mask."""
    import torch

    shared = shared_mask.reshape(-1)[:width]
    if valid_mask is None:
        return shared.clone()
    valid = valid_mask.to(device=shared.device)
    if valid.shape[-1] < width:
        raise ValueError(f"action_valid_mask width {valid.shape[-1]} < action width {width}")
    keep = torch.ones_like(valid, dtype=torch.bool)
    keep[..., :width] = shared
    return valid & keep


def _canonical_sha256(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def merge_semantic_extension(doc, extension):
    """The frozen semantic space plus an extension that adds contracts, layouts and their mappings.

    The extension names the digest of the space it extends and may only add: a contract id, layout name, layout id or
    context id the frozen space already has is rejected here, and every mapping must pair an added contract with an
    added layout. The caller builds a SemanticSpace from the merged document, which runs every check again on the
    whole (paired EEF banks, representation by bank, context = layout, unique pairs, slot bounds).
    """
    if (extension.get("schema_name"), extension.get("schema_version")) != ("greenwam_semantic_extensions", 1):
        raise ValueError("expected greenwam_semantic_extensions v1")
    if extension.get("parent_semantic_space_sha256") != _canonical_sha256(doc):
        raise ValueError("semantic extension was written for a different semantic space")
    contracts, layouts, mappings = (extension.get(key) for key in ("contracts", "layouts", "mappings"))
    if not contracts or not layouts or not mappings:
        raise ValueError("semantic extension must add contracts, layouts and mappings")
    added_contracts = {int(key) for key in contracts}
    if added_contracts & {int(key) for key in doc["contracts"]}:
        raise ValueError(
            f"semantic extension redefines contracts {sorted(added_contracts & set(map(int, doc['contracts'])))}"
        )
    if set(layouts) & set(doc["layouts"]) or set(layouts.values()) & set(doc["layouts"].values()):
        raise ValueError("semantic extension redefines a layout name or id")
    frozen_contexts = {row.get("context_id") for row in doc["mappings"]}
    for row in mappings:
        if row.get("contract_id") not in added_contracts or row.get("layout_id") not in set(layouts.values()):
            raise ValueError("semantic extension mappings must pair an added contract with an added layout")
        if row.get("representation_id") not in (1, 2):
            raise ValueError("semantic extension mappings must be EEF (representation 1 or 2)")
        if row.get("context_id") in frozen_contexts:
            raise ValueError(f"semantic extension reuses context {row.get('context_id')}")
    return dict(
        doc,
        contracts={**doc["contracts"], **{str(key): value for key, value in contracts.items()}},
        layouts={**doc["layouts"], **layouts},
        mappings=[*doc["mappings"], *mappings],
    )


@lru_cache(maxsize=None)
def _load_cached(resolved_path, extension_path=None):
    space = SemanticSpace(json.loads(Path(resolved_path).read_text()))
    if Path(resolved_path) == DEFAULT_SEMANTIC_SPACE_PATH.resolve():
        space.validate_sources()
    if extension_path is None:
        return space
    extension = json.loads(Path(extension_path).read_text())
    extended = SemanticSpace(merge_semantic_extension(space.doc, extension))
    extended.frozen_doc, extended.extension = space.doc, extension
    return extended
