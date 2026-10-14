"""One shared learned action boundary over a static semantic action space."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


def _space_doc(space: Any) -> Mapping[str, Any]:
    doc = getattr(space, "doc", space)
    if not isinstance(doc, Mapping):
        raise TypeError("semantic_space must expose a mapping .doc")
    return doc


class _SemanticProjector(nn.Module):
    """Static contract maps shared by the encoder and decoder."""

    def __init__(self, semantic_space: Any) -> None:
        super().__init__()
        doc = _space_doc(semantic_space)
        self.semantic_dim = int(getattr(semantic_space, "semantic_dim", doc["semantic_dim"]))
        self.transport_dim = int(getattr(semantic_space, "transport_dim", doc["transport_dim"]))
        contracts = doc["contracts"]
        if not contracts:
            raise ValueError("semantic space needs at least one contract")
        self.requires_layout_ids = doc.get("schema_version") == 2
        if self.requires_layout_ids:
            specs = sorted(doc["mappings"], key=lambda spec: (spec["contract_id"], spec["layout_id"]))
            pairs = [(int(spec["contract_id"]), int(spec["layout_id"])) for spec in specs]
            if not pairs or len(pairs) != len(set(pairs)):
                raise ValueError("duplicate or empty contract/layout mappings")
            parsed = [(int(spec["contract_id"]), spec) for spec in specs]
            self._layout_values_init = tuple(int(spec["layout_id"]) for spec in specs)
        else:
            parsed = sorted((int(contract_id), spec) for contract_id, spec in contracts.items())
            self._layout_values_init = tuple(0 for _ in parsed)
        self._representation_init = tuple(int(spec.get("representation_id", 0)) for _, spec in parsed)
        if any(rid not in (0, 1, 2) for rid in self._representation_init):
            raise ValueError("unknown representation_id")
        self._contract_values_init = tuple(contract_id for contract_id, _ in parsed)
        semantic_to_transport_init = [[-1] * self.semantic_dim for _ in parsed]
        semantic_shared_init = [[False] * self.semantic_dim for _ in parsed]
        contract_context_init = [0] * len(parsed)
        for row, (_, spec) in enumerate(parsed):
            source_slots = tuple(int(slot) for slot in spec["source_slots"])
            semantic_slots = tuple(int(slot) for slot in spec["semantic_slots"])
            shared_flags = tuple(bool(x) for x in spec.get("shared_mask", [True] * len(source_slots)))
            if len(shared_flags) != len(source_slots):
                raise ValueError("shared_mask must align with source_slots")
            if len(source_slots) != len(semantic_slots):
                raise ValueError("source_slots and semantic_slots must have equal length")
            if len(set(source_slots)) != len(source_slots) or len(set(semantic_slots)) != len(semantic_slots):
                raise ValueError("contract slots must be unique")
            if any(slot < 0 or slot >= self.transport_dim for slot in source_slots):
                raise ValueError("source slot outside transport dimension")
            if any(slot < 0 or slot >= self.semantic_dim for slot in semantic_slots):
                raise ValueError("semantic slot outside semantic dimension")
            for semantic_slot, source_slot, keep in zip(semantic_slots, source_slots, shared_flags, strict=True):
                semantic_to_transport_init[row][semantic_slot] = source_slot
                semantic_shared_init[row][semantic_slot] = keep
            contract_context_init[row] = int(spec["context_id"])
        self._semantic_to_transport_init = tuple(tuple(row) for row in semantic_to_transport_init)
        self._semantic_shared_init = tuple(tuple(row) for row in semantic_shared_init)
        self._contract_context_init = tuple(contract_context_init)
        self._allowed_transport_init = tuple(
            tuple(slot in mapping for slot in range(self.transport_dim)) for mapping in self._semantic_to_transport_init
        )
        self.register_buffer(
            "contract_values", torch.tensor(self._contract_values_init, dtype=torch.long), persistent=True
        )
        self.register_buffer("layout_values", torch.tensor(self._layout_values_init, dtype=torch.long), persistent=True)
        self.register_buffer(
            "semantic_to_transport", torch.tensor(self._semantic_to_transport_init, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "contract_context", torch.tensor(self._contract_context_init, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "allowed_transport", torch.tensor(self._allowed_transport_init, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "mapping_representation", torch.tensor(self._representation_init, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "semantic_shared", torch.tensor(self._semantic_shared_init, dtype=torch.bool), persistent=True
        )
        self.first_contract_id = self._contract_values_init[0]
        self.first_layout_id = self._layout_values_init[0]

    def _reset_mapping_buffers(self, device: torch.device) -> None:
        self.mapping_representation = torch.tensor(self._representation_init, dtype=torch.long, device=device)
        self.contract_values = torch.tensor(self._contract_values_init, dtype=torch.long, device=device)
        self.layout_values = torch.tensor(self._layout_values_init, dtype=torch.long, device=device)
        self.semantic_to_transport = torch.tensor(self._semantic_to_transport_init, dtype=torch.long, device=device)
        self.contract_context = torch.tensor(self._contract_context_init, dtype=torch.long, device=device)
        self.allowed_transport = torch.tensor(self._allowed_transport_init, dtype=torch.bool, device=device)
        self.semantic_shared = torch.tensor(self._semantic_shared_init, dtype=torch.bool, device=device)

    def _rows(self, contract_ids: torch.Tensor, count: int, layout_ids: torch.Tensor | None = None) -> torch.Tensor:
        if contract_ids.dtype != torch.long:
            raise TypeError("contract_ids must be torch.long")
        if contract_ids.ndim != 1 or contract_ids.numel() != count:
            raise ValueError(f"contract_ids must be [N] for {count} action rows")
        matches = contract_ids[:, None].eq(self.contract_values[None, :])
        if self.requires_layout_ids:
            if layout_ids is None:
                raise ValueError("explicit layout_ids required for semantic actions")
            if layout_ids.dtype != torch.long or layout_ids.shape != contract_ids.shape:
                raise ValueError("layout_ids must be torch.long with the contract ID shape")
            matches = matches & layout_ids[:, None].eq(self.layout_values[None, :])
        if not bool(matches.any(dim=1).all()):
            raise ValueError("unknown action contract/layout id pair")
        return matches.to(torch.int64).argmax(dim=1)

    def _flat_inputs(
        self, x: torch.Tensor, contract_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        if x.ndim not in (2, 3) or x.shape[-1] != self.transport_dim:
            raise ValueError(f"action input must be [B,{self.transport_dim}] or [B,T,{self.transport_dim}]")
        if contract_ids.dtype != torch.long or contract_ids.ndim != 1 or contract_ids.shape[0] != x.shape[0]:
            raise ValueError("contract_ids must be torch.long [B]")
        shape = tuple(x.shape[:-1])
        if x.ndim == 3:
            contract_ids = contract_ids[:, None].expand(-1, x.shape[1]).reshape(-1)
        return x.reshape(-1, self.transport_dim), contract_ids, shape

    def _flat_layouts(self, layout_ids: torch.Tensor | None, shape: tuple[int, ...]) -> torch.Tensor | None:
        if layout_ids is None:
            return None
        if layout_ids.dtype != torch.long or layout_ids.shape != (shape[0],):
            raise ValueError("layout_ids must be torch.long [B]")
        return layout_ids if len(shape) == 1 else layout_ids[:, None].expand(*shape).reshape(-1)

    def _semantic_from_transport(
        self, x: torch.Tensor, rows: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != x.shape:
                raise ValueError("valid_mask must be bool with the action transport shape")
            allowed = self.allowed_transport.index_select(0, rows)
            if bool((valid_mask & ~allowed).any()):
                raise ValueError("unmapped active transport coordinate")
        mapping = self.semantic_to_transport.index_select(0, rows)
        active = mapping.ge(0) & self.semantic_shared.index_select(0, rows)
        source = mapping.clamp_min(0)
        selected = x.gather(1, source)
        if valid_mask is None:
            valid = active & torch.isfinite(selected)
        else:
            valid = active & valid_mask.gather(1, source) & torch.isfinite(selected)
        return torch.where(valid, selected, torch.zeros_like(selected)), valid

    def _transport_from_semantic(self, semantic: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        mapping = self.semantic_to_transport.index_select(0, rows)
        active = mapping.ge(0) & self.semantic_shared.index_select(0, rows)
        result = semantic.new_zeros((semantic.shape[0], self.transport_dim))
        row = torch.arange(semantic.shape[0], device=semantic.device)[:, None].expand_as(mapping)
        result[row[active], mapping[active]] = semantic[active]
        return result


class SharedSemanticActionInput(_SemanticProjector):
    """Gather each action contract into semantics, then use one learned boundary."""

    def __init__(self, semantic_space: Any, hidden_size: int) -> None:
        super().__init__(semantic_space)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.input_size = self.transport_dim
        self.output_size = int(hidden_size)
        self.value = nn.Linear(self.semantic_dim, hidden_size)
        self.mask = nn.Linear(self.semantic_dim, hidden_size, bias=False)
        self.presence = nn.Linear(self.semantic_dim, hidden_size, bias=False)
        context_count = max(self._contract_context_init) + 1
        self.context = nn.Embedding(context_count, hidden_size)
        self.representation = nn.Embedding(3, hidden_size)
        self.initialize_action_parameters(1.0 / math.sqrt(self.semantic_dim))

    def initialize_action_parameters(self, std: float) -> None:
        self._reset_mapping_buffers(self.value.weight.device)
        nn.init.trunc_normal_(self.value.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.value.bias)
        nn.init.normal_(self.mask.weight, std=std * 0.1)
        nn.init.normal_(self.presence.weight, std=std * 0.1)
        nn.init.normal_(self.context.weight, std=std * 0.1)
        nn.init.normal_(self.representation.weight, std=std * 0.1)

    def encode_semantic(
        self,
        x: torch.Tensor,
        contract_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        layout_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
        x, ids, shape = self._flat_inputs(x, contract_ids)
        if valid_mask is not None:
            if tuple(valid_mask.shape) != shape + (self.transport_dim,):
                raise ValueError("valid_mask must be bool with the action transport shape")
            valid_mask = valid_mask.reshape_as(x)
        layouts = self._flat_layouts(layout_ids, shape)
        rows = self._rows(ids, x.shape[0], layouts)
        values, semantic_mask = self._semantic_from_transport(x, rows, valid_mask)
        return values, semantic_mask, rows, shape

    def forward(
        self,
        x: torch.Tensor,
        contract_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        layout_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values, semantic_mask, rows, shape = self.encode_semantic(x, contract_ids, valid_mask, layout_ids=layout_ids)
        output = self.value(values) + self.mask(semantic_mask.to(dtype=values.dtype))
        present = self.semantic_to_transport.index_select(0, rows).ge(0) & self.semantic_shared.index_select(0, rows)
        output = output + self.presence(present.to(dtype=values.dtype))
        output = output + self.context(self.contract_context.index_select(0, rows))
        output = output + self.representation(self.mapping_representation.index_select(0, rows))
        return output.reshape(*shape, self.output_size)


class SharedSemanticActionOutput(_SemanticProjector):
    """Use one learned semantic decoder and scatter back to padded transport slots."""

    def __init__(self, semantic_space: Any, hidden_size: int) -> None:
        super().__init__(semantic_space)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.input_size = int(hidden_size)
        self.output_size = self.transport_dim
        self.value = nn.Linear(hidden_size, self.semantic_dim)
        self.initialize_action_parameters(1.0 / math.sqrt(hidden_size))

    def initialize_action_parameters(self, std: float) -> None:
        self._reset_mapping_buffers(self.value.weight.device)
        nn.init.trunc_normal_(self.value.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.value.bias)

    def project_semantic(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x)

    def forward(
        self,
        x: torch.Tensor,
        contract_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        layout_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim not in (2, 3) or x.shape[-1] != self.input_size:
            raise ValueError(f"hidden input must be [B,{self.input_size}] or [B,T,{self.input_size}]")
        if contract_ids.dtype != torch.long or contract_ids.ndim != 1 or contract_ids.shape[0] != x.shape[0]:
            raise ValueError("contract_ids must be torch.long [B]")
        shape = tuple(x.shape[:-1])
        flat = x.reshape(-1, self.input_size)
        ids = contract_ids if x.ndim == 2 else contract_ids[:, None].expand(-1, x.shape[1]).reshape(-1)
        rows = self._rows(ids, flat.shape[0], self._flat_layouts(layout_ids, shape))
        semantic = self.project_semantic(flat)
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != shape + (self.semantic_dim,):
                raise ValueError("output valid_mask must be bool with semantic action shape")
            semantic = torch.where(valid_mask.reshape_as(semantic), semantic, torch.zeros_like(semantic))
        return self._transport_from_semantic(semantic, rows).reshape(*shape, self.transport_dim)
