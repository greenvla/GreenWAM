# SPDX-License-Identifier: OpenMDW-1.1
"""Semantic extensions: the frozen space is unchanged without one, and an extension may only add."""

import copy
import json
from pathlib import Path

import pytest

from greenwam.data import semantic_space
from greenwam.data.semantic_space import (
    DEFAULT_SEMANTIC_SPACE_PATH,
    SEMANTIC_EXTENSIONS_ENV,
    SemanticSpace,
    _canonical_sha256,
    merge_semantic_extension,
)

FRANKA_EEF = Path(__file__).resolve().parents[2] / "benchmarks" / "semantic_extensions" / "franka_eef_v1.json"


@pytest.fixture
def frozen():
    return json.loads(DEFAULT_SEMANTIC_SPACE_PATH.read_text())


@pytest.fixture
def extension():
    return json.loads(FRANKA_EEF.read_text())


@pytest.fixture
def fresh_cache():
    semantic_space._load_cached.cache_clear()
    yield
    semantic_space._load_cached.cache_clear()


def test_default_space_is_the_frozen_file(monkeypatch, frozen, fresh_cache):
    monkeypatch.delenv(SEMANTIC_EXTENSIONS_ENV, raising=False)
    space = SemanticSpace.load()
    assert space.extension is None
    assert space.frozen_doc is space.doc
    assert _canonical_sha256(space.doc) == _canonical_sha256(frozen)
    assert max(space.layouts.values()) == 43


def test_extension_adds_the_franka_eef_layout(monkeypatch, frozen, fresh_cache):
    monkeypatch.setenv(SEMANTIC_EXTENSIONS_ENV, str(FRANKA_EEF))
    space = SemanticSpace.load()
    assert _canonical_sha256(space.frozen_doc) == _canonical_sha256(frozen)
    assert space.extension is not None
    layout = space.layout_id("franka_eef_v1")
    assert layout == 44
    for contract, representation in ((76, 1), (90, 2)):
        row = space.mapping(contract, layout)
        assert row["representation_id"] == representation and row["context_id"] == 44
        assert row["source_slots"] == list(range(10))
        assert row["semantic_slots"] == [44, 45, 46, 47, 48, 49, 50, 51, 52, 78]
    for cid, lid in ((66, 4), (80, 4)):  # frozen rows are untouched
        assert space.mapping(cid, lid) == SemanticSpace(frozen).mapping(cid, lid)


def test_extension_gives_the_projector_its_context_row(monkeypatch, fresh_cache):
    from cosmos_framework.model.generator.mot.semantic_action_projector import SharedSemanticActionInput

    monkeypatch.delenv(SEMANTIC_EXTENSIONS_ENV, raising=False)
    assert SharedSemanticActionInput(SemanticSpace.load(), 8).context.num_embeddings == 44
    semantic_space._load_cached.cache_clear()
    monkeypatch.setenv(SEMANTIC_EXTENSIONS_ENV, str(FRANKA_EEF))
    projector = SharedSemanticActionInput(SemanticSpace.load(), 8)
    assert projector.context.num_embeddings == 45
    rows = [
        i
        for i, (c, l) in enumerate(zip(projector.contract_values.tolist(), projector.layout_values.tolist()))
        if l == 44
    ]
    assert sorted(projector.contract_values[rows].tolist()) == [76, 90]
    assert projector.contract_context[rows].tolist() == [44, 44]
    assert sorted(projector.mapping_representation[rows].tolist()) == [1, 2]


def test_extension_attaches_only_to_the_default_space(monkeypatch, tmp_path, frozen, fresh_cache):
    monkeypatch.setenv(SEMANTIC_EXTENSIONS_ENV, str(FRANKA_EEF))
    copy_path = tmp_path / "space.json"
    copy_path.write_text(json.dumps(frozen))
    assert SemanticSpace.load(copy_path).extension is None


def _merged(frozen, extension):
    return SemanticSpace(merge_semantic_extension(frozen, extension))


def test_merged_franka_eef_passes_every_space_check(frozen, extension):
    space = _merged(frozen, extension)
    assert {76, 90} <= set(space.contracts)


@pytest.mark.parametrize(
    "edit, message",
    [
        (
            lambda ext: ext["contracts"].update({"66": {"family_id": "x", "representation": "eef_relative"}}),
            "redefines contracts",
        ),
        (lambda ext: ext.update(layouts={"franka_eef_v1": 4}), "redefines a layout"),
        (lambda ext: ext.update(layouts={"dual_arm_b__dual_arm_b__33__34": 44}), "redefines a layout"),
        (lambda ext: [row.update(context_id=4) for row in ext["mappings"]], "reuses context"),
        (lambda ext: [row.update(representation_id=0) for row in ext["mappings"]], "representation 1 or 2"),
        (lambda ext: ext["mappings"][0].update(contract_id=66), "pair an added contract"),
        (lambda ext: ext.update(parent_semantic_space_sha256="0" * 64), "different semantic space"),
    ],
)
def test_extension_collisions_are_rejected_before_the_merge(frozen, extension, edit, message):
    bad = copy.deepcopy(extension)
    edit(bad)
    with pytest.raises(ValueError, match=message):
        merge_semantic_extension(frozen, bad)


@pytest.mark.parametrize(
    "edit",
    [
        lambda ext: [row.update(context_id=45) for row in ext["mappings"]],  # context must equal the layout
        lambda ext: ext["mappings"][1].update(representation_id=1),  # absolute bank needs representation 2
        lambda ext: ext["contracts"].pop("90") and ext["mappings"].pop(),  # relative EEF without its absolute twin
        lambda ext: ext["mappings"][0].update(semantic_slots=[44] * 10),  # duplicate semantic slots
        lambda ext: ext.update(layouts={"franka_eef_v1": 45}),  # layout ids must stay contiguous
    ],
)
def test_merged_document_runs_the_space_checks_again(frozen, extension, edit):
    bad = copy.deepcopy(extension)
    edit(bad)
    with pytest.raises(ValueError):
        _merged(frozen, bad)
