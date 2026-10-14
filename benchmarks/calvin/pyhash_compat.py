# SPDX-License-Identifier: OpenMDW-1.1
"""Pure-Python ``pyhash.fnv1_32`` for the official CALVIN evaluation.

``calvin_agent.evaluation.utils`` imports ``pyhash`` at module load and seeds the random part of every initial
scene with ``pyhash.fnv1_32()(str(initial_condition.values()))``. The C extension does not build on current Python
versions, so this module registers a stand-in that computes the same 32-bit FNV-1 hash (multiply by the FNV prime,
then XOR the next byte of the UTF-8 encoding). It is always used, so every run seeds the scenes the same way.
"""

from __future__ import annotations

import sys
import types

_FNV_PRIME = 0x01000193
_FNV_OFFSET_BASIS = 0x811C9DC5


class fnv1_32:  # noqa: N801  (mirrors the pyhash class name)
    def __call__(self, data: str | bytes) -> int:
        if isinstance(data, str):
            data = data.encode("utf-8")
        value = _FNV_OFFSET_BASIS
        for byte in data:
            value = (value * _FNV_PRIME) % (1 << 32)
            value ^= byte
        return value


def install() -> None:
    """Register the stand-in as ``pyhash`` before ``calvin_agent`` is imported."""
    module = types.ModuleType("pyhash")
    module.fnv1_32 = fnv1_32
    sys.modules["pyhash"] = module
