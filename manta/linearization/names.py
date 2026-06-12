"""Freeze helpers shared by the transforms.

(Suffix name resolution lives in `manta.ir._names.resolve_suffix` — a low
module that `bus`/`state_spec` and the transforms all reach.)
"""

from __future__ import annotations

import numpy as np


def freeze_complement(full_spec, kept, init_flat: dict,
                      into: dict | None = None) -> dict:
    """Freeze every full-spec slot NOT in `kept` at its `init_flat` value
    (zeros if absent), as a flat numpy column."""
    frozen = into if into is not None else {}
    for s in full_spec.slots:
        if s.name not in kept:
            val = init_flat.get(s.name, np.zeros(s.ambient_dim))
            frozen[s.name] = np.atleast_1d(
                np.asarray(val, dtype=float)).reshape(-1)
    return frozen


def slot_of_tangent_index(spec) -> list[str]:
    """Map each tangent index to its slot name (slot granularity keeps an
    SO(3) orientation atomic for closure/partitioning)."""
    m: list[str] = [""] * spec.tangent_dim
    for s in spec.slots:
        for i in range(s.tangent_offset, s.tangent_offset + s.tangent_dim):
            m[i] = s.name
    return m
