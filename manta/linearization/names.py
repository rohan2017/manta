"""Name resolution + freeze helpers shared by the transforms."""

from __future__ import annotations

import numpy as np


def resolve_suffix(key: str, candidates, *, label: str, who: str) -> str:
    """Resolve one user-supplied name against `candidates`.

    Accepts an exact match, else a unique `.<suffix>` match (craft-relative
    shorthand like ``"t.throttle"`` for ``"drone.t.throttle"``). Raises
    `KeyError` on an unknown or ambiguous key.
    """
    cands = list(candidates)
    if key in cands:
        return key
    matches = [n for n in cands if n.endswith("." + key)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"{who}: ambiguous {label} name {key!r}; matches {matches}. "
            f"Use the fully-qualified form.")
    raise KeyError(
        f"{who}: unknown {label} name {key!r}. Available: {sorted(cands)}")


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
