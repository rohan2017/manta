"""Shared machinery for the model-linearizing transforms (EKF, LQR, …).

`EKF(world)` and `LQR(world)` both do the same prologue: walk the tick
signature, flatten the world's nested initial state, resolve user-supplied
names (full or unambiguous suffix), optionally subset the `StateSpec` over a
dependency closure, and freeze the untracked slots at their reference value
before handing the result to `Linearization`. These free functions are that
shared prologue, so the two sibling transforms can't drift (and a third —
iLQR/MPC — reuses them for free).

The dependency-closure itself lives on `Linearization.dependency_closure`
(it reads `F`'s structural sparsity), so both transforms close `track` over
the dynamics identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_suffix(key: str, candidates, *, label: str, who: str) -> str:
    """Resolve one user-supplied name against `candidates`.

    Accepts an exact match, else a unique `.<suffix>` match (craft-relative
    shorthand like ``"t.throttle"`` for ``"drone.t.throttle"``). Raises
    `KeyError` on an unknown or ambiguous key. `who` names the caller for
    the error (e.g. ``"EKF.predict"``); `label` names the kind (``"input"``,
    ``"sensor"``, ``"slot"``).
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


def flatten_nested(nested: dict) -> dict[str, Any]:
    """`{owner: {slot: v}}` → `{"owner.slot": v}`; passes flat dicts through."""
    flat: dict[str, Any] = {}
    for owner, slots in nested.items():
        if isinstance(slots, dict):
            for slot, v in slots.items():
                flat[f"{owner}.{slot}"] = v
        else:
            flat[owner] = slots
    return flat


def freeze_complement(full_spec, kept, init_flat: dict,
                      into: dict | None = None) -> dict:
    """Freeze every full-spec slot NOT in `kept` at its `init_flat` value
    (zeros if absent), as a flat numpy column. Returns the frozen dict
    (updating `into` in place when given)."""
    frozen = into if into is not None else {}
    for s in full_spec.slots:
        if s.name not in kept:
            val = init_flat.get(s.name, np.zeros(s.ambient_dim))
            frozen[s.name] = np.atleast_1d(
                np.asarray(val, dtype=float)).reshape(-1)
    return frozen
