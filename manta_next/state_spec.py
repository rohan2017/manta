"""StateSpec — flat-vector serialization of a Craft's state.

A Craft's tick function takes a state dict keyed by slot name
(`position`, `orientation`, `<part>.<state>`, …) and returns a new state
dict. For filtering / control / autodiff at the system level it's
convenient to view the state as a single flat vector. StateSpec is the
layout descriptor for that view.

Two parallel layouts:

  * **Ambient** — concatenated values exactly as they live in the tick
    dict. Rigid-body slots contribute (3, 4, 3, 3) = 13 floats per craft
    (position, orientation quaternion, velocity, angular velocity).
    Each `State(manifold='R1')` slot on a part contributes 1.

  * **Tangent** — the natural error/perturbation space. Same as ambient
    *except* `orientation` collapses 4→3 (the SO(3) tangent is axis-angle,
    not quaternion). M4 ships the ambient view only; tangent layout
    lives here as a hook for the manifold-aware ESKF in M5.

API::

    spec = StateSpec.from_craft(craft)
    spec.ambient_dim                    # total ambient floats
    spec.slots                          # list of (name, dim, manifold)
    flat = spec.pack(state_dict)        # → np.ndarray, length ambient_dim
    state_dict = spec.unpack(flat)      # round-trip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .parts.base import Part, State


@dataclass(frozen=True)
class StateSlot:
    """One named slot in the flat layout.

    Attrs:
        name      — string key in the tick dict (e.g., "position",
                    "wheel.angle"). Same key the tick function uses.
        offset    — start index in the ambient vector.
        dim       — width in the ambient vector (1, 3, or 4).
        manifold  — 'R1' (scalar), 'R3' (3-vector), 'SO3' (quaternion 4→3 tan).
                    Determines the tangent layout for an ESKF.
        tangent_dim — width in the tangent vector. 4 → 3 for SO3, else
                      equal to `dim`.
    """
    name: str
    offset: int
    dim: int
    manifold: str
    tangent_dim: int


class StateSpec:
    """Layout descriptor for a flat ambient state vector."""

    def __init__(self, slots: list[StateSlot]) -> None:
        self._slots = list(slots)
        self._slot_by_name = {s.name: s for s in self._slots}
        self._ambient_dim = sum(s.dim for s in self._slots)
        self._tangent_dim = sum(s.tangent_dim for s in self._slots)

    # ----- Construction --------------------------------------------------

    @classmethod
    def from_craft(cls, craft) -> "StateSpec":
        """Build the canonical state spec for a single-craft world.

        Layout:
          position           (3, R3)
          orientation        (4, SO3 → tangent 3)
          velocity           (3, R3)
          angular_velocity   (3, R3)
          <part>.<slot>      (1, R1)  in declaration order

        Future: multi-craft worlds will concatenate per-craft layouts here.
        """
        slots: list[StateSlot] = []
        offset = 0

        def add(name: str, dim: int, manifold: str, tangent_dim: int):
            nonlocal offset
            slots.append(StateSlot(name=name, offset=offset, dim=dim,
                                   manifold=manifold, tangent_dim=tangent_dim))
            offset += dim

        add("position",         3, "R3",  3)
        add("orientation",      4, "SO3", 3)
        add("velocity",         3, "R3",  3)
        add("angular_velocity", 3, "R3",  3)

        for part in craft.parts:
            for sname, sdecl in part.state_declarations().items():
                key = f"{part.name}.{sname}"
                if sdecl.manifold == "R1":
                    add(key, 1, "R1", 1)
                else:
                    raise NotImplementedError(
                        f"StateSpec: manifold {sdecl.manifold!r} on "
                        f"{part.name}.{sname} not yet supported.")

        return cls(slots)

    # ----- Accessors -----------------------------------------------------

    @property
    def slots(self) -> tuple[StateSlot, ...]:
        return tuple(self._slots)

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    @property
    def tangent_dim(self) -> int:
        return self._tangent_dim

    def slot(self, name: str) -> StateSlot:
        return self._slot_by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._slot_by_name

    def __repr__(self) -> str:
        return (f"<StateSpec ambient={self._ambient_dim} "
                f"tangent={self._tangent_dim} "
                f"slots={[s.name for s in self._slots]}>")

    # ----- Pack / Unpack -------------------------------------------------

    def pack(self, state_dict: dict[str, Any]) -> np.ndarray:
        """Flatten a tick-style dict into the ambient vector."""
        flat = np.zeros(self._ambient_dim, dtype=float)
        for slot in self._slots:
            value = state_dict[slot.name]
            arr = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
            if arr.size != slot.dim:
                raise ValueError(
                    f"StateSpec.pack: slot {slot.name!r} expects dim {slot.dim}, "
                    f"got len={arr.size}")
            flat[slot.offset : slot.offset + slot.dim] = arr
        return flat

    def unpack(self, flat: np.ndarray) -> dict[str, Any]:
        """Slice an ambient vector back into a tick-style dict."""
        flat = np.asarray(flat, dtype=float)
        if flat.shape != (self._ambient_dim,):
            raise ValueError(
                f"StateSpec.unpack: expected shape ({self._ambient_dim},), "
                f"got {flat.shape}")
        out: dict[str, Any] = {}
        for slot in self._slots:
            chunk = flat[slot.offset : slot.offset + slot.dim]
            if slot.dim == 1:
                out[slot.name] = float(chunk[0])
            else:
                out[slot.name] = chunk.copy()
        return out
