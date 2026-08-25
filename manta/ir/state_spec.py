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
    not quaternion). The ESKF lives here: covariance is tangent-dim, F
    and H are tangent-shaped, and the state update is applied via
    `boxplus(x_ambient, K·y_tangent)`.

API::

    spec = manta.state_spec_from_craft(craft)
    spec.ambient_dim                    # total ambient floats
    spec.tangent_dim                    # total tangent floats
    spec.slots                          # list of StateSlot
    flat = spec.pack(state_dict)        # → np.ndarray, length ambient_dim
    state_dict = spec.unpack(flat)      # round-trip

    # ESKF-style perturbation (symbolic, casadi MX):
    x_pert = spec.boxplus_sym(x_ambient_mx, delta_tangent_mx)
    delta  = spec.boxminus_sym(x_ambient_a_mx, x_ambient_b_mx)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Any

import casadi as ca
import numpy as np

from ._names import resolve_suffix
from .manifold import Manifold


class SlotSet(IntFlag):
    """Aliases for the rigid-body slots an EKF should track.

    Combine with ``|``::

        EKF(world, track={"drone": POSE | TWIST})

    The composites map to the rigid-body slot-name suffixes
    (``position`` / ``orientation`` / ``velocity`` / ``angular_velocity``).
    They name ONLY the rigid-body block — part R1/R3 states, RW-bias
    slots, and disturbance slots are never selected by an alias; they
    enter the tracked set via dynamics closure or because a chosen
    sensor observes them.
    """
    POSITION         = auto()
    ORIENTATION      = auto()
    VELOCITY         = auto()
    ANGULAR_VELOCITY = auto()
    POSE  = POSITION | ORIENTATION
    TWIST = VELOCITY | ANGULAR_VELOCITY
    ALL   = POSE | TWIST


# Convenience module-level aliases (the common spelling at call sites).
POSE  = SlotSet.POSE
TWIST = SlotSet.TWIST
ALL   = SlotSet.ALL

_SLOTSET_SUFFIX = {
    SlotSet.POSITION:         "position",
    SlotSet.ORIENTATION:      "orientation",
    SlotSet.VELOCITY:         "velocity",
    SlotSet.ANGULAR_VELOCITY: "angular_velocity",
}


def resolve_slotset(craft_name: str, slotset: SlotSet) -> set[str]:
    """Expand a `SlotSet` into the full rigid-body slot names for a craft.

    `SlotSet.POSE` on craft ``"drone"`` →
    ``{"drone.position", "drone.orientation"}``.
    """
    return {f"{craft_name}.{suffix}"
            for flag, suffix in _SLOTSET_SUFFIX.items()
            if slotset & flag}


def flatten_nested(nested: dict) -> dict[str, Any]:
    """`{owner: {slot: v}}` → `{"owner.slot": v}`; passes flat entries
    (and dotless keys) through unchanged. The one nested→flat rule."""
    if not isinstance(nested, dict):
        raise TypeError(
            f"state must be a dict, got {type(nested).__name__}")
    flat: dict[str, Any] = {}
    for owner, slots in nested.items():
        if isinstance(slots, dict):
            for slot, v in slots.items():
                full = f"{owner}.{slot}"
                if full in flat:
                    raise ValueError(f"duplicate state key {full!r}")
                flat[full] = v
        else:
            if owner in flat:
                raise ValueError(f"duplicate state key {owner!r}")
            flat[owner] = slots
    return flat


@dataclass(frozen=True)
class StateSlot:
    """One named slot in the flat layout.

    Attrs:
        name           — string key in the tick dict (e.g., "position",
                         "wheel.angle"). Same key the tick function uses.
        ambient_offset — start index in the ambient vector.
        manifold       — `Manifold` instance describing the slot's storage
                         and tangent dims. `ambient_dim` and `tangent_dim`
                         are derived. Backends key on `manifold.kind`.
        tangent_offset — start index in the tangent vector.

    The ambient pair (`ambient_offset`/`ambient_dim`) and the tangent pair
    (`tangent_offset`/`tangent_dim`) are named symmetrically on purpose:
    covariance / Jacobian work indexes by the *tangent* pair, packing /
    unpacking by the *ambient* pair, and mixing them is a classic bug.
    """
    name: str
    ambient_offset: int
    manifold: Manifold
    tangent_offset: int

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name
                or any(not part.isidentifier()
                       for part in self.name.split("."))):
            raise ValueError(f"StateSlot: invalid qualified name {self.name!r}")
        for attr in ("ambient_offset", "tangent_offset"):
            value = getattr(self, attr)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ValueError(
                    f"StateSlot {self.name!r}: {attr} must be a non-negative int")
        if not isinstance(self.manifold, Manifold):
            raise TypeError(
                f"StateSlot {self.name!r}: manifold must be a Manifold")

    @property
    def ambient_dim(self) -> int:
        return self.manifold.ambient_dim

    @property
    def tangent_dim(self) -> int:
        return self.manifold.tangent_dim


class StateSpec:
    """Layout descriptor for a flat ambient state vector."""

    def __init__(self, slots: list[StateSlot]) -> None:
        self._slots = tuple(slots)
        ambient_offset = tangent_offset = 0
        for slot in self._slots:
            if not isinstance(slot, StateSlot):
                raise TypeError(
                    f"StateSpec: slots must be StateSlot instances, got "
                    f"{type(slot).__name__}")
            if (slot.ambient_offset != ambient_offset
                    or slot.tangent_offset != tangent_offset):
                raise ValueError(
                    f"StateSpec: slot {slot.name!r} has offsets "
                    f"({slot.ambient_offset}, {slot.tangent_offset}); expected "
                    f"dense offsets ({ambient_offset}, {tangent_offset})")
            ambient_offset += slot.ambient_dim
            tangent_offset += slot.tangent_dim
        names = [s.name for s in self._slots]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"StateSpec: duplicate slot name(s) {duplicates}")
        self._slot_by_name = {s.name: s for s in self._slots}
        self._ambient_dim = sum(s.ambient_dim for s in self._slots)
        self._tangent_dim = sum(s.tangent_dim for s in self._slots)

    # ----- Construction --------------------------------------------------

    @classmethod
    def from_layout(cls, layout) -> StateSpec:
        """Build a spec from an ordered list of `(name, Manifold)` pairs.

        The world-agnostic constructor: a `RecurrenceBlock` (PID, Madgwick,
        the IMU integrator) declares its internal state this way, with no
        craft / world to walk. Offsets densify from 0 in the given order.
        """
        slots: list[StateSlot] = []
        offset = 0
        tan_offset = 0
        for name, manifold in layout:
            slots.append(StateSlot(name=name,
                                   ambient_offset=offset,
                                   manifold=manifold,
                                   tangent_offset=tan_offset))
            offset += manifold.ambient_dim
            tan_offset += manifold.tangent_dim
        return cls(slots)

    @classmethod
    def subset(cls, full_spec: StateSpec,
               kept_names) -> StateSpec:
        """Build a standalone spec over a subset of `full_spec`'s slots.

        Slots are kept in their original `full_spec` order; ambient and
        tangent offsets re-densify from 0 so every existing method
        (pack/unpack/boxplus/boxminus) works on the result unchanged.
        """
        kept = set(kept_names)
        return cls.from_layout([(s.name, s.manifold)
                                for s in full_spec.slots
                                if s.name in kept])

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
        """Lookup a slot by name.

        Exact match wins. As a convenience for single-craft worlds, a
        suffix match also works — `spec.slot("position")` finds
        `"drone.position"` when there's exactly one such slot. The
        suffix match raises if ambiguous (the shared `resolve_suffix`
        rule)."""
        full = resolve_suffix(name, self._slot_by_name,
                              label="slot", who="StateSpec.slot")
        return self._slot_by_name[full]

    def __contains__(self, name: str) -> bool:
        return name in self._slot_by_name

    def __repr__(self) -> str:
        return (f"<StateSpec ambient={self._ambient_dim} "
                f"tangent={self._tangent_dim} "
                f"slots={[s.name for s in self._slots]}>")

    # ----- Pack / Unpack -------------------------------------------------

    def pack(self, state_dict: dict[str, Any]) -> np.ndarray:
        """Flatten a tick-style dict into the ambient vector."""
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"StateSpec.pack: expected dict, got {type(state_dict).__name__}")
        expected = set(self._slot_by_name)
        supplied = set(state_dict)
        unknown = sorted(supplied - expected)
        missing = sorted(expected - supplied)
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unknown keys {unknown}")
            if missing:
                details.append(f"missing keys {missing}")
            raise ValueError(f"StateSpec.pack: {'; '.join(details)}")
        flat = np.zeros(self._ambient_dim, dtype=float)
        for slot in self._slots:
            value = state_dict[slot.name]
            raw = np.asarray(value)
            if raw.dtype.kind not in "iuf":
                raise TypeError(
                    f"StateSpec.pack: slot {slot.name!r} must contain real "
                    f"numeric data, got dtype {raw.dtype}")
            arr = np.atleast_1d(np.asarray(raw, dtype=float)).reshape(-1)
            if arr.size != slot.ambient_dim:
                raise ValueError(
                    f"StateSpec.pack: slot {slot.name!r} expects dim {slot.ambient_dim}, "
                    f"got len={arr.size}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"StateSpec.pack: slot {slot.name!r} contains non-finite values")
            flat[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim] = arr
        return flat

    def unpack(self, flat: np.ndarray) -> dict[str, Any]:
        """Slice an ambient vector back into a tick-style dict."""
        raw = np.asarray(flat)
        if raw.dtype.kind not in "iuf":
            raise TypeError(
                f"StateSpec.unpack: expected real numeric data, got {raw.dtype}")
        flat = np.asarray(raw, dtype=float)
        if flat.shape != (self._ambient_dim,):
            raise ValueError(
                f"StateSpec.unpack: expected shape ({self._ambient_dim},), "
                f"got {flat.shape}")
        if not np.all(np.isfinite(flat)):
            raise ValueError("StateSpec.unpack: state contains non-finite values")
        out: dict[str, Any] = {}
        for slot in self._slots:
            chunk = flat[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
            if slot.ambient_dim == 1:
                out[slot.name] = float(chunk[0])
            else:
                out[slot.name] = chunk.copy()
        return out

    def pack_any(self, state=None, *, base=None) -> np.ndarray:
        """Pack a state given in ANY of the three shapes — nested dict
        (`{owner: {slot: v}}`), flat dict (`{"owner.slot": v}`), or ambient
        vector — merged over `base` (ambient vector or dict), into the
        ambient vector. The single owner of shape conversion: every
        runtime/analysis call site goes through here instead of hand-rolling
        the flatten + merge + filter loop.

        Rules:
          * `state` is an ndarray → taken verbatim (length-checked).
          * `state` is a dict     → flattened; unknown keys rejected; merged
            over `base`.
          * `state` is None       → `base` packed as-is.
          * every slot must be covered by `state` or `base`.
        """
        if isinstance(state, np.ndarray):
            if state.dtype.kind not in "iuf":
                raise TypeError(
                    f"StateSpec.pack_any: expected real numeric data, got "
                    f"{state.dtype}")
            arr = np.asarray(state, dtype=float).reshape(-1)
            if arr.shape != (self._ambient_dim,):
                raise ValueError(
                    f"StateSpec.pack_any: expected ambient vector of length "
                    f"{self._ambient_dim}, got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("StateSpec.pack_any: state contains non-finite values")
            return arr.copy()
        merged: dict[str, Any] = {}
        if base is not None:
            merged.update(self.unpack(np.asarray(base, dtype=float))
                          if isinstance(base, np.ndarray)
                          else flatten_nested(base))
        if state is not None:
            merged.update(flatten_nested(state))
        unknown = sorted(set(merged) - set(self._slot_by_name))
        if unknown:
            raise ValueError(f"StateSpec.pack_any: unknown keys {unknown}")
        return self.pack(merged)

    def pack_projected(self, source: dict[str, Any]) -> np.ndarray:
        """Pack this spec's slots from a deliberately broader model record.

        World authoring records also contain commands and noise placeholders.
        Internal code that consumes that mixed record must opt into projection
        explicitly; ordinary state APIs use :meth:`pack_any` and reject every
        unknown key.
        """
        flat = flatten_nested(source)
        selected = {slot.name: flat[slot.name]
                    for slot in self._slots if slot.name in flat}
        return self.pack(selected)

    def to_nested(self, flat_vector: np.ndarray) -> dict[str, Any]:
        """Ambient vector → nested `{owner: {slot: v}}` dict (the runtime's
        user-facing shape). A dotless slot name stays a top-level entry."""
        nested: dict[str, Any] = {}
        for full, val in self.unpack(np.asarray(flat_vector,
                                                dtype=float)).items():
            if "." in full:
                owner, slot = full.split(".", 1)
                nested.setdefault(owner, {})[slot] = val
            else:
                nested[full] = val
        return nested

    # ----- Manifold operations (symbolic, for the ESKF tracer) ----------

    def boxplus_sym(self, x_ambient: ca.MX, delta_tangent: ca.MX) -> ca.MX:
        """Apply a tangent-space perturbation to an ambient state.

        Per-slot manifold operation:
          R3 / R1    →  x_new = x + δ
          SO3        →  q_new = boxplus(q, δ_θ)         (quaternion + axis-angle)

        Returns an ambient-dimensioned MX. Used by the ESKF to build a
        symbolic perturbed state for Jacobian extraction:

            δ_out = boxminus(f(boxplus(x, δ_in)), f(x))
            F     = ∂δ_out / ∂δ_in  evaluated at δ_in = 0
        """
        chunks: list[ca.MX] = []
        for slot in self._slots:
            x_chunk = x_ambient[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
            d_chunk = delta_tangent[
                slot.tangent_offset : slot.tangent_offset + slot.tangent_dim]
            chunks.append(slot.manifold.boxplus_sym(x_chunk, d_chunk))
        return ca.vertcat(*chunks)

    def boxminus_sym(self, x_a: ca.MX, x_b: ca.MX) -> ca.MX:
        """Compute the tangent-space delta between two ambient states.

        Per-slot:
          R3 / R1    →  δ = x_a − x_b
          SO3        →  δ_θ = boxminus(q_a, q_b) = log(q_a ⊗ q_b⁻¹)
        """
        chunks: list[ca.MX] = []
        for slot in self._slots:
            a_chunk = x_a[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
            b_chunk = x_b[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
            chunks.append(slot.manifold.boxminus_sym(a_chunk, b_chunk))
        return ca.vertcat(*chunks)

    def boxplus_num(self, x_ambient: np.ndarray,
                    delta_tangent: np.ndarray) -> np.ndarray:
        """Numeric (numpy) boxplus. Used by the ESKF update step to apply
        the Kalman correction onto the manifold without going through a
        compiled symbolic function."""
        x = np.asarray(x_ambient, dtype=float).copy()
        d = np.asarray(delta_tangent, dtype=float)
        for slot in self._slots:
            x_chunk = x[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
            d_chunk = d[slot.tangent_offset :
                        slot.tangent_offset + slot.tangent_dim]
            new_chunk = slot.manifold.boxplus_num(x_chunk, d_chunk)
            x[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim] = new_chunk
        return x
