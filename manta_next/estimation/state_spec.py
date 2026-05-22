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

    spec = StateSpec.from_craft(craft)
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
from typing import Any

import casadi as ca
import numpy as np

from ..parts.base import Part, State


@dataclass(frozen=True)
class StateSlot:
    """One named slot in the flat layout.

    Attrs:
        name        — string key in the tick dict (e.g., "position",
                      "wheel.angle"). Same key the tick function uses.
        offset      — start index in the ambient vector.
        dim         — width in the ambient vector (1, 3, or 4).
        manifold    — 'R1' (scalar), 'R3' (3-vector), 'SO3' (quaternion 4→3 tan).
                      Determines the tangent layout for an ESKF.
        tangent_dim — width in the tangent vector. 4 → 3 for SO3, else
                      equal to `dim`.
        tangent_offset — start index in the tangent vector.
    """
    name: str
    offset: int
    dim: int
    manifold: str
    tangent_dim: int
    tangent_offset: int


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
        tan_offset = 0

        def add(name: str, dim: int, manifold: str, tangent_dim: int):
            nonlocal offset, tan_offset
            slots.append(StateSlot(name=name,
                                   offset=offset,
                                   dim=dim,
                                   manifold=manifold,
                                   tangent_dim=tangent_dim,
                                   tangent_offset=tan_offset))
            offset += dim
            tan_offset += tangent_dim

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
            # RW Noise declarations synthesize a bias state slot per
            # channel — `<part>.<name>` is the bias; the driver is a
            # noise input handled by the EKF auto-Q path. R3 for vec3
            # channels, R1 for scalar channels. Channels with
            # sigma == 0 are inert (no bias state created).
            for nname, ndecl in part.noise_declarations().items():
                if ndecl.kind != "rw":
                    continue
                sigma = float(getattr(part, f"{nname}_sigma"))
                if sigma <= 0.0:
                    continue
                key = f"{part.name}.{nname}"
                if ndecl.shape == "scalar":
                    add(key, 1, "R1", 1)
                else:
                    add(key, 3, "R3", 3)

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
        from ..ir.manifold import _so3_exp
        chunks: list[ca.MX] = []
        for slot in self._slots:
            x_chunk = x_ambient[slot.offset : slot.offset + slot.dim]
            d_chunk = delta_tangent[
                slot.tangent_offset : slot.tangent_offset + slot.tangent_dim]
            if slot.manifold in ("R1", "R3"):
                chunks.append(x_chunk + d_chunk)
            elif slot.manifold == "SO3":
                # q ⊞ δ = q ⊗ exp(δ): apply tangent perturbation on the
                # RIGHT (left-trivialized would be exp(δ) ⊗ q). Convention
                # matches manta_next.ir.manifold.SO3.boxplus where the
                # delta lives in the from_frame's tangent space and the
                # rotation is appended on the left, but for state-vector
                # purposes either convention works as long as it's
                # consistent across boxplus and boxminus.
                dq = _so3_exp(d_chunk)
                # q_new = dq · q_chunk  (left trivialization).
                w1, x1, y1, z1 = dq[0], dq[1], dq[2], dq[3]
                w2 = x_chunk[0]; x2 = x_chunk[1]
                y2 = x_chunk[2]; z2 = x_chunk[3]
                w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                chunks.append(ca.vertcat(w, x, y, z))
            else:
                raise NotImplementedError(
                    f"boxplus_sym: manifold {slot.manifold!r} not supported")
        return ca.vertcat(*chunks)

    def boxminus_sym(self, x_a: ca.MX, x_b: ca.MX) -> ca.MX:
        """Compute the tangent-space delta between two ambient states.

        Per-slot:
          R3 / R1    →  δ = x_a − x_b
          SO3        →  δ_θ = boxminus(q_a, q_b) = log(q_a ⊗ q_b⁻¹)
        """
        from ..ir.manifold import _so3_log
        chunks: list[ca.MX] = []
        for slot in self._slots:
            a_chunk = x_a[slot.offset : slot.offset + slot.dim]
            b_chunk = x_b[slot.offset : slot.offset + slot.dim]
            if slot.manifold in ("R1", "R3"):
                chunks.append(a_chunk - b_chunk)
            elif slot.manifold == "SO3":
                # q_a · q_b⁻¹ → tangent via log.
                aw = a_chunk[0]; ax = a_chunk[1]
                ay = a_chunk[2]; az = a_chunk[3]
                bw =  b_chunk[0]; bx = -b_chunk[1]
                by = -b_chunk[2]; bz = -b_chunk[3]   # conjugate
                w = aw*bw - ax*bx - ay*by - az*bz
                x = aw*bx + ax*bw + ay*bz - az*by
                y = aw*by - ax*bz + ay*bw + az*bx
                z = aw*bz + ax*by - ay*bx + az*bw
                dq = ca.vertcat(w, x, y, z)
                chunks.append(_so3_log(dq))
            else:
                raise NotImplementedError(
                    f"boxminus_sym: manifold {slot.manifold!r} not supported")
        return ca.vertcat(*chunks)

    def boxplus(self, x_ambient: np.ndarray,
                delta_tangent: np.ndarray) -> np.ndarray:
        """Numeric (numpy) boxplus. Used by the ESKF update step to apply
        the Kalman correction onto the manifold without going through a
        compiled symbolic function."""
        x = np.asarray(x_ambient, dtype=float).copy()
        d = np.asarray(delta_tangent, dtype=float)
        for slot in self._slots:
            x_chunk = x[slot.offset : slot.offset + slot.dim]
            d_chunk = d[slot.tangent_offset :
                        slot.tangent_offset + slot.tangent_dim]
            if slot.manifold in ("R1", "R3"):
                x[slot.offset : slot.offset + slot.dim] = x_chunk + d_chunk
            elif slot.manifold == "SO3":
                # exp of axis-angle then left-multiply onto current q.
                omega = d_chunk
                theta = float(np.linalg.norm(omega) + 1e-30)
                half = 0.5 * theta
                w = np.cos(half)
                v = omega * (np.sin(half) / theta)
                dq = np.array([w, v[0], v[1], v[2]])
                # q_new = dq * q_current.
                w1, x1, y1, z1 = dq
                w2, x2, y2, z2 = x_chunk
                q_new = np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])
                # Renormalize defensively (tangent step may be large).
                q_new = q_new / np.linalg.norm(q_new)
                x[slot.offset : slot.offset + slot.dim] = q_new
        return x
