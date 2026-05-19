"""EKF — Extended Kalman Filter operating on a manta_next Craft.

Design intent:

  * The user defines a Craft as usual (parts + state).
  * They wrap it: `ekf = EKF(craft)`.
  * Internally the EKF compiles two things via CasADi:
      - the predict function f(x, dt) → x_new from the Craft's tick;
      - the Jacobian F(x, dt) = ∂f/∂x, symbolically differentiated.
    Compilation happens once at construction; the resulting CasADi
    Functions are reused for every predict/update step.
  * User loop:
        ekf.predict(dt, Q)
        ekf.update(measurement_fn, z, R)
        ...
    `measurement_fn` is a Python callable that takes a StateSpec and a
    state dict and returns (z_pred_callable, h_jacobian_callable). For
    M4 we provide a helper `state_slot_measurement(name)` that picks a
    scalar/vector slot directly out of state (e.g., "z position").

Scope notes:
  * The ambient state vector includes the orientation quaternion as 4
    floats; covariance is sized to the AMBIENT dim. That's the simple-but-
    imperfect formulation; manifold-aware ESKF (with 3-dim tangent for
    SO3) is M5.
  * Single-craft only for M4. Multi-craft via concatenated StateSpec
    arrives with the World/Coupling layer.
  * No external inputs (no Input declarations yet); the predict function
    ingests only (state, dt).
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from . import ir
from .state_spec import StateSpec


class EKF:
    """A simple ambient-state EKF over a manta_next Craft."""

    def __init__(self,
                 craft,
                 *,
                 gravity_anchor: tuple[float, float, float] = (0.0, 0.0, -9.81),
                 ) -> None:
        self.craft = craft
        self.spec  = StateSpec.from_craft(craft)

        # The Craft's tick is a CompiledGraph that takes named inputs.
        # For the EKF we want it as a flat-vector function:
        #     f(x_flat, dt) → x_flat_new
        # Build that on top of the craft's compiled tick.
        compiled_tick = craft.compile_tick(gravity_anchor=gravity_anchor)
        cf = compiled_tick.casadi_function
        # cf input order: as registered on the graph. We rely on the fact
        # that compile_tick declares them in the same order StateSpec uses
        # for the rigid-body block, followed by per-part states in declaration
        # order. dt is also an input.
        n_in  = cf.n_in()
        in_names  = [cf.name_in(i)  for i in range(n_in)]
        in_sizes  = [cf.size_in(i)  for i in range(n_in)]
        n_out = cf.n_out()
        out_names = [cf.name_out(i) for i in range(n_out)]

        # Build a symbolic flat state vector x, plus dt.
        x_sym  = ca.MX.sym("x",  self.spec.ambient_dim, 1)
        dt_sym = ca.MX.sym("dt", 1, 1)

        # Slice x_sym into the input order the compiled tick expects.
        sliced_args: list[ca.MX] = []
        for name, size in zip(in_names, in_sizes):
            if name == "dt":
                sliced_args.append(dt_sym)
                continue
            if name in self.spec:
                slot = self.spec.slot(name)
                chunk = x_sym[slot.offset : slot.offset + slot.dim]
                sliced_args.append(chunk)
            else:
                raise RuntimeError(
                    f"EKF: tick input {name!r} not found in StateSpec.")

        result = cf(*sliced_args)
        # Concat outputs in StateSpec order to get the new flat state.
        result_by_name = (
            {out_names[0]: result}
            if n_out == 1
            else {n: result[i] for i, n in enumerate(out_names)}
        )
        new_chunks = []
        for slot in self.spec.slots:
            if slot.name not in result_by_name:
                raise RuntimeError(
                    f"EKF: tick output {slot.name!r} missing.")
            chunk = result_by_name[slot.name]
            # Ensure column vector.
            if chunk.shape == (1, 1):
                new_chunks.append(chunk)
            elif chunk.shape == (1,):
                new_chunks.append(chunk)
            else:
                new_chunks.append(ca.reshape(chunk, slot.dim, 1))
        x_new = ca.vertcat(*new_chunks)

        # F = ∂x_new/∂x evaluated symbolically — sparse-aware via CasADi.
        F_sym = ca.jacobian(x_new, x_sym)

        self._f_fn = ca.Function("ekf_predict", [x_sym, dt_sym], [x_new],
                                  ["x", "dt"], ["x_new"])
        self._F_fn = ca.Function("ekf_F",       [x_sym, dt_sym], [F_sym],
                                  ["x", "dt"], ["F"])
        # Cache the symbolic state vector so update() can build h(x) easily.
        self._x_sym = x_sym

        # State + covariance, initialized at compile time. The user can
        # overwrite via reset(...).
        self._x = self.spec.pack(craft.initial_state())
        self._P = np.eye(self.spec.ambient_dim) * 1e-2

    # ---- Accessors ------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """The current (ambient) state vector. Use `state_dict()` for the
        per-slot keyed view."""
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        """The current covariance matrix (ambient × ambient)."""
        return self._P.copy()

    def state_dict(self) -> dict[str, Any]:
        return self.spec.unpack(self._x)

    def reset(self, *,
              state: dict | None = None,
              P: np.ndarray | None = None) -> None:
        """Reset the EKF state and/or covariance."""
        if state is not None:
            full = self.craft.initial_state()
            full.update(state)
            self._x = self.spec.pack(full)
        if P is not None:
            P = np.asarray(P, dtype=float)
            assert P.shape == (self.spec.ambient_dim, self.spec.ambient_dim)
            self._P = P.copy()

    # ---- Predict --------------------------------------------------------

    def predict(self, dt: float, Q: np.ndarray | None = None) -> None:
        """Advance the state and covariance by `dt`.

        State: x ← f(x, dt).
        Covariance: P ← F P Fᵀ + Q.
        """
        x_new = np.asarray(self._f_fn(self._x, dt)).reshape(-1)
        F     = np.asarray(self._F_fn(self._x, dt))
        if Q is None:
            Q = np.zeros((self.spec.ambient_dim, self.spec.ambient_dim))
        self._x = x_new
        self._P = F @ self._P @ F.T + Q
        # Symmetrize.
        self._P = 0.5 * (self._P + self._P.T)
        # Ambient EKF: renormalize any SO3 slots so the quaternion stays on
        # the unit sphere despite floating-point drift in the tick + the
        # additive structure of P·Fᵀ·F that doesn't preserve unit norm.
        self._renormalize_manifold_slots()

    # ---- Update --------------------------------------------------------

    def update(self,
               h_sym: Callable[[ca.MX], ca.MX],
               z: np.ndarray,
               R: np.ndarray) -> None:
        """Apply a measurement.

        Args:
            h_sym  — a function `MX → MX` that, given the symbolic state
                     vector x, returns a symbolic measurement vector
                     h(x). Used to extract a CasADi Function for the
                     measurement and its Jacobian.
            z      — the observed measurement (numpy 1-D).
            R      — measurement noise covariance (numpy 2-D).

        Updates state and covariance via the standard EKF gain formula::

            y = z - h(x)
            S = H P Hᵀ + R
            K = P Hᵀ S⁻¹
            x ← x + K y
            P ← (I - K H) P (I - K H)ᵀ + K R Kᵀ      (Joseph form)
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"EKF.update: R shape {R.shape} doesn't match z size {z.size}")

        h_mx = h_sym(self._x_sym)
        # Allow callers to return a scalar or 1-D MX; reshape to column.
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        H_mx = ca.jacobian(h_mx, self._x_sym)

        h_fn = ca.Function("h_fn", [self._x_sym], [h_mx])
        H_fn = ca.Function("H_fn", [self._x_sym], [H_mx])

        h_x = np.asarray(h_fn(self._x)).reshape(-1)
        H   = np.asarray(H_fn(self._x))

        if h_x.size != z.size:
            raise ValueError(
                f"EKF.update: h(x) has size {h_x.size}, z has size {z.size}")

        y = z - h_x
        S = H @ self._P @ H.T + R
        # K = P Hᵀ S⁻¹ via solve for numerical stability.
        K = np.linalg.solve(S.T, (self._P @ H.T).T).T

        self._x = self._x + K @ y

        # Joseph form: P = (I − KH) P (I − KH)ᵀ + K R Kᵀ.
        I = np.eye(self.spec.ambient_dim)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)
        self._renormalize_manifold_slots()

    # ---- Manifold maintenance ------------------------------------------

    def _renormalize_manifold_slots(self) -> None:
        """Project SO(3) slot back onto the unit-norm quaternion manifold.

        Pure ambient-EKF formulation: the Kalman update can pull q off the
        unit sphere; we renormalize to recover. The covariance keeps its
        ambient form (a 4×4 block on the quaternion); a proper ESKF would
        carry a 3-DOF tangent covariance instead. M5 work.
        """
        for slot in self.spec.slots:
            if slot.manifold != "SO3":
                continue
            chunk = self._x[slot.offset : slot.offset + slot.dim]
            n = float(np.linalg.norm(chunk))
            if n > 1e-12:
                self._x[slot.offset : slot.offset + slot.dim] = chunk / n
            else:
                # Degenerate — reset to identity rotation. Shouldn't happen
                # in practice but avoids NaN propagation.
                self._x[slot.offset : slot.offset + slot.dim] = np.array(
                    [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Helpers — common measurement-function builders
# ---------------------------------------------------------------------------

def measurement_slot(spec: StateSpec, name: str):
    """Return an h_sym callable that reads slot `name` directly from x.

    Equivalent to:  h(x) = x[slot.offset : slot.offset + slot.dim].
    Useful for synthetic / oracle sensors that observe a single state
    slot — typically only realistic for testing, but lets the EKF
    plumbing be exercised before fully realistic sensor parts arrive.
    """
    slot = spec.slot(name)
    def h_sym(x):
        return x[slot.offset : slot.offset + slot.dim]
    return h_sym


def measurement_component(spec: StateSpec, name: str, component: int):
    """Return an h_sym callable for a single component of a slot
    (e.g., the z component of position)."""
    slot = spec.slot(name)
    if not (0 <= component < slot.dim):
        raise IndexError(
            f"measurement_component: slot {name!r} has dim {slot.dim}, "
            f"component {component} out of range")
    def h_sym(x):
        return x[slot.offset + component]
    return h_sym
