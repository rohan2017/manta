"""EKF — Error-state (manifold-aware) Extended Kalman Filter on a Craft.

Design intent:

  * The user defines a Craft as usual.
  * They wrap it: `ekf = EKF(craft)`.
  * Internally the EKF compiles three CasADi Functions:
      - the nominal predict f(x_ambient, dt) → x_new_ambient;
      - the tangent-space F(x, dt) = ∂δ_out / ∂δ_in evaluated at δ_in=0,
        where δ_out = boxminus(f(boxplus(x, δ_in)), f(x));
      - the tangent layout's dimensions for sizing P / Q.
  * State is held in two parts:
      - mean `x` lives on the manifold (ambient form, e.g. q stays unit).
      - covariance `P` lives on the tangent (no redundant radial direction
        on SO(3); 12-D rigid-body block instead of 13-D).
  * Per step:
        ekf.predict(dt, Q)
        ekf.update(h_sym, z, R)
    `h_sym` is `MX → MX` returning the symbolic measurement vector h(x).
    The EKF builds the tangent-space H = ∂h_pert/∂δ at δ=0 internally,
    so user h doesn't need to know about tangents.
  * Update applies via `x ← boxplus(x, K · y)` — keeps q on the unit
    sphere by construction; no defensive renormalization needed.

Scope notes:
  * Single craft only. Multi-craft via concatenated StateSpec lands with
    the World/Coupling layer.
  * No external Inputs into the predict yet (no Input declarations).
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from .. import ir
from .state_spec import StateSpec


class EKF:
    """Error-state EKF over a manta_next Craft.

    Constructor compiles the predict + F functions once; per-step
    measurement updates re-build h's symbolic Jacobian on the fly
    (cheap since the graph is small).
    """

    def __init__(self,
                 craft,
                 *,
                 gravity_anchor: tuple[float, float, float] = (0.0, 0.0, -9.81),
                 ) -> None:
        self.craft = craft
        self.spec  = StateSpec.from_craft(craft)

        n_ambient = self.spec.ambient_dim
        n_tangent = self.spec.tangent_dim

        # --- Lift the Craft's tick to a flat ambient → ambient function -----
        compiled_tick = craft.compile_tick(gravity_anchor=gravity_anchor)
        cf = compiled_tick.casadi_function

        # Walk the tick's input names to enumerate part Inputs (order
        # matches CompiledGraph input order — stable across compiles).
        self._input_names: list[str] = self._discover_input_names(cf)
        n_u = len(self._input_names)
        # Defaults from each part's current attribute value at compile time.
        self._u_defaults = np.array(
            [self._input_default(name) for name in self._input_names],
            dtype=float)

        x_sym  = ca.MX.sym("x",  n_ambient, 1)
        u_sym  = ca.MX.sym("u",  n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
        dt_sym = ca.MX.sym("dt", 1, 1)
        x_new  = self._tick_on_flat(cf, x_sym, u_sym, dt_sym)

        # --- Tangent-space linearization -----------------------------------
        # δ_out(δ_in) = boxminus( f( boxplus(x, δ_in), u ), f(x, u) )
        # F = ∂δ_out / ∂δ_in evaluated at δ_in = 0. F is independent of u
        # in the linearization, but we keep u in the signature so callers
        # don't have to track when it matters.
        delta_in = ca.MX.sym("delta_in", n_tangent, 1)
        x_pert    = self.spec.boxplus_sym(x_sym, delta_in)
        x_pert_new = self._tick_on_flat(cf, x_pert, u_sym, dt_sym)
        delta_out = self.spec.boxminus_sym(x_pert_new, x_new)
        F_sym = ca.substitute(
            ca.jacobian(delta_out, delta_in),
            delta_in,
            ca.MX.zeros(n_tangent, 1),
        )

        self._f_fn = ca.Function("ekf_predict",
                                  [x_sym, u_sym, dt_sym], [x_new],
                                  ["x", "u", "dt"], ["x_new"])
        self._F_fn = ca.Function("ekf_F",
                                  [x_sym, u_sym, dt_sym], [F_sym],
                                  ["x", "u", "dt"], ["F"])
        # Cache for update-time use.
        self._x_sym = x_sym
        self._delta_zero_sym = ca.MX.sym("delta_zero", n_tangent, 1)

        # --- Initial state + covariance ------------------------------------
        self._x = self.spec.pack(craft.initial_state())
        self._P = np.eye(n_tangent) * 1e-2

    # ----- helpers ----------------------------------------------------------

    def _discover_input_names(self, cf: ca.Function) -> list[str]:
        """Return the ordered list of part-Input names present in the
        tick's CasADi-Function signature. Used at __init__ to size the u
        vector and at predict() to route u-dict values into the call."""
        out: list[str] = []
        n_in = cf.n_in()
        for i in range(n_in):
            name = cf.name_in(i)
            if name == "dt" or name in self.spec:
                continue
            if "." in name:
                part_name, input_name = name.split(".", 1)
                part = next(
                    (p for p in self.craft.parts if p.name == part_name),
                    None,
                )
                if part and input_name in part.input_declarations():
                    out.append(name)
                    continue
            raise RuntimeError(
                f"EKF: tick input {name!r} not in StateSpec and not "
                f"recognized as a part Input.")
        return out

    def _input_default(self, full_name: str) -> float:
        part_name, input_name = full_name.split(".", 1)
        part = next(p for p in self.craft.parts if p.name == part_name)
        return float(getattr(part, input_name))

    def _tick_on_flat(self,
                      cf: ca.Function,
                      x_sym: ca.MX,
                      u_sym: ca.MX,
                      dt_sym: ca.MX) -> ca.MX:
        """Apply the Craft's compiled tick CasADi Function to a flat
        ambient-state symbolic vector + flat input vector + dt, and concat
        the named outputs back to a flat ambient vector in the StateSpec's
        slot order. Part-Output slots in the tick result are ignored
        (they're observables, not propagated state)."""
        n_in = cf.n_in()
        in_names  = [cf.name_in(i)  for i in range(n_in)]
        n_out = cf.n_out()
        out_names = [cf.name_out(i) for i in range(n_out)]

        # Build name → index in u_sym lookup once.
        u_index = {name: i for i, name in enumerate(self._input_names)}

        sliced: list[ca.MX] = []
        for name in in_names:
            if name == "dt":
                sliced.append(dt_sym)
            elif name in self.spec:
                slot = self.spec.slot(name)
                sliced.append(x_sym[slot.offset : slot.offset + slot.dim])
            elif name in u_index:
                sliced.append(u_sym[u_index[name]])
            else:
                raise RuntimeError(
                    f"EKF: tick input {name!r} not handled.")

        result = cf(*sliced)
        result_by_name = (
            {out_names[0]: result}
            if n_out == 1
            else {n: result[i] for i, n in enumerate(out_names)}
        )

        chunks: list[ca.MX] = []
        for slot in self.spec.slots:
            if slot.name not in result_by_name:
                raise RuntimeError(f"EKF: tick output {slot.name!r} missing")
            chunk = result_by_name[slot.name]
            if chunk.shape != (slot.dim, 1):
                chunk = ca.reshape(chunk, slot.dim, 1)
            chunks.append(chunk)
        return ca.vertcat(*chunks)

    # ----- Accessors --------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """The current ambient state vector. Use `state_dict()` for the
        per-slot keyed view."""
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        """The current tangent-space covariance matrix."""
        return self._P.copy()

    def state_dict(self) -> dict[str, Any]:
        return self.spec.unpack(self._x)

    def reset(self, *,
              state: dict | None = None,
              P: np.ndarray | None = None) -> None:
        """Reset the EKF state and/or covariance.

        `P` must be tangent-dimensioned (spec.tangent_dim × spec.tangent_dim).
        """
        if state is not None:
            full = self.craft.initial_state()
            full.update(state)
            self._x = self.spec.pack(full)
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (self.spec.tangent_dim, self.spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"EKF.reset: P shape {P.shape} doesn't match tangent "
                    f"dim {expected}")
            self._P = P.copy()

    # ----- Predict ----------------------------------------------------------

    def predict(self,
                dt: float,
                u: dict[str, float] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the nominal state and tangent covariance by `dt`.

        x       ← f(x, u, dt)                         (ambient)
        P       ← F P Fᵀ + Q                          (tangent)

        Args:
            dt    integrator timestep.
            u     dict of `"<part>.<input>"` → float for per-tick input
                  values. Missing entries fall back to the part-instance
                  default captured at construction time. With no Input
                  slots in the craft, `u` is ignored.
            Q     process-noise covariance (tangent-dim square).
        """
        u_vec = self._build_u(u)
        x_new = np.asarray(self._f_fn(self._x, u_vec, dt)).reshape(-1)
        F     = np.asarray(self._F_fn(self._x, u_vec, dt))
        n = self.spec.tangent_dim
        if Q is None:
            Q = np.zeros((n, n))
        self._x = x_new
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)
        # No need to project the mean — the tick already runs through
        # SO3.boxplus in Craft's integrator and renormalizes the quat.

    def _build_u(self, u: dict[str, float] | None) -> np.ndarray:
        if not self._input_names:
            return np.zeros(0)
        if u is None:
            return self._u_defaults.copy()
        unknown = set(u) - set(self._input_names)
        if unknown:
            raise KeyError(
                f"EKF.predict: unknown input name(s) {sorted(unknown)}. "
                f"Available: {sorted(self._input_names)}")
        out = self._u_defaults.copy()
        for i, name in enumerate(self._input_names):
            if name in u:
                out[i] = float(u[name])
        return out

    # ----- Update -----------------------------------------------------------

    def update(self,
               h_sym: Callable[[ca.MX], ca.MX],
               z: np.ndarray,
               R: np.ndarray) -> None:
        """Apply a measurement.

        Args:
            h_sym  — a function `MX → MX` that, given the symbolic ambient
                     state vector x, returns h(x) (the model prediction).
            z      — the observed measurement (numpy 1-D).
            R      — measurement noise covariance.

        The EKF builds the tangent-space H = ∂h_pert/∂δ at δ=0 internally,
        applies the standard Kalman gain on the tangent, and projects the
        correction back through `boxplus(x, K·y)`. Joseph form keeps P
        PSD over long runs.
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"EKF.update: R shape {R.shape} doesn't match z size {z.size}")

        h_mx = h_sym(self._x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)

        # Tangent-space H: how does h change in response to a δ-perturbation
        # of x?  H = ∂h(boxplus(x, δ))/∂δ at δ=0.
        delta = ca.MX.sym("delta_h", self.spec.tangent_dim, 1)
        x_pert = self.spec.boxplus_sym(self._x_sym, delta)
        h_pert = h_sym(x_pert)
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(
            ca.jacobian(h_pert, delta),
            delta,
            ca.MX.zeros(self.spec.tangent_dim, 1),
        )

        h_fn = ca.Function("h_fn", [self._x_sym], [h_mx])
        H_fn = ca.Function("H_fn", [self._x_sym], [H_sym])

        h_x = np.asarray(h_fn(self._x)).reshape(-1)
        H   = np.asarray(H_fn(self._x))

        if h_x.size != z.size:
            raise ValueError(
                f"EKF.update: h(x) has size {h_x.size}, z has size {z.size}")

        y = z - h_x
        S = H @ self._P @ H.T + R
        # K = P Hᵀ S⁻¹ via solve for numerical stability.
        K = np.linalg.solve(S.T, (self._P @ H.T).T).T

        # Apply tangent correction to the manifold-valued mean.
        delta_x = K @ y
        self._x = self.spec.boxplus(self._x, delta_x)

        # Joseph form: P = (I − KH) P (I − KH)ᵀ + K R Kᵀ.
        I = np.eye(self.spec.tangent_dim)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)


# ---------------------------------------------------------------------------
# Helpers — common measurement-function builders
# ---------------------------------------------------------------------------

def measurement_slot(spec: StateSpec, name: str):
    """Return an h_sym callable that reads slot `name` directly from x.

    Equivalent to:  h(x) = x[slot.offset : slot.offset + slot.dim].
    Useful for synthetic / oracle sensors that observe a single state
    slot — primarily for testing the EKF plumbing.
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
