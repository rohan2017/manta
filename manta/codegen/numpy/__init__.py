"""TargetNumpy — native-Python runtime evaluator backend.

`TargetNumpy(ir)` dispatches by IR type:

  * `TargetNumpy(Sim)` → `NumpyWorld` with `.step()` and
    `.initial_state()`.
  * `TargetNumpy(EKF)`           → `NumpyEKF` with `.predict()`,
    `.update()`, `.state_dict()`, `.reset()`, `.x`, `.P`.

The IR objects themselves (`Sim`, `EKF`) describe the
compiled symbolic graph but are not directly callable for ticking /
predicting — choose a target to get the runtime.
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np


# ---------------------------------------------------------------------------
# NumpyWorld — runtime for Sim
# ---------------------------------------------------------------------------

class NumpyWorld:
    """Native-Python evaluator wrapping a `Sim` IR.

    Provides the user-facing tick API: `initial_state()` and
    `step(state, dt, t=0)`. Internally calls the IR's CasADi function
    after translating between the user's nested state dict and the
    flat-prefixed casadi-input names.
    """

    def __init__(self, cw) -> None:
        from ...sim import Sim
        if not isinstance(cw, Sim):
            raise TypeError(
                f"NumpyWorld: expected Sim, got "
                f"{type(cw).__name__}")
        self._cw = cw

    # ---- IR passthroughs ----------------------------------------------

    @property
    def world(self):
        return self._cw.world

    @property
    def crafts(self):
        return self._cw.crafts

    @property
    def tick(self):
        return self._cw.tick

    # ---- State + step --------------------------------------------------

    def initial_state(self) -> dict[str, dict[str, Any]]:
        """Fresh copy of the per-owner initial state."""
        import copy
        return copy.deepcopy(self._cw._initial)

    def step(self,
             state: dict[str, dict[str, Any]],
             dt: float,
             t: float = 0.0) -> dict[str, dict[str, Any]]:
        """Advance the world by `dt`. Returns a fresh state dict.

        `t` is the world-clock time at the start of this step.
        Owners may be crafts or field disturbances; both use the same
        `<owner>.<slot>` convention in the underlying casadi function.
        """
        flat: dict[str, Any] = {"dt": dt, "t": t}
        for owner_name, owner_state in state.items():
            for slot, val in owner_state.items():
                flat[f"{owner_name}.{slot}"] = val

        out = self._cw.tick(**flat)

        new_state: dict[str, dict[str, Any]] = {k: {} for k in state}
        for key, val in out.items():
            owner_name, slot = key.split(".", 1)
            if owner_name not in new_state:
                new_state[owner_name] = {}
            new_state[owner_name][slot] = val
        # Preserve input-only slots the tick doesn't write back.
        for owner_name, owner_state in state.items():
            for slot, val in owner_state.items():
                if slot not in new_state[owner_name]:
                    new_state[owner_name][slot] = val
        return new_state

    def __repr__(self) -> str:
        return f"<NumpyWorld over {self._cw!r}>"


# ---------------------------------------------------------------------------
# NumpyEKF — runtime for EKF (mutable state + predict/update)
# ---------------------------------------------------------------------------

class NumpyEKF:
    """Native-Python evaluator wrapping an `EKF` IR.

    Holds mutable `_x` (ambient state vector) and `_P` (tangent
    covariance). Calls the IR's compiled predict / measurement
    functions during `predict()` and `update()`.
    """

    def __init__(self, ekf) -> None:
        from ...estimation.ekf import EKF
        if not isinstance(ekf, EKF):
            raise TypeError(
                f"NumpyEKF: expected EKF, got {type(ekf).__name__}")
        self._ekf = ekf
        self._x   = self._initial_x()
        self._P   = np.eye(ekf.spec.tangent_dim) * 1e-2

        # Measurement bus: one mailbox per registered sensor, in
        # registration order (= the order step() applies them). The user
        # drops a reading in via feed(); step() consumes fresh ones.
        self._meas: dict[str, dict[str, Any]] = {}
        for key, spec_o in ekf._sensors.items():
            self._meas[spec_o["full"]] = {
                "value": None, "fresh": False, "t": None,
                "dim": spec_o["dim"], "key": key,
            }
        # Latched (zero-order-hold) control inputs, read by step()'s
        # predict. Full or craft-relative names; resolved per predict.
        self.inputs: dict[str, float] = {}
        # Filter clock — advances by dt each step (or to the supplied t).
        self._t = 0.0

    def _initial_x(self) -> np.ndarray:
        """Pack the world's nested initial-state dict into the spec's
        flat ambient layout."""
        ekf = self._ekf
        init_flat: dict[str, Any] = {}
        for owner_name, owner_state in ekf.world._initial_state_dict().items():
            for k, v in owner_state.items():
                init_flat[f"{owner_name}.{k}"] = v
        init_for_pack = {k: v for k, v in init_flat.items() if k in ekf.spec}
        return ekf.spec.pack(init_for_pack)

    # ---- IR passthroughs ----------------------------------------------

    @property
    def ekf(self):
        """The wrapped EKF IR (carries the symbolic functions,
        sensor table, etc.)."""
        return self._ekf

    @property
    def spec(self):
        return self._ekf.spec

    @property
    def world(self):
        return self._ekf.world

    @property
    def crafts(self):
        return self._ekf.crafts

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    # ---- State view + reset --------------------------------------------

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner — same shape as
        `NumpyWorld.initial_state()`.
        """
        flat = self._ekf.spec.unpack(self._x)
        nested: dict[str, dict[str, Any]] = {}
        for full_name, val in flat.items():
            owner_name, slot = full_name.split(".", 1)
            nested.setdefault(owner_name, {})[slot] = val
        return nested

    def reset(self, *,
              state: dict | None = None,
              P: np.ndarray | None = None) -> None:
        """Reset the EKF state and/or covariance.

        `state` accepts either:
          * Nested `{owner: {slot: value}}` (canonical, matches
            `state_dict()`).
          * Flat dict keyed by full slot name (`"<owner>.<slot>"`).
        Missing entries fall back to the world's add_craft defaults.
        """
        ekf = self._ekf
        if state is not None:
            full_flat: dict[str, Any] = {}
            for owner_name, owner_state in ekf.world._initial_state_dict().items():
                for k, v in owner_state.items():
                    full_flat[f"{owner_name}.{k}"] = v
            for k, v in state.items():
                if isinstance(v, dict):
                    for slot, val in v.items():
                        full_flat[f"{k}.{slot}"] = val
                else:
                    full_flat[k] = v
            full_flat = {k: v for k, v in full_flat.items() if k in ekf.spec}
            self._x = ekf.spec.pack(full_flat)
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (ekf.spec.tangent_dim, ekf.spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"NumpyEKF.reset: P shape {P.shape} doesn't match "
                    f"tangent dim {expected}")
            self._P = P.copy()

    # ---- Predict -------------------------------------------------------

    def predict(self,
                dt: float,
                *,
                t: float = 0.0,
                u: dict[str, float] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the nominal state and tangent covariance by `dt`.

        Args mirror `EKF.predict` from the old monolithic class:
        `dt` (timestep), `t` (world-clock time), `u` (per-tick part
        Input override dict), `Q` (process-noise override; default is
        auto-assembly from registered Noise channels).

        The covariance is propagated per independent-subsystem block
        (`ekf._blocks`): with one block this is the usual dense
        `F P Fᵀ + Q`; with several (uncoupled crafts estimated jointly)
        each block propagates on its own — Σ O(n_b³) instead of O(n³).
        Off-diagonal (cross-block) covariance stays zero by construction
        of the partition, so it is never touched.
        """
        ekf = self._ekf
        u_vec = ekf._build_u(u)
        x_new = np.asarray(ekf._f_fn(self._x, u_vec, dt, t)).reshape(-1)
        F     = np.asarray(ekf._F_fn(self._x, u_vec, dt, t))
        L = None
        if Q is None and ekf._L_fn is not None:
            L = np.asarray(ekf._L_fn(self._x, u_vec, dt, t))
        P = self._P
        for idx in ekf._blocks:
            sub = np.ix_(idx, idx)
            Fbb = F[sub]
            if Q is not None:
                Qbb = Q[sub]
            elif L is not None:
                Lb  = L[idx, :]
                Qbb = Lb @ ekf._Sigma @ Lb.T
            else:
                Qbb = 0.0
            P[sub] = Fbb @ P[sub] @ Fbb.T + Qbb
        self._x = x_new
        self._P = 0.5 * (P + P.T)

    # ---- Update --------------------------------------------------------

    def update(self, *args, **kwargs) -> None:
        """Polymorphic update.

        Sensor form (auto h, H, R from Noise declarations):
            ekf.update(part, gyro=z_gyro, accel=z_accel)

        Low-level form (caller-supplied h, R):
            ekf.update(h_sym, z, R)
            ekf.update(h_sym, z=z, R=R)
        """
        if args and callable(args[0]):
            h_sym = args[0]
            try:
                z = args[1] if len(args) > 1 else kwargs["z"]
                R = args[2] if len(args) > 2 else kwargs["R"]
            except KeyError as e:
                raise TypeError(
                    f"NumpyEKF.update: low-level form needs h_sym + z + R; "
                    f"missing {e}") from None
            return self._update_low_level(h_sym, z, R)
        if len(args) == 1 and not callable(args[0]):
            return self._update_sensor(args[0], kwargs)
        raise TypeError(
            "NumpyEKF.update: pass (h_sym, z, R) for the low-level form "
            "or (part, **{output_name: z}) for the sensor form.")

    def _update_sensor(self, part, measurements: dict[str, Any]) -> None:
        if not measurements:
            raise ValueError(
                f"NumpyEKF.update: no measurements provided for "
                f"{type(part).__name__}('{part.name}').")
        ekf = self._ekf
        for out_name, z in measurements.items():
            key = (id(part), out_name)
            if key not in ekf._sensors:
                avail = [k[1] for k in ekf._sensors if k[0] == id(part)]
                raise KeyError(
                    f"NumpyEKF.update: part '{part.name}' has no "
                    f"registered output {out_name!r} (available: {avail}).")
            self._apply_sensor_update(ekf._sensors[key], z,
                                      ekf._u_defaults)

    def _apply_sensor_update(self, spec_o: dict[str, Any], z,
                             u_vec: np.ndarray) -> None:
        """Evaluate one sensor's cached h/H/R at the current state + `u_vec`
        and fold the measurement `z` in via the Joseph-form update."""
        ekf = self._ekf
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != spec_o["dim"]:
            raise ValueError(
                f"NumpyEKF: {spec_o['full']}: expected z of size "
                f"{spec_o['dim']}, got {z_arr.size}.")
        h_val = np.asarray(spec_o["h_fn"](self._x, u_vec, 0.0, 0.0)
                            ).reshape(-1)
        H     = np.asarray(spec_o["H_fn"](self._x, u_vec, 0.0, 0.0))
        if spec_o["L_h_fn"] is not None:
            L_h = np.asarray(spec_o["L_h_fn"](self._x, u_vec, 0.0, 0.0))
            R   = L_h @ ekf._Sigma @ L_h.T
        else:
            R   = np.zeros((spec_o["dim"], spec_o["dim"]))
        self._apply_update(z_arr, h_val, H, R)

    # ---- Measurement bus + step ----------------------------------------

    def _resolve_meas_name(self, name: str) -> str:
        """Resolve a full or craft-relative sensor name to its full key."""
        if name in self._meas:
            return name
        matches = [n for n in self._meas if n.endswith("." + name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"NumpyEKF.feed: ambiguous sensor name {name!r}; matches "
                f"{matches}. Use the fully-qualified form.")
        raise KeyError(
            f"NumpyEKF.feed: unknown sensor {name!r}. Registered: "
            f"{sorted(self._meas)}")

    def feed(self, name: str, z, *, t: float | None = None) -> None:
        """Drop a measurement into the bus and mark it fresh.

        `name` is a registered sensor's full `"craft.part.output"` name or
        an unambiguous suffix. `t` is an optional measurement timestamp
        used for staleness rejection in `step()`. The reading is consumed
        (and the fresh flag cleared) by the next `step()`.
        """
        m = self._meas[self._resolve_meas_name(name)]
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != m["dim"]:
            raise ValueError(
                f"NumpyEKF.feed: {name}: expected z of size {m['dim']}, "
                f"got {z_arr.size}.")
        m["value"] = z_arr
        m["fresh"] = True
        m["t"]     = t

    def step(self, dt: float, *, t: float | None = None) -> None:
        """Predict by `dt`, then fold in every fresh measurement.

        Latched `self.inputs` drive the predict. Fresh measurements are
        applied sequentially in registration order at a single
        linearization point (the post-predict state); their fresh flags
        are then cleared. A timestamped measurement older than the
        interval start is dropped as stale. Sub-`dt` measurement timing is
        quantized to the step boundary; full out-of-sequence handling is
        out of scope.
        """
        t_start = t if t is not None else self._t
        u_dict  = self.inputs if self.inputs else None
        self.predict(dt, t=t_start, u=u_dict)
        u_vec = self._ekf._build_u(u_dict)
        for full, m in self._meas.items():
            if not m["fresh"]:
                continue
            m["fresh"] = False
            if m["t"] is not None and m["t"] < t_start - 1e-9:
                continue  # stale: predates the interval we just integrated
            self._apply_sensor_update(self._ekf._sensors[m["key"]],
                                      m["value"], u_vec)
        self._t = t_start + dt

    def _update_low_level(self,
                          h_sym: Callable[[ca.MX], ca.MX],
                          z: np.ndarray,
                          R: np.ndarray) -> None:
        """Caller-supplied h(x) callable + R. Used for direct-slot
        measurements (`measurement_slot` helpers)."""
        ekf = self._ekf
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"NumpyEKF.update: R shape {R.shape} doesn't match z size {z.size}")

        h_mx = h_sym(ekf._x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta_h", ekf.spec.tangent_dim, 1)
        x_pert = ekf.spec.boxplus_sym(ekf._x_sym, delta)
        h_pert = h_sym(x_pert)
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(
            ca.jacobian(h_pert, delta),
            delta, ca.MX.zeros(ekf.spec.tangent_dim, 1))

        h_fn = ca.Function("h_low", [ekf._x_sym], [h_mx])
        H_fn = ca.Function("H_low", [ekf._x_sym], [H_sym])
        h_x  = np.asarray(h_fn(self._x)).reshape(-1)
        H    = np.asarray(H_fn(self._x))
        if h_x.size != z.size:
            raise ValueError(
                f"NumpyEKF.update: h(x) size {h_x.size} doesn't match z "
                f"size {z.size}")
        self._apply_update(z, h_x, H, R)

    def _apply_update(self,
                      z: np.ndarray,
                      h_x: np.ndarray,
                      H: np.ndarray,
                      R: np.ndarray) -> None:
        y = z - h_x
        S = H @ self._P @ H.T + R
        K = np.linalg.solve(S.T, (self._P @ H.T).T).T
        delta_x = K @ y
        self._x = self._ekf.spec.boxplus_num(self._x, delta_x)
        I = np.eye(self._ekf.spec.tangent_dim)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    def __repr__(self) -> str:
        return f"<NumpyEKF over {self._ekf!r}>"


# ---------------------------------------------------------------------------
# NumpyLQR — runtime for LQR
# ---------------------------------------------------------------------------

class NumpyLQR:
    """Native-Python evaluator wrapping an `LQR` IR.

    Stateless: the gain is baked at construction, so `control` is a pure
    function of the supplied state estimate."""

    def __init__(self, lqr) -> None:
        from ...control.lqr import LQR
        if not isinstance(lqr, LQR):
            raise TypeError(
                f"NumpyLQR: expected LQR, got {type(lqr).__name__}")
        self._lqr = lqr

    @property
    def K(self) -> np.ndarray:
        return self._lqr.K

    @property
    def spec(self):
        return self._lqr.spec

    @property
    def input_names(self) -> list:
        return self._lqr.input_names

    def u(self, x_flat) -> np.ndarray:
        """Control vector for a flat ambient state (spec layout)."""
        return np.asarray(
            self._lqr.control_fn(np.asarray(x_flat, dtype=float))).reshape(-1)

    def control(self, state: dict) -> dict[str, float]:
        """Map a state estimate → `{input_name: value}`.

        `state` is nested `{owner: {slot: value}}` (e.g.
        `NumpyEKF.state_dict()` or a `NumpyWorld` state) or flat
        `{"owner.slot": value}`. Slots not supplied fall back to the
        world's initial state.
        """
        lqr = self._lqr
        flat: dict[str, Any] = {}
        for owner_name, owner_state in lqr.world._initial_state_dict().items():
            for k, v in owner_state.items():
                flat[f"{owner_name}.{k}"] = v
        for k, v in state.items():
            if isinstance(v, dict):
                for slot, val in v.items():
                    flat[f"{k}.{slot}"] = val
            else:
                flat[k] = v
        x = lqr.spec.pack({k: v for k, v in flat.items() if k in lqr.spec})
        u_vec = self.u(x)
        return {name: float(u_vec[i])
                for i, name in enumerate(lqr.input_names)}

    def __repr__(self) -> str:
        return f"<NumpyLQR over {self._lqr!r}>"


# ---------------------------------------------------------------------------
# TargetNumpy factory
# ---------------------------------------------------------------------------

def TargetNumpy(ir):
    """Native-Python backend factory.

    Lowers a compiled-model IR to a Python-native runtime evaluator.
    Dispatches by IR type:

      * `Sim` → `NumpyWorld` (`.step()`, `.initial_state()`).
      * `EKF`           → `NumpyEKF`   (`.predict()`, `.update()`,
                                        `.state_dict()`, `.reset()`,
                                        `.x`, `.P`).
      * `LQR`           → `NumpyLQR`   (`.control()`, `.u()`, `.K`).
    """
    from ...sim import Sim
    from ...estimation.ekf import EKF
    from ...control.lqr import LQR
    if isinstance(ir, Sim):
        return NumpyWorld(ir)
    if isinstance(ir, EKF):
        return NumpyEKF(ir)
    if isinstance(ir, LQR):
        return NumpyLQR(ir)
    raise TypeError(
        f"TargetNumpy: no handler for IR type {type(ir).__name__}. "
        f"Expected Sim, EKF, or LQR.")
