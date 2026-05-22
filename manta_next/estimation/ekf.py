"""EKF — Error-state (manifold-aware) Extended Kalman Filter wrapping a World.

Design:

  * The user builds a World containing the est-side Craft plus any
    fields / planets the craft's parts query. `EKF(world)` compiles a
    predict + per-sensor measurement bundle out of that World; the
    EKF then ticks alongside the sim:

        cw  = sim_world.compile()
        ekf = EKF(est_world)
        for _ in range(N):
            state = cw.step(state, t=t, dt=dt)
            ekf.predict(t=t, dt=dt, u={"thrust.throttle": cmd})
            ekf.update(est_imu, gyro=measured_gyro, accel=measured_accel)
            ekf.update(est_gps, position=measured_pos)
            t += dt

  * Q is assembled automatically from every Noise channel that affects
    the next-tick state (via autodiff). The user can override with an
    explicit `Q=` argument to predict.

  * R is assembled automatically per sensor output from the Noise
    channels feeding that output. `ekf.update(part, **measurements)`
    routes each measurement through its sensor's cached (h_fn, H_fn,
    R_builder).

  * `predict` / `update` mutate the EKF in place; the user reads the
    current estimate via `ekf.state_dict()`, `ekf.x`, `ekf.P`.

  * Initial state seeds from `world.add_craft(c, position=..., ...)`.
    PlanetState wrappers in those overrides have already been resolved
    to WorldFrame seeds by `world.compile()` at EKF construction.

Scope notes:
  * Single craft only in v1 (raises if the world has more than one).
    Multi-craft StateSpec generalization extends this trivially later.
  * Random-walk bias channels (`Noise(kind="rw", ...)`) deferred to a
    follow-up; only white-Gaussian channels are wired today.
  * Couplings are not yet observed by the EKF — the World's compile()
    builds coupled ticks but the EKF compiles a per-craft tick.
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from .. import ir
from .state_spec import StateSpec


class EKF:
    """Error-state EKF wrapping a single-craft `World`."""

    def __init__(self, world) -> None:
        # Ensure planet disturbances are registered + PlanetState
        # initial values resolved.
        if not world._planets_registered:
            for p in world._planets:
                p.register_disturbances(world)
            world._planets_registered = True
        world._resolve_planet_state_overrides()

        if len(world.crafts) != 1:
            raise NotImplementedError(
                f"EKF: only single-craft worlds are supported in v1 "
                f"(world has {len(world.crafts)} craft(s)). Multi-craft "
                f"StateSpec lands in a follow-up.")

        self.world = world
        self.craft = world.crafts[0]
        self.spec  = StateSpec.from_craft(self.craft)

        # Compile the est-side tick using the world's registered fields.
        from ..fields import (
            CollisionField, FluidField, GravityField, MagField,
        )
        compiled_tick = self.craft.compile_tick(
            gravity_field=world.get_field(GravityField),
            fluid_field=world.get_field(FluidField),
            mag_field=world.get_field(MagField),
            collision_field=world.get_field(CollisionField),
        )
        cf = compiled_tick.casadi_function

        # Walk the tick signature: collect Inputs (ordered → flat u
        # vector) and Noise channels (ordered → flat n vector).
        # Noise channels include both `kind="white"` (raw `<part>.<n>`)
        # and `kind="rw"` drivers (`<part>.<n>_driver` — the RW bias
        # itself is a state slot listed by `self.spec`, not a noise).
        self._input_names: list[str] = []
        self._noise_specs: list[dict[str, Any]] = []
        for i in range(cf.n_in()):
            name = cf.name_in(i)
            if name in ("dt", "t") or name in self.spec:
                continue
            if "." not in name:
                raise RuntimeError(
                    f"EKF: unrecognized tick input {name!r}.")
            part_name, sub = name.split(".", 1)
            part = next((p for p in self.craft.parts
                         if p.name == part_name), None)
            if part is None:
                raise RuntimeError(
                    f"EKF: tick input {name!r} references unknown part.")
            if sub in part.input_declarations():
                self._input_names.append(name)
                continue
            ndecls = part.noise_declarations()
            if sub in ndecls and ndecls[sub].kind == "white":
                ndecl = ndecls[sub]
                dim   = 1 if ndecl.shape == "scalar" else 3
                sigma = float(getattr(part, f"{sub}_sigma"))
                self._noise_specs.append({
                    "part": part, "name": sub, "full": name,
                    "dim": dim, "sigma": sigma,
                })
                continue
            if sub.endswith("_driver"):
                bias_name = sub[: -len("_driver")]
                if bias_name in ndecls and ndecls[bias_name].kind == "rw":
                    ndecl = ndecls[bias_name]
                    dim   = 1 if ndecl.shape == "scalar" else 3
                    sigma = float(getattr(part, f"{bias_name}_sigma"))
                    self._noise_specs.append({
                        "part": part, "name": sub, "full": name,
                        "dim": dim, "sigma": sigma,
                    })
                    continue
            raise RuntimeError(
                f"EKF: tick input {name!r} is neither an Input "
                f"nor a recognized Noise channel.")

        self._u_defaults = np.array(
            [float(getattr(next(p for p in self.craft.parts
                                if p.name == n.split('.', 1)[0]),
                           n.split('.', 1)[1]))
             for n in self._input_names], dtype=float)

        n_ambient = self.spec.ambient_dim
        n_tangent = self.spec.tangent_dim
        n_u       = len(self._input_names)
        n_noise   = sum(s["dim"] for s in self._noise_specs)

        # Top-level MX symbols.
        x_sym  = ca.MX.sym("x",  n_ambient, 1)
        u_sym  = (ca.MX.sym("u", n_u, 1) if n_u > 0
                  else ca.MX.zeros(0, 1))
        dt_sym = ca.MX.sym("dt", 1, 1)
        t_sym  = ca.MX.sym("t",  1, 1)
        n_sym  = (ca.MX.sym("noise", n_noise, 1) if n_noise > 0
                  else ca.MX.zeros(0, 1))

        # Cache for reuse during update().
        self._x_sym   = x_sym
        self._n_sym   = n_sym
        self._u_sym   = u_sym
        self._n_noise = n_noise

        # Run the tick once symbolically; gather all outputs by name.
        outputs_n = self._tick_outputs(cf, x_sym, u_sym, dt_sym, t_sym,
                                       n_sym)

        # Build the new ambient state at noise=n_sym (used to derive L).
        x_new_n = self._gather_state(outputs_n)

        # Same tick with noise=0 — the nominal predict.
        zero_n = ca.MX.zeros(n_noise, 1)
        x_new_0 = ca.substitute(x_new_n, n_sym, zero_n)

        self._f_fn = ca.Function(
            "ekf_predict",
            [x_sym, u_sym, dt_sym, t_sym], [x_new_0],
            ["x", "u", "dt", "t"], ["x_new"])

        # F = ∂boxminus(f_pert, f_clean) / ∂δ at δ=0, noise=0.
        delta_in = ca.MX.sym("delta_in", n_tangent, 1)
        x_pert    = self.spec.boxplus_sym(x_sym, delta_in)
        outputs_pert = self._tick_outputs(cf, x_pert, u_sym, dt_sym,
                                           t_sym, zero_n)
        x_pert_new = self._gather_state(outputs_pert)
        delta_out = self.spec.boxminus_sym(x_pert_new, x_new_0)
        F_sym = ca.substitute(
            ca.jacobian(delta_out, delta_in), delta_in,
            ca.MX.zeros(n_tangent, 1))
        self._F_fn = ca.Function(
            "ekf_F",
            [x_sym, u_sym, dt_sym, t_sym], [F_sym],
            ["x", "u", "dt", "t"], ["F"])

        # L_x = ∂boxminus(f_n, f_0) / ∂n at n=0 — process-noise gain.
        if n_noise > 0:
            delta_out_n = self.spec.boxminus_sym(x_new_n, x_new_0)
            L_sym = ca.substitute(
                ca.jacobian(delta_out_n, n_sym), n_sym, zero_n)
            self._L_fn = ca.Function(
                "ekf_L",
                [x_sym, u_sym, dt_sym, t_sym], [L_sym],
                ["x", "u", "dt", "t"], ["L"])
            sigmas_sq = []
            for ns in self._noise_specs:
                sigmas_sq.extend([ns["sigma"] ** 2] * ns["dim"])
            self._Sigma = np.diag(sigmas_sq)
        else:
            self._L_fn = None
            self._Sigma = None

        # Per-(part, output) measurement plumbing.
        # For every Part with at least one Output, cache:
        #   * h_fn(x, u) → output value (with noise=0).
        #   * H_fn(x, u) → ∂h/∂δ at δ=0, noise=0.
        #   * L_h_fn(x, u) → ∂h/∂n at noise=0 (if any noise feeds this output).
        # R is built from L_h_fn @ Σ @ L_h_fnᵀ at update time.
        self._sensors: dict[tuple[str, str], dict[str, Any]] = {}
        for part in self.craft.parts:
            outs = part.output_declarations()
            if not outs:
                continue
            for out_name in outs:
                full = f"{part.name}.{out_name}"
                if full not in outputs_n:
                    raise RuntimeError(
                        f"EKF: tick is missing output {full!r}.")
                h_n_mx = outputs_n[full]
                h_dim  = int(h_n_mx.numel())
                h_n_flat = ca.reshape(h_n_mx, h_dim, 1)
                # h at noise=0.
                h_0_flat = ca.substitute(h_n_flat, n_sym, zero_n)
                # H at noise=0, δ=0.
                h_pert_mx   = outputs_pert[full]
                h_pert_flat = ca.reshape(h_pert_mx, h_dim, 1)
                H_sym = ca.substitute(
                    ca.jacobian(h_pert_flat, delta_in),
                    delta_in, ca.MX.zeros(n_tangent, 1))
                # L_h at noise=0.
                if n_noise > 0:
                    L_h_sym = ca.substitute(
                        ca.jacobian(h_n_flat, n_sym), n_sym, zero_n)
                    L_h_fn = ca.Function(
                        f"Lh_{full}".replace(".", "_"),
                        [x_sym, u_sym, dt_sym, t_sym], [L_h_sym],
                        ["x", "u", "dt", "t"], ["L_h"])
                else:
                    L_h_fn = None
                # Sensor h / H may symbolically depend on dt and t even
                # if numerically constant in those args (the compiled
                # tick threads them through). Declare with all four
                # inputs; the EKF passes a sentinel dt/t = 0 at update
                # time since measurements are instantaneous.
                h_fn = ca.Function(
                    f"h_{full}".replace(".", "_"),
                    [x_sym, u_sym, dt_sym, t_sym], [h_0_flat],
                    ["x", "u", "dt", "t"], ["h"])
                H_fn = ca.Function(
                    f"H_{full}".replace(".", "_"),
                    [x_sym, u_sym, dt_sym, t_sym], [H_sym],
                    ["x", "u", "dt", "t"], ["H"])
                self._sensors[(part.name, out_name)] = {
                    "dim":     h_dim,
                    "h_fn":    h_fn,
                    "H_fn":    H_fn,
                    "L_h_fn":  L_h_fn,
                    "part":    part,
                }

        # Initial state + covariance.
        entry = world._crafts[0]
        init = self.craft.initial_state(**entry["initial_state_overrides"])
        self._x = self.spec.pack(init)
        self._P = np.eye(n_tangent) * 1e-2

    # ------------------------------------------------------------------
    # Symbolic-tick helpers (build the new-state + outputs MX dict)
    # ------------------------------------------------------------------

    def _tick_outputs(self,
                      cf: ca.Function,
                      x_sym: ca.MX,
                      u_sym: ca.MX,
                      dt_sym: ca.MX,
                      t_sym: ca.MX,
                      n_sym: ca.MX) -> dict[str, ca.MX]:
        """Evaluate the compiled tick on flat symbolic inputs; return a
        dict mapping every output name (state slots + sensor outputs)
        to its MX expression."""
        in_names  = [cf.name_in(i)  for i in range(cf.n_in())]
        out_names = [cf.name_out(i) for i in range(cf.n_out())]
        u_index   = {name: i for i, name in enumerate(self._input_names)}
        noise_offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for ns in self._noise_specs:
            noise_offsets[ns["full"]] = (off, ns["dim"])
            off += ns["dim"]

        sliced: list[ca.MX] = []
        for name in in_names:
            if name == "dt":
                sliced.append(dt_sym)
            elif name == "t":
                sliced.append(t_sym)
            elif name in self.spec:
                slot = self.spec.slot(name)
                sliced.append(x_sym[slot.offset : slot.offset + slot.dim])
            elif name in u_index:
                sliced.append(u_sym[u_index[name]])
            elif name in noise_offsets:
                start, dim = noise_offsets[name]
                sliced.append(n_sym[start : start + dim])
            else:
                raise RuntimeError(f"EKF: tick input {name!r} not handled.")

        result = cf(*sliced)
        if len(out_names) == 1:
            return {out_names[0]: result}
        return {n: result[i] for i, n in enumerate(out_names)}

    def _gather_state(self, outputs_by_name: dict[str, ca.MX]) -> ca.MX:
        """Concatenate state-slot outputs in spec order → ambient vector."""
        chunks = []
        for slot in self.spec.slots:
            r = outputs_by_name[slot.name]
            if r.shape != (slot.dim, 1):
                r = ca.reshape(r, slot.dim, 1)
            chunks.append(r)
        return ca.vertcat(*chunks)

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

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    def state_dict(self) -> dict[str, Any]:
        return self.spec.unpack(self._x)

    def reset(self, *,
              state: dict | None = None,
              P: np.ndarray | None = None) -> None:
        if state is not None:
            full = self.craft.initial_state()
            full.update(state)
            self._x = self.spec.pack(full)
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (self.spec.tangent_dim, self.spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"EKF.reset: P shape {P.shape} doesn't match "
                    f"tangent dim {expected}")
            self._P = P.copy()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self,
                dt: float,
                *,
                t: float = 0.0,
                u: dict[str, float] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the nominal state and tangent covariance by `dt`.

        Args:
            dt    integrator timestep.
            t     world-clock time at the start of this step (defaults
                  to 0; matters when planet-attached disturbances are
                  present).
            u     dict of `"<part>.<input>"` → float for per-tick input
                  values. Missing entries fall back to the part-instance
                  default captured at construction time.
            Q     process-noise covariance, tangent-dim square. If
                  None (default), Q is assembled from registered Noise
                  channels via `L · Σ · Lᵀ` (autodiff).
        """
        u_vec = self._build_u(u)
        x_new = np.asarray(self._f_fn(self._x, u_vec, dt, t)).reshape(-1)
        F     = np.asarray(self._F_fn(self._x, u_vec, dt, t))
        n     = self.spec.tangent_dim
        if Q is None:
            if self._L_fn is not None:
                L = np.asarray(self._L_fn(self._x, u_vec, dt, t))
                Q = L @ self._Sigma @ L.T
            else:
                Q = np.zeros((n, n))
        self._x = x_new
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, *args, **kwargs) -> None:
        """Polymorphic update.

        Sensor form (auto h, H, R):
            ekf.update(part, gyro=z_gyro, accel=z_accel)

        Low-level form (caller-supplied h, R):
            ekf.update(h_sym, z, R)
            ekf.update(h_sym, z=z, R=R)
        """
        # Low-level: first arg is a callable h_sym; z and R may be
        # positional or keyword.
        if args and callable(args[0]):
            h_sym = args[0]
            try:
                z = args[1] if len(args) > 1 else kwargs["z"]
                R = args[2] if len(args) > 2 else kwargs["R"]
            except KeyError as e:
                raise TypeError(
                    f"EKF.update: low-level form needs h_sym + z + R; "
                    f"missing {e}") from None
            return self._update_low_level(h_sym, z, R)
        # Sensor form: first arg is a Part.
        if len(args) == 1 and not callable(args[0]):
            return self._update_sensor(args[0], kwargs)
        raise TypeError(
            "EKF.update: pass (h_sym, z, R) for the low-level form or "
            "(part, **{output_name: z}) for the sensor form.")

    def _update_sensor(self, part, measurements: dict[str, Any]) -> None:
        if not measurements:
            raise ValueError(
                f"EKF.update: no measurements provided for "
                f"{type(part).__name__}('{part.name}').")
        u_vec = self._u_defaults
        # Measurements are instantaneous; pass sentinel dt/t = 0 to
        # any h / H / L_h evaluation that may symbolically depend on
        # them (CasADi can't always prove independence statically).
        dt0, t0 = 0.0, 0.0
        for out_name, z in measurements.items():
            key = (part.name, out_name)
            if key not in self._sensors:
                raise KeyError(
                    f"EKF.update: part '{part.name}' has no registered "
                    f"output {out_name!r} (available: "
                    f"{[k[1] for k in self._sensors if k[0] == part.name]}).")
            spec_o = self._sensors[key]
            z_arr  = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
            if z_arr.size != spec_o["dim"]:
                raise ValueError(
                    f"EKF.update: {part.name}.{out_name}: expected "
                    f"z of size {spec_o['dim']}, got {z_arr.size}.")
            h_val = np.asarray(spec_o["h_fn"](self._x, u_vec, dt0, t0)
                                ).reshape(-1)
            H     = np.asarray(spec_o["H_fn"](self._x, u_vec, dt0, t0))
            if spec_o["L_h_fn"] is not None:
                L_h = np.asarray(spec_o["L_h_fn"](self._x, u_vec, dt0, t0))
                R   = L_h @ self._Sigma @ L_h.T
            else:
                R   = np.zeros((spec_o["dim"], spec_o["dim"]))
            self._apply_update(z_arr, h_val, H, R)

    def _update_low_level(self,
                          h_sym: Callable[[ca.MX], ca.MX],
                          z: np.ndarray,
                          R: np.ndarray) -> None:
        """Caller-supplied h(x) callable and R. Used by tests that
        synthesize bespoke measurements (e.g., observe a single state
        component directly)."""
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"EKF.update: R shape {R.shape} doesn't match z size {z.size}")

        h_mx = h_sym(self._x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta_h", self.spec.tangent_dim, 1)
        x_pert = self.spec.boxplus_sym(self._x_sym, delta)
        h_pert = h_sym(x_pert)
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(
            ca.jacobian(h_pert, delta),
            delta, ca.MX.zeros(self.spec.tangent_dim, 1))

        h_fn = ca.Function("h_low", [self._x_sym], [h_mx])
        H_fn = ca.Function("H_low", [self._x_sym], [H_sym])
        h_x  = np.asarray(h_fn(self._x)).reshape(-1)
        H    = np.asarray(H_fn(self._x))
        if h_x.size != z.size:
            raise ValueError(
                f"EKF.update: h(x) size {h_x.size} doesn't match z size {z.size}")
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
        self._x = self.spec.boxplus(self._x, delta_x)
        I = np.eye(self.spec.tangent_dim)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)


# ---------------------------------------------------------------------------
# Low-level h_sym helpers (kept for tests / advanced custom measurements)
# ---------------------------------------------------------------------------

def measurement_slot(spec: StateSpec, name: str):
    """Return an h_sym callable that reads slot `name` directly from x."""
    slot = spec.slot(name)
    def h_sym(x):
        return x[slot.offset : slot.offset + slot.dim]
    return h_sym


def measurement_component(spec: StateSpec, name: str, component: int):
    """Return an h_sym callable for a single component of a slot."""
    slot = spec.slot(name)
    if not (0 <= component < slot.dim):
        raise IndexError(
            f"measurement_component: slot {name!r} has dim {slot.dim}, "
            f"component {component} out of range")
    def h_sym(x):
        return x[slot.offset + component]
    return h_sym
