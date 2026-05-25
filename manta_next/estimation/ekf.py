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
    """Error-state EKF wrapping a `World`.

    Compiles its own world tick (separate from the sim's `cw.tick`)
    using the same fields + planets + couplings. State spans every
    craft in the world (`StateSpec.from_world`).
    """

    def __init__(self, world) -> None:
        # Ensure planet disturbances are registered + PlanetState
        # initial values resolved (idempotent).
        if not world._planets_registered:
            for p in world._planets:
                p.register_disturbances(world)
            world._planets_registered = True
        world._resolve_planet_state_overrides()

        self.world  = world
        self.crafts = tuple(world.crafts)
        if not self.crafts:
            raise ValueError("EKF: world has no crafts.")
        self.spec   = StateSpec.from_world(world)

        # Each part belongs to exactly one craft. Cache the lookup so
        # `ekf.update(part, ...)` can route to the right state slice.
        self._craft_of_part: dict[int, "Craft"] = {}
        for craft in self.crafts:
            for part in craft.parts:
                self._craft_of_part[id(part)] = craft

        # Compile the est-side world tick using the world's registered
        # fields + couplings.
        from ..coupled_tick import compile_coupled_tick
        from ..fields import (
            CollisionField, FluidField, GravityField, MagField,
        )
        compiled_tick = compile_coupled_tick(
            list(self.crafts), list(world._couplings),
            gravity_field=world.get_field(GravityField),
            fluid_field=world.get_field(FluidField),
            mag_field=world.get_field(MagField),
            collision_field=world.get_field(CollisionField),
        )
        cf = compiled_tick.casadi_function

        # Walk the tick signature: collect Inputs and Noise channels.
        # Names are flat-prefixed `<owner>.<sub>` where `<owner>` is a
        # craft name or a field-disturbance name. The EKF's StateSpec
        # uses the same prefixed names — spec membership is the
        # "is this a state slot?" check.
        # Index field disturbances by name for noise-channel routing.
        dist_by_name: dict[str, Any] = {}
        from ..fields.base import Disturbance
        for field in world.fields:
            for dist in field._disturbances:
                if isinstance(dist, Disturbance):
                    dist_by_name[dist.name] = dist

        self._input_names: list[str] = []
        self._noise_specs: list[dict[str, Any]] = []
        for i in range(cf.n_in()):
            name = cf.name_in(i)
            if name in ("dt", "t") or name in self.spec:
                continue
            head, _, rest = name.partition(".")
            craft = next((c for c in self.crafts if c.name == head), None)
            dist  = dist_by_name.get(head)
            if craft is not None:
                # Per-craft Input or Noise.
                if "." not in rest:
                    raise RuntimeError(
                        f"EKF: tick input {name!r} not in spec and not "
                        f"in the `<craft>.<part>.<sub>` shape.")
                part_name, sub = rest.split(".", 1)
                part = next((p for p in craft.parts
                             if p.name == part_name), None)
                if part is None:
                    raise RuntimeError(
                        f"EKF: tick input {name!r}: unknown part "
                        f"{part_name!r} on craft {craft.name!r}.")
                if sub in part.input_declarations():
                    self._input_names.append(name)
                    continue
                self._classify_noise_input(
                    name, part, sub, owner_label=f"craft '{craft.name}'")
                continue
            if dist is not None:
                # Per-disturbance Noise (state slots already in spec).
                self._classify_noise_input(
                    name, dist, rest,
                    owner_label=f"disturbance '{dist.name}'")
                continue
            raise RuntimeError(
                f"EKF: tick input {name!r} doesn't match any craft or "
                f"registered disturbance.")

        # Input defaults: look up each `<craft>.<part>.<input>` on its
        # owning part instance.
        self._u_defaults = np.array(
            [self._lookup_input_default(n) for n in self._input_names],
            dtype=float)

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
        for craft in self.crafts:
            for part in craft.parts:
                outs = part.output_declarations()
                if not outs:
                    continue
                for out_name in outs:
                    full = f"{craft.name}.{part.name}.{out_name}"
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
                    if n_noise > 0:
                        L_h_sym = ca.substitute(
                            ca.jacobian(h_n_flat, n_sym), n_sym, zero_n)
                        L_h_fn = ca.Function(
                            f"Lh_{full}".replace(".", "_"),
                            [x_sym, u_sym, dt_sym, t_sym], [L_h_sym],
                            ["x", "u", "dt", "t"], ["L_h"])
                    else:
                        L_h_fn = None
                    h_fn = ca.Function(
                        f"h_{full}".replace(".", "_"),
                        [x_sym, u_sym, dt_sym, t_sym], [h_0_flat],
                        ["x", "u", "dt", "t"], ["h"])
                    H_fn = ca.Function(
                        f"H_{full}".replace(".", "_"),
                        [x_sym, u_sym, dt_sym, t_sym], [H_sym],
                        ["x", "u", "dt", "t"], ["H"])
                    # Key on (part identity, output name) — part-name
                    # collisions across crafts are possible.
                    self._sensors[(id(part), out_name)] = {
                        "dim":     h_dim,
                        "h_fn":    h_fn,
                        "H_fn":    H_fn,
                        "L_h_fn":  L_h_fn,
                        "part":    part,
                        "craft":   craft,
                    }

        # Mutable runtime state (`_x`, `_P`) lives on the backend
        # evaluator (`NumpyEKF`), not on the IR. The IR keeps the
        # symbolic functions + sensor table; backends instantiate
        # their own state from `world._initial_state_dict()`.

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

    def _classify_noise_input(self, full_name: str, owner, sub: str,
                              owner_label: str) -> None:
        """Append a noise-spec entry for `<owner>.<sub>` if it matches a
        white noise channel directly, or for `<owner>.<sub>_driver` if
        it matches an RW Noise channel via its driver-suffix.

        `owner` can be a Part or a Disturbance — both expose
        `noise_declarations()` with the same shape.
        """
        ndecls = owner.noise_declarations()
        if sub in ndecls and ndecls[sub].kind == "white":
            ndecl = ndecls[sub]
            dim   = 1 if ndecl.shape == "scalar" else 3
            sigma = float(getattr(owner, f"{sub}_sigma"))
            self._noise_specs.append({
                "owner": owner, "name": sub, "full": full_name,
                "dim": dim, "sigma": sigma,
            })
            return
        if sub.endswith("_driver"):
            bias_name = sub[: -len("_driver")]
            if bias_name in ndecls and ndecls[bias_name].kind == "rw":
                ndecl = ndecls[bias_name]
                dim   = 1 if ndecl.shape == "scalar" else 3
                sigma = float(getattr(owner, f"{bias_name}_sigma"))
                self._noise_specs.append({
                    "owner": owner, "name": sub, "full": full_name,
                    "dim": dim, "sigma": sigma,
                })
                return
        raise RuntimeError(
            f"EKF: tick input {full_name!r} on {owner_label} is neither "
            f"an Input nor a recognized Noise channel.")

    def _lookup_input_default(self, full_name: str) -> float:
        """Resolve `<craft>.<part>.<input>` to its part-instance value."""
        craft_name, rest = full_name.split(".", 1)
        part_name, input_name = rest.split(".", 1)
        craft = next(c for c in self.crafts if c.name == craft_name)
        part  = next(p for p in craft.parts if p.name == part_name)
        return float(getattr(part, input_name))

    def _build_u(self, u: dict[str, float] | None) -> np.ndarray:
        """Resolve `u` to a flat input vector in `_input_names` order.

        Accepts either full names (`"drone.t.throttle"`) or
        craft-relative shorthand (`"t.throttle"`) when the shorthand
        uniquely identifies one input across all crafts.
        """
        if not self._input_names:
            return np.zeros(0)
        if u is None:
            return self._u_defaults.copy()
        # Resolve each user-supplied key to a full input name.
        resolved: dict[str, float] = {}
        for user_key, value in u.items():
            if user_key in self._input_names:
                resolved[user_key] = float(value)
                continue
            candidates = [n for n in self._input_names
                          if n.endswith("." + user_key)]
            if len(candidates) == 1:
                resolved[candidates[0]] = float(value)
            elif len(candidates) > 1:
                raise KeyError(
                    f"EKF.predict: ambiguous input name {user_key!r}; "
                    f"matches {candidates}. Use the fully-qualified form.")
            else:
                raise KeyError(
                    f"EKF.predict: unknown input name {user_key!r}. "
                    f"Available: {sorted(self._input_names)}")
        out = self._u_defaults.copy()
        for i, name in enumerate(self._input_names):
            if name in resolved:
                out[i] = resolved[name]
        return out

    # Predict / update / state_dict / reset / x / P live on the
    # runtime evaluator (`NumpyEKF`), not the IR. The split keeps
    # EKF as a pure compile-time description that any backend
    # (numpy, C++, …) can consume. Build a runtime via:
    #     from manta_next import TargetNumpy
    #     ekf = TargetNumpy(EKF(world))
    #     ekf.predict(dt=dt, t=t)
    #     ekf.update(part, gyro=z)
    # The IR holds the symbolic functions (_f_fn, _F_fn, _L_fn,
    # _sensors) the backends need.



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
