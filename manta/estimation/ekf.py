"""EKF — Error-state (manifold-aware) Extended Kalman Filter IR.

`EKF(world)` walks the world's crafts + fields, building:
  * a `StateSpec` (joint state across every craft + every state-bearing
    disturbance). `track=` carves a subset out of this — see below.
  * a `Linearization` over the world tick (`manta.linearization`), which
    produces the tangent-space `F`, process-noise gain `L`, and per-output
    `h`/`H`/`L_h` (the Jacobian machinery — also the seam LQR/iLQR reuse).
  * **the full Kalman recursion as fused `ca.Function`s** via
    `Linearization.kalman_functions`: a predict step `_predict_fn`
    `(x,P,Q,u,dt,t) → (x',P')`, a process-noise kernel `_process_noise_fn`
    `(x,u,dt,t) → Q = L Σ Lᵀ`, and a per-sensor Joseph update `_update_fns`
    `(x,P,z,u,t) → (x',P')`. The linear algebra lives HERE, once — both
    backends (numpy + emitted C++) just *evaluate* these kernels; neither
    reimplements the predict / Joseph update. The EKF still does the model
    introspection (Inputs vs Noise, which outputs are sensors) and exposes
    the building-block Jacobians for analysis (`observability`).
  * Per-sensor measurement bundles: `_sensors[(id(part), output_name)]`
    holds the sensor's `update_fn` (+ the h/H/L_h Jacobians for analysis),
    the sensor dim, and a back-ref to the owning part + craft.

The result is an IR — a description of the symbolic predict/measurement
graph. Lower to a backend to actually run::

    from manta import TargetNumpy, EKF

    cw  = TargetNumpy(Sim(sim_world))
    ekf = TargetNumpy(EKF(est_world))
    for _ in range(N):
        state = cw.step(state, t=t, dt=dt)
        ekf.predict(t=t, dt=dt, u={"thrust.throttle": cmd})
        ekf.update(est_imu, gyro=measured_gyro, accel=measured_accel)
        ekf.update(est_gps, position=measured_pos)
        t += dt

State subsetting + measurement bus (the friendlier path)::

    from manta import EKF, TargetNumpy, POSE, TWIST

    # Estimate only craft "chaser"; ignore the rest. Use just these
    # sensors; treat the thruster command as a known input.
    ekf = TargetNumpy(EKF(world,
                          track={"chaser": POSE | TWIST},
                          sensors=["chaser.imu.gyro", "chaser.gps.position"],
                          inputs=["chaser.thruster.throttle"]))
    for _ in range(N):
        ekf.inputs["chaser.thruster.throttle"] = cmd      # latched (ZOH)
        ekf.feed("chaser.imu.gyro", gyro_z, t=t)          # 400 Hz
        if new_fix:
            ekf.feed("chaser.gps.position", pos_z, t=t)   # 1 Hz
        ekf.step(dt, t=t)            # predict + fold in fresh measurements

`track` is a *lower bound*: the framework expands it (and any slots the
chosen sensors observe) to a set closed under the dynamics — via the
structural sparsity of F — and freezes the rest at their initial values.
A fully independent craft you don't track drops out of the O(n³) predict
entirely; a craft you're coupled to is pulled back in automatically.

Auto-assembly contracts:

  * Q is built from every Noise channel that affects the next-tick
    state (via autodiff). The runtime can override with an explicit
    `Q=` argument to predict.

  * R is built per sensor output from the Noise channels feeding that
    output. `ekf.update(part, **measurements)` routes each measurement
    through its sensor's cached (h_fn, H_fn, R_builder).

  * Initial state seeds from `world._initial_state_dict()` (already
    PlanetState-resolved by `Sim(world)`). The runtime instantiates
    `_x` and `_P` from this seed.

Scope notes:
  * Couplings between crafts are honored by the world-tick (which the
    EKF reuses) — F propagates through them automatically.
  * RW bias channels are first-class (`RandomWalkNoise` declarations on
    Parts or Disturbances synthesize a state slot + a driver input with
    `bias_next = bias + sqrt(dt)·driver`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..linearization import Linearization
from ..linearized_world import flatten_nested, freeze_complement, resolve_suffix
from .state_spec import StateSpec, resolve_slotset


class EKF:
    """Error-state EKF wrapping a `World`.

    Compiles its own world tick (separate from the sim's `cw.tick`)
    using the same fields + planets + couplings. State spans every
    craft in the world (`StateSpec.from_world`).
    """

    # Lowerable-block kind (see manta.codegen.block.KIND_EKF): a backend
    # lowers an EKF through its `lower_ekf` handler.
    RUNTIME_KIND = "ekf"

    def __init__(self, world, *,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None) -> None:
        """Build the EKF IR over `world`.

        Args:
            track:   `{craft_name: SlotSet}` — the *lower bound* of what
                     to estimate per craft. The framework expands this
                     (and any slots the chosen sensors observe) to a set
                     closed under the dynamics, freezing the rest at
                     their initial values. `None` (default) keeps the
                     full state for every craft — the legacy behavior.
            sensors: full `"craft.part.output"` names (or unambiguous
                     suffixes) the EKF may use as measurements. `None`
                     keeps every Part output as a sensor.
            inputs:  full `"craft.part.input"` names the EKF is aware of.
                     `None` keeps every Part input; excluded inputs are
                     frozen at their default.
        """
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

        # Full state layout across every craft + disturbance. `self.spec`
        # ends up either this (track=None) or a `StateSpec.subset` of it.
        full_spec = StateSpec.from_world(world)

        # Compile the est-side world tick using the world's registered
        # fields + couplings.
        from ..tick import compile_world_tick
        from ..fields import (
            CollisionField, FluidField, GravityField, MagField,
        )
        compiled_tick = compile_world_tick(
            list(self.crafts), list(world._couplings),
            gravity_field=world.get_field(GravityField),
            fluid_field=world.get_field(FluidField),
            mag_field=world.get_field(MagField),
            collision_field=world.get_field(CollisionField),
        )
        cf = compiled_tick.casadi_function
        # Per-input rate declarations (ctx.hold) — the runtime gates its
        # command ports so predict sees the same ZOH command as truth.
        self._sample_rates = getattr(compiled_tick, "sample_rates", {})

        # Classify the tick's I/O against the model — Inputs (→ u), Noise
        # channels (→ process noise), and candidate sensors. Shared with
        # the C++ extractor; membership in the FULL spec is the "is this a
        # state slot?" check.
        from ..tick import walk_tick_signature
        sig = walk_tick_signature(cf, world, full_spec)
        self._noise_specs = sig.noise
        all_input_defaults = sig.input_defaults

        # `frozen` collects every tick input the symbolic graph should
        # treat as a baked constant rather than a live variable: excluded
        # Inputs first, then any frozen (untracked) state slots once the
        # closure runs.
        frozen: dict[str, Any] = {}
        if inputs is None:
            self._input_names = sig.input_names
        else:
            chosen_inputs = self._resolve_names(
                inputs, sig.input_names, "input")
            for n in sig.input_names:
                if n not in chosen_inputs:
                    frozen[n] = all_input_defaults[n]
            self._input_names = [n for n in sig.input_names
                                 if n in chosen_inputs]
        self._u_defaults = np.array(
            [all_input_defaults[n] for n in self._input_names], dtype=float)

        # Candidate sensors = every Part Output. Default keeps all.
        all_sensor_names = sig.sensor_names
        sensor_lookup = {s.full: (s.part, s.craft, s.output_name)
                         for s in sig.sensors}
        if sensors is None:
            if track is None:
                chosen_sensor_names = list(all_sensor_names)
            else:
                # Default to sensors on tracked crafts only — otherwise an
                # untracked craft's sensor would auto-expand the state and
                # pull that craft back in. Bring in another craft's
                # measurement (relative nav) by listing it explicitly.
                chosen_sensor_names = [n for n in all_sensor_names
                                       if n.split(".", 1)[0] in track]
        else:
            chosen = self._resolve_names(sensors, all_sensor_names, "sensor")
            chosen_sensor_names = [n for n in all_sensor_names if n in chosen]

        # Flat initial-state values (source of frozen-slot constants).
        init_flat = flatten_nested(world._initial_state_dict())

        # --- State subsetting via dependency closure --------------------
        if track is None:
            self.spec = full_spec
        else:
            craft_names = {c.name for c in self.crafts}
            unknown = set(track) - craft_names
            if unknown:
                raise KeyError(
                    f"EKF: track references unknown craft(s) "
                    f"{sorted(unknown)}; world has {sorted(craft_names)}.")
            # Linearize the FULL state (structure only) to read F's
            # dependency pattern + each chosen sensor's observed slots.
            full_lin = Linearization(
                cf, full_spec, frozen=dict(frozen),
                input_names=self._input_names, noise_specs=self._noise_specs,
                outputs=chosen_sensor_names, build_functions=False)
            seed: set[str] = set(full_lin.observed_slots)
            for craft_name, slotset in track.items():
                seed |= resolve_slotset(craft_name, slotset)
            kept = full_lin.dependency_closure(seed)
            self.spec = StateSpec.subset(full_spec, kept)
            freeze_complement(full_spec, kept, init_flat, into=frozen)

        # --- Linearize over the (sub)spec → predict/F/L + sensors -------
        lin = Linearization(
            cf, self.spec, frozen=frozen,
            input_names=self._input_names, noise_specs=self._noise_specs,
            outputs=chosen_sensor_names, build_functions=True)
        self._x_sym   = lin.x_sym
        self._f_fn    = lin.predict_fn
        self._F_fn    = lin.F_fn
        self._L_fn    = lin.L_fn
        self._Sigma   = lin.Sigma
        # Independent-subsystem partition of the tangent state. With one
        # block the predict is the usual dense propagation; with several
        # (e.g. uncoupled crafts estimated jointly) each propagates on its
        # own, turning the O(n³) covariance step into Σ O(n_b³).
        self._blocks  = lin.blocks
        # Wrap each output's linearization with its part/craft metadata,
        # keyed for `ekf.update(part, **measurements)` routing.
        self._sensors: dict[tuple[int, str], dict[str, Any]] = {}
        for full in chosen_sensor_names:
            part, craft, out_name = sensor_lookup[full]
            o = lin.outputs[full]
            self._sensors[(id(part), out_name)] = {
                "dim":    o.dim,
                "h_fn":   o.h_fn,
                "H_fn":   o.H_fn,
                "L_h_fn": o.L_h_fn,
                "part":   part,
                "craft":  craft,
                "full":   full,
            }

        # The full Kalman recursion, symbolic + once (see
        # `Linearization.kalman_functions`): predict + per-sensor Joseph
        # update as fused ca.Functions. Both backends EVALUATE these —
        # neither reimplements the linear algebra. `_update_fns` is keyed by
        # the sensor's full name; pair it with `_sensors` for routing.
        self._predict_fn, self._process_noise_fn, _upd = lin.kalman_functions()
        self._update_fns = _upd
        for key, o in self._sensors.items():
            o["update_fn"] = _upd[o["full"]]

        # Mutable runtime state (`_x`, `_P`) lives on the backend
        # evaluator (`NumpyEKF`), not on the IR. The IR keeps the
        # symbolic functions + sensor table; backends instantiate
        # their own state from `world._initial_state_dict()`.

    @property
    def n_blocks(self) -> int:
        """Number of independent subsystems in the tangent state. >1 means
        the predict propagates each block separately (see `_blocks`)."""
        return len(self._blocks)

    # ------------------------------------------------------------------
    # Name resolution / input vector (see manta.linearized_world)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_names(user_names, candidates, label: str) -> set[str]:
        """Resolve user-supplied names (full or unambiguous suffix) against
        a candidate list; raise on unknown/ambiguous."""
        return {resolve_suffix(k, candidates, label=label, who="EKF")
                for k in user_names}

    def _build_u(self, u: dict[str, float] | None) -> np.ndarray:
        """Resolve `u` to a flat input vector in `_input_names` order.

        Accepts either full names (`"drone.t.throttle"`) or craft-relative
        shorthand (`"t.throttle"`) when the shorthand uniquely identifies
        one input across all crafts.
        """
        if not self._input_names:
            return np.zeros(0)
        if u is None:
            return self._u_defaults.copy()
        out = self._u_defaults.copy()
        index = {n: i for i, n in enumerate(self._input_names)}
        for user_key, value in u.items():
            full = resolve_suffix(user_key, self._input_names,
                                  label="input", who="EKF.predict")
            out[index[full]] = float(value)
        return out

    # Predict / update / state_dict / reset / x / P live on the
    # runtime evaluator (`NumpyEKF`), not the IR. The split keeps
    # EKF as a pure compile-time description that any backend
    # (numpy, C++, …) can consume. Build a runtime via:
    #     from manta import TargetNumpy
    #     ekf = TargetNumpy(EKF(world))
    #     ekf.predict(dt=dt, t=t)
    #     ekf.update(part, gyro=z)
    # The IR holds the baked Kalman kernels (_predict_fn,
    # _process_noise_fn, _update_fns) the backends evaluate.



# ---------------------------------------------------------------------------
# Low-level h_sym helpers (kept for tests / advanced custom measurements)
# ---------------------------------------------------------------------------

def measurement_slot(spec: StateSpec, name: str):
    """Return an h_sym callable that reads slot `name` directly from x."""
    slot = spec.slot(name)
    def h_sym(x):
        return x[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
    return h_sym


def measurement_component(spec: StateSpec, name: str, component: int):
    """Return an h_sym callable for a single component of a slot."""
    slot = spec.slot(name)
    if not (0 <= component < slot.ambient_dim):
        raise IndexError(
            f"measurement_component: slot {name!r} has dim {slot.ambient_dim}, "
            f"component {component} out of range")
    def h_sym(x):
        return x[slot.ambient_offset + component]
    return h_sym
