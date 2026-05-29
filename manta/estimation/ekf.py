"""EKF — Error-state (manifold-aware) Extended Kalman Filter IR.

`EKF(world)` walks the world's crafts + fields, building:
  * a `StateSpec` (joint state across every craft + every state-bearing
    disturbance). `track=` carves a subset out of this — see below.
  * a `Linearization` over the world tick (`manta.linearization`), which
    produces the symbolic predict (`_f_fn`), tangent-space F (`_F_fn`),
    process-noise gain L (`_L_fn`), and per-output h/H/L_h — all CasADi
    functions. The EKF does the model introspection (which tick inputs
    are Inputs vs. Noise, which outputs are sensors) and the Kalman
    bookkeeping (Σ, state subsetting, sensor table); the Jacobian
    machinery is the shared `Linearization` transform (also the seam an
    LQR/iLQR would reuse).
  * Per-sensor measurement bundles: `_sensors[(id(part), output_name)]`
    holds the cached h/H/L_h ca.Function objects, the sensor dim,
    and a back-ref to the owning part + craft.

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

import casadi as ca
import numpy as np

from ..linearization import Linearization, slot_of_tan
from .state_spec import StateSpec, resolve_slotset


class EKF:
    """Error-state EKF wrapping a `World`.

    Compiles its own world tick (separate from the sim's `cw.tick`)
    using the same fields + planets + couplings. State spans every
    craft in the world (`StateSpec.from_world`).
    """

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

        # Each part belongs to exactly one craft. Cache the lookup so
        # `ekf.update(part, ...)` can route to the right state slice.
        self._craft_of_part: dict[int, "Craft"] = {}
        for craft in self.crafts:
            for part in craft.parts:
                self._craft_of_part[id(part)] = craft

        # Compile the est-side world tick using the world's registered
        # fields + couplings.
        from ..world_tick import compile_world_tick
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

        # Classify the tick's I/O against the model — Inputs (→ u), Noise
        # channels (→ process noise), and candidate sensors. Shared with
        # the C++ extractor; membership in the FULL spec is the "is this a
        # state slot?" check.
        from ..tick_signature import walk_tick_signature
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
        init_flat: dict[str, Any] = {}
        for owner_name, owner_state in world._initial_state_dict().items():
            for k, v in owner_state.items():
                init_flat[f"{owner_name}.{k}"] = v

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
            kept = self._closure(full_spec, full_lin.F_sym, seed)
            self.spec = StateSpec.subset(full_spec, kept)
            # Freeze every full-spec slot we are not keeping.
            for s in full_spec.slots:
                if s.name not in kept:
                    val = init_flat.get(s.name, np.zeros(s.dim))
                    frozen[s.name] = np.atleast_1d(
                        np.asarray(val, dtype=float)).reshape(-1)

        # --- Linearize over the (sub)spec → predict/F/L + sensors -------
        lin = Linearization(
            cf, self.spec, frozen=frozen,
            input_names=self._input_names, noise_specs=self._noise_specs,
            outputs=chosen_sensor_names, build_functions=True)
        self._lin     = lin
        self._x_sym   = lin.x_sym
        self._u_sym   = lin.u_sym
        self._n_sym   = lin.n_sym
        self._n_noise = lin.n_noise
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
    # State subsetting (dependency closure over the linearized dynamics)
    # ------------------------------------------------------------------

    def _closure(self, full_spec, F_sym, seed_slots) -> set[str]:
        """Backward reachability on the structural dependency graph.

        `F_sym[i, j] != 0` (structurally) means slot-row i's next value
        depends on slot-col j — so if i is kept, j must be too. Iterate
        from `seed_slots` to a fixpoint over the slot-level graph (slot
        granularity keeps SO3 orientation atomic).
        """
        sot = slot_of_tan(full_spec)
        pattern = np.array(ca.DM(F_sym.sparsity()))
        rows, cols = np.nonzero(pattern)
        deps: dict[str, set[str]] = {}
        for i, j in zip(rows.tolist(), cols.tolist()):
            a, b = sot[i], sot[j]
            if a != b:
                deps.setdefault(a, set()).add(b)
        kept = set(seed_slots)
        frontier = list(seed_slots)
        while frontier:
            s = frontier.pop()
            for dep in deps.get(s, ()):
                if dep not in kept:
                    kept.add(dep)
                    frontier.append(dep)
        return kept

    @staticmethod
    def _resolve_names(user_names, candidates, label: str) -> set[str]:
        """Resolve user-supplied names (full or unambiguous suffix) against
        a candidate list; raise on unknown/ambiguous."""
        resolved: set[str] = set()
        for key in user_names:
            if key in candidates:
                resolved.add(key)
                continue
            matches = [n for n in candidates if n.endswith("." + key)]
            if len(matches) == 1:
                resolved.add(matches[0])
            elif len(matches) > 1:
                raise KeyError(
                    f"EKF: ambiguous {label} name {key!r}; matches "
                    f"{matches}. Use the fully-qualified form.")
            else:
                raise KeyError(
                    f"EKF: unknown {label} name {key!r}. Available: "
                    f"{sorted(candidates)}")
        return resolved


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
    #     from manta import TargetNumpy
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
