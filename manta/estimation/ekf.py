"""EKF — Error-state (manifold-aware) Extended Kalman Filter IR.

`EKF(world)` walks the world's crafts + fields, building:
  * a `StateSpec` (joint state across every craft + every state-bearing
    disturbance). `track=` carves a subset out of this — see below.
  * Symbolic predict (`_f_fn`), tangent-space F (`_F_fn`), and
    process-noise gain L (`_L_fn`) — all CasADi functions.
  * Per-sensor measurement bundles: `_sensors[(id(part), output_name)]`
    holds the cached h/H/L_h ca.Function objects, the sensor dim,
    and a back-ref to the owning part + craft.

The result is an IR — a description of the symbolic predict/measurement
graph. Lower to a backend to actually run::

    from manta import TargetNumpy, EKF

    cw  = TargetNumpy(sim_world.compile())
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
    PlanetState-resolved by `world.compile()`). The runtime instantiates
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

        # Walk the tick signature: collect Inputs and Noise channels.
        # Names are flat-prefixed `<owner>.<sub>` where `<owner>` is a
        # craft name or a field-disturbance name. Membership in the FULL
        # spec is the "is this a state slot?" check.
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
            if name in ("dt", "t") or name in full_spec:
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

        # Defaults for every candidate input. `frozen` collects every
        # tick input the symbolic graph should treat as a baked constant
        # rather than a live variable: excluded Inputs first, then any
        # frozen (untracked) state slots once the closure runs.
        all_input_defaults = {n: self._lookup_input_default(n)
                              for n in self._input_names}
        frozen: dict[str, Any] = {}
        if inputs is not None:
            chosen_inputs = self._resolve_names(
                inputs, self._input_names, "input")
            for n in self._input_names:
                if n not in chosen_inputs:
                    frozen[n] = all_input_defaults[n]
            self._input_names = [n for n in self._input_names
                                 if n in chosen_inputs]
        self._u_defaults = np.array(
            [all_input_defaults[n] for n in self._input_names], dtype=float)

        # Candidate sensors = every (part, output). Default keeps all.
        all_sensor_names: list[str] = []
        sensor_lookup: dict[str, tuple] = {}
        for craft in self.crafts:
            for part in craft.parts:
                for out_name in part.output_declarations():
                    full = f"{craft.name}.{part.name}.{out_name}"
                    all_sensor_names.append(full)
                    sensor_lookup[full] = (part, craft, out_name)
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
            # Build the FULL symbolic graph (sparsity only) to read F's
            # structural dependency pattern + each chosen sensor's
            # observed slots.
            full_g = self._build_symbolic(
                cf, full_spec, dict(frozen),
                chosen_sensor_names, sensor_lookup, build_functions=False)
            seed: set[str] = set(full_g["observed_slots"])
            for craft_name, slotset in track.items():
                seed |= resolve_slotset(craft_name, slotset)
            kept = self._closure(full_spec, full_g["F_sym"], seed)
            self.spec = StateSpec.subset(full_spec, kept)
            # Freeze every full-spec slot we are not keeping.
            for s in full_spec.slots:
                if s.name not in kept:
                    val = init_flat.get(s.name, np.zeros(s.dim))
                    frozen[s.name] = np.atleast_1d(
                        np.asarray(val, dtype=float)).reshape(-1)

        # --- Final symbolic build over the (sub)spec --------------------
        g = self._build_symbolic(cf, self.spec, frozen,
                                 chosen_sensor_names, sensor_lookup,
                                 build_functions=True)
        self._x_sym   = g["x_sym"]
        self._u_sym   = g["u_sym"]
        self._n_sym   = g["n_sym"]
        self._n_noise = g["n_noise"]
        self._f_fn    = g["f_fn"]
        self._F_fn    = g["F_fn"]
        self._L_fn    = g["L_fn"]
        self._Sigma   = g["Sigma"]
        self._sensors = g["sensors"]
        # Independent-subsystem partition of the tangent state. With one
        # block the predict is the usual dense propagation; with several
        # (e.g. uncoupled crafts estimated jointly) each propagates on its
        # own, turning the O(n³) covariance step into Σ O(n_b³).
        self._blocks  = g["blocks"]

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
    # Symbolic graph construction
    # ------------------------------------------------------------------

    def _build_symbolic(self, cf, spec, frozen, chosen_sensor_names,
                        sensor_lookup, *, build_functions: bool) -> dict:
        """Build the predict / F / L / sensor graph over `spec`.

        `frozen` maps tick-input names (untracked state slots + excluded
        inputs) to constant values that are baked into the graph instead
        of being live variables — so `f`/`F`/`L`/`h`/`H` are born reduced
        to the kept dimensions, no post-hoc row/column deletion.

        With `build_functions=False` only the structural artifacts needed
        for the closure are produced (`F_sym` + each chosen sensor's set
        of `observed_slots`); the `ca.Function`s are skipped.
        """
        n_ambient = spec.ambient_dim
        n_tangent = spec.tangent_dim
        n_u       = len(self._input_names)
        n_noise   = sum(s["dim"] for s in self._noise_specs)

        x_sym  = ca.MX.sym("x",  n_ambient, 1)
        u_sym  = ca.MX.sym("u", n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
        dt_sym = ca.MX.sym("dt", 1, 1)
        t_sym  = ca.MX.sym("t",  1, 1)
        n_sym  = (ca.MX.sym("noise", n_noise, 1) if n_noise > 0
                  else ca.MX.zeros(0, 1))
        zero_n = ca.MX.zeros(n_noise, 1)

        outputs_n = self._tick_outputs(cf, spec, x_sym, frozen,
                                       u_sym, dt_sym, t_sym, n_sym)
        x_new_n = self._gather_state(spec, outputs_n)
        x_new_0 = ca.substitute(x_new_n, n_sym, zero_n)

        delta_in = ca.MX.sym("delta_in", n_tangent, 1)
        x_pert   = spec.boxplus_sym(x_sym, delta_in)
        outputs_pert = self._tick_outputs(cf, spec, x_pert, frozen,
                                          u_sym, dt_sym, t_sym, zero_n)
        x_pert_new = self._gather_state(spec, outputs_pert)
        delta_out  = spec.boxminus_sym(x_pert_new, x_new_0)
        F_sym = ca.substitute(
            ca.jacobian(delta_out, delta_in), delta_in,
            ca.MX.zeros(n_tangent, 1))

        result: dict[str, Any] = {
            "x_sym": x_sym, "u_sym": u_sym, "n_sym": n_sym,
            "n_noise": n_noise, "F_sym": F_sym,
            "f_fn": None, "F_fn": None, "L_fn": None,
            "Sigma": None, "sensors": {}, "blocks": [],
        }
        L_pattern = None             # tangent × noise sparsity (for blocks)
        h_supports: list = []        # per-sensor observed tangent columns

        if build_functions:
            result["f_fn"] = ca.Function(
                "ekf_predict", [x_sym, u_sym, dt_sym, t_sym], [x_new_0],
                ["x", "u", "dt", "t"], ["x_new"])
            result["F_fn"] = ca.Function(
                "ekf_F", [x_sym, u_sym, dt_sym, t_sym], [F_sym],
                ["x", "u", "dt", "t"], ["F"])
            if n_noise > 0:
                delta_out_n = spec.boxminus_sym(x_new_n, x_new_0)
                L_sym = ca.substitute(
                    ca.jacobian(delta_out_n, n_sym), n_sym, zero_n)
                L_pattern = np.array(ca.DM(L_sym.sparsity()))
                result["L_fn"] = ca.Function(
                    "ekf_L", [x_sym, u_sym, dt_sym, t_sym], [L_sym],
                    ["x", "u", "dt", "t"], ["L"])
                sigmas_sq = []
                for ns in self._noise_specs:
                    sigmas_sq.extend([ns["sigma"] ** 2] * ns["dim"])
                result["Sigma"] = np.diag(sigmas_sq)

        # Per-(part, output) measurement plumbing. The observed-slot set
        # (columns of H with structural nonzeros) seeds the closure.
        slot_of_tan = self._slot_of_tan(spec)
        observed: set[str] = set()
        sensors: dict[tuple[int, str], dict[str, Any]] = {}
        for full in chosen_sensor_names:
            part, craft, out_name = sensor_lookup[full]
            if full not in outputs_n:
                raise RuntimeError(f"EKF: tick is missing output {full!r}.")
            h_dim       = int(outputs_n[full].numel())
            h_n_flat    = ca.reshape(outputs_n[full], h_dim, 1)
            h_pert_flat = ca.reshape(outputs_pert[full], h_dim, 1)
            H_sym = ca.substitute(
                ca.jacobian(h_pert_flat, delta_in),
                delta_in, ca.MX.zeros(n_tangent, 1))
            Hpat = np.array(ca.DM(H_sym.sparsity()))
            cols_touched = np.flatnonzero(Hpat.any(axis=0))
            for j in cols_touched:
                observed.add(slot_of_tan[int(j)])
            h_supports.append(cols_touched)
            if not build_functions:
                continue
            h_0_flat = ca.substitute(h_n_flat, n_sym, zero_n)
            if n_noise > 0:
                L_h_sym = ca.substitute(
                    ca.jacobian(h_n_flat, n_sym), n_sym, zero_n)
                L_h_fn = ca.Function(
                    f"Lh_{full}".replace(".", "_"),
                    [x_sym, u_sym, dt_sym, t_sym], [L_h_sym],
                    ["x", "u", "dt", "t"], ["L_h"])
            else:
                L_h_fn = None
            sensors[(id(part), out_name)] = {
                "dim":    h_dim,
                "h_fn":   ca.Function(
                    f"h_{full}".replace(".", "_"),
                    [x_sym, u_sym, dt_sym, t_sym], [h_0_flat],
                    ["x", "u", "dt", "t"], ["h"]),
                "H_fn":   ca.Function(
                    f"H_{full}".replace(".", "_"),
                    [x_sym, u_sym, dt_sym, t_sym], [H_sym],
                    ["x", "u", "dt", "t"], ["H"]),
                "L_h_fn": L_h_fn,
                "part":   part,
                "craft":  craft,
                "full":   full,
            }
        result["sensors"] = sensors
        result["observed_slots"] = observed
        if build_functions:
            result["blocks"] = self._compute_blocks(
                n_tangent, F_sym, L_pattern, h_supports)
        return result

    @staticmethod
    def _compute_blocks(n_tangent, F_sym, L_pattern, h_supports) -> list:
        """Partition the tangent state into independent subsystems.

        Two tangent indices land in the same block if they're coupled by
        the dynamics (`F`), share a process-noise channel (`L` column), or
        are jointly observed by one sensor (`H` support). Under this
        partition the covariance stays block-diagonal forever (from a
        block-diagonal start) — nothing creates a cross-block term — so the
        predict can propagate each block independently. Returns a list of
        sorted tangent-index arrays, ordered by first index.
        """
        parent = list(range(n_tangent))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Dynamics coupling (F[i, j] != 0 ⇒ i and j interact).
        rows, cols = np.nonzero(np.array(ca.DM(F_sym.sparsity())))
        for i, j in zip(rows.tolist(), cols.tolist()):
            union(i, j)
        # Shared process noise: a noise column feeding rows {r} ties them.
        if L_pattern is not None:
            for k in range(L_pattern.shape[1]):
                rws = np.flatnonzero(L_pattern[:, k])
                for r in rws[1:]:
                    union(int(rws[0]), int(r))
        # Joint observation: a sensor touching columns {c} ties them.
        for cols_touched in h_supports:
            for c in cols_touched[1:]:
                union(int(cols_touched[0]), int(c))

        groups: dict[int, list[int]] = {}
        for i in range(n_tangent):
            groups.setdefault(find(i), []).append(i)
        blocks = [np.array(sorted(v)) for v in groups.values()]
        blocks.sort(key=lambda b: int(b[0]))
        return blocks

    @staticmethod
    def _slot_of_tan(spec) -> list[str]:
        """Map each tangent index to the name of the slot it belongs to."""
        m: list[str] = [""] * spec.tangent_dim
        for s in spec.slots:
            for i in range(s.tangent_offset, s.tangent_offset + s.tangent_dim):
                m[i] = s.name
        return m

    def _closure(self, full_spec, F_sym, seed_slots) -> set[str]:
        """Backward reachability on the structural dependency graph.

        `F_sym[i, j] != 0` (structurally) means slot-row i's next value
        depends on slot-col j — so if i is kept, j must be too. Iterate
        from `seed_slots` to a fixpoint over the slot-level graph (slot
        granularity keeps SO3 orientation atomic).
        """
        slot_of_tan = self._slot_of_tan(full_spec)
        pattern = np.array(ca.DM(F_sym.sparsity()))
        rows, cols = np.nonzero(pattern)
        deps: dict[str, set[str]] = {}
        for i, j in zip(rows.tolist(), cols.tolist()):
            a, b = slot_of_tan[i], slot_of_tan[j]
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

    def _tick_outputs(self, cf, spec, x_sym, frozen,
                      u_sym, dt_sym, t_sym, n_sym) -> dict[str, ca.MX]:
        """Evaluate the compiled tick on flat symbolic inputs; return a
        dict mapping every output name to its MX expression.

        State slots in `spec` are sliced from `x_sym`; state slots / inputs
        in `frozen` are injected as baked constants; live inputs come from
        `u_sym`; noise channels from `n_sym`.
        """
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
            elif name in spec:
                slot = spec.slot(name)
                sliced.append(x_sym[slot.offset : slot.offset + slot.dim])
            elif name in frozen:
                val = np.atleast_1d(
                    np.asarray(frozen[name], dtype=float)).reshape(-1, 1)
                sliced.append(ca.DM(val))
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

    def _gather_state(self, spec, outputs_by_name: dict[str, ca.MX]) -> ca.MX:
        """Concatenate state-slot outputs in spec order → ambient vector."""
        chunks = []
        for slot in spec.slots:
            r = outputs_by_name[slot.name]
            if r.shape != (slot.dim, 1):
                r = ca.reshape(r, slot.dim, 1)
            chunks.append(r)
        return ca.vertcat(*chunks)

    def _classify_noise_input(self, full_name: str, owner, sub: str,
                              owner_label: str) -> None:
        """Append a noise-spec entry for `<owner>.<sub>` if it matches
        any noise channel's per-tick stochastic driver. The lookup is
        polymorphic on the Noise subclass — each subclass declares its
        own driver-input naming via `driver_input_name(name)`.

        `owner` can be a Part or a Disturbance — both expose
        `noise_declarations()` with the same shape.
        """
        for nname, ndecl in owner.noise_declarations().items():
            if ndecl.driver_input_name(nname) != sub:
                continue
            dim   = 1 if ndecl.shape == "scalar" else 3
            sigma = float(getattr(owner, f"{nname}_sigma"))
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
