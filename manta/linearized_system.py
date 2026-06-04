"""LinearizedSystem — the linearized model every transform reads.

`Sim`, `EKF`, and `LQR` are all "math over a linearized world": the forward
recurrence for the sim, the Kalman recursion for the filter, the Riccati
solve for the regulator. Everything *before* that math is identical
plumbing — compile the world tick, classify its I/O, resolve user-supplied
input/sensor names, carve the tracked sub-state (closed under the dynamics
for a filter, verbatim for a regulator), freeze the complement — and the
manifold-aware differentiation itself. `LinearizedSystem` owns ALL of it,
in one class.

What it exposes (everything a transform composes into kernels):

  symbols      `x_sym, u_sym, n_sym, dt_sym, t_sym`
  forward      `x_new` (noise-zeroed next state), `x_new_noisy` (noise live)
  derivatives  `F_sym` (∂δ'/∂δ), `B_sym` (∂δ'/∂u, when `control=True`),
               `L_sym` (∂δ'/∂noise), `Sigma` (channel covariance)
  sensors      `{full: SensorModel}` — per chosen output: `h_sym`
               (noise-zeroed), `h_noisy_sym`, `H_sym`, `L_h_sym`
  layout       `spec` (tracked StateSpec), `full_spec`, `tracked`, `frozen`,
               `ref_flat`, `blocks` (independent tangent subsystems)
  routing      `input_names`, `u_defaults`, `input_defaults`,
               `noise_specs`, `sample_rates`
  convenience  `predict_fn / F_fn / B_fn` and per-sensor `h_fn / H_fn`
               (`(x,u,dt,t)`-signature `ca.Function`s for point evaluation)

It contains NO filter or controller math — the Kalman recursion lives in
`manta.estimation.ekf`, Riccati in `manta.control.lqr`; each writes its
equations over these artifacts and emits a `Module`.

The differentiation is the standard manifold error-state recipe, built on
`spec.boxplus_sym/boxminus_sym` only:

    δ_out = boxminus(f(boxplus(x, δ_in), …), f(x, …)),  F = ∂δ_out/∂δ_in |₀

so SO(3) (and any future manifold) linearizes correctly with no
special-case code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .ir.state_spec import StateSpec, resolve_slotset


# ---------------------------------------------------------------------------
# Name / state-dict helpers
# ---------------------------------------------------------------------------

def resolve_suffix(key: str, candidates, *, label: str, who: str) -> str:
    """Resolve one user-supplied name against `candidates`.

    Accepts an exact match, else a unique `.<suffix>` match (craft-relative
    shorthand like ``"t.throttle"`` for ``"drone.t.throttle"``). Raises
    `KeyError` on an unknown or ambiguous key.
    """
    cands = list(candidates)
    if key in cands:
        return key
    matches = [n for n in cands if n.endswith("." + key)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"{who}: ambiguous {label} name {key!r}; matches {matches}. "
            f"Use the fully-qualified form.")
    raise KeyError(
        f"{who}: unknown {label} name {key!r}. Available: {sorted(cands)}")


def flatten_nested(nested: dict) -> dict[str, Any]:
    """`{owner: {slot: v}}` → `{"owner.slot": v}`; passes flat dicts through."""
    flat: dict[str, Any] = {}
    for owner, slots in nested.items():
        if isinstance(slots, dict):
            for slot, v in slots.items():
                flat[f"{owner}.{slot}"] = v
        else:
            flat[owner] = slots
    return flat


def freeze_complement(full_spec, kept, init_flat: dict,
                      into: dict | None = None) -> dict:
    """Freeze every full-spec slot NOT in `kept` at its `init_flat` value
    (zeros if absent), as a flat numpy column."""
    frozen = into if into is not None else {}
    for s in full_spec.slots:
        if s.name not in kept:
            val = init_flat.get(s.name, np.zeros(s.ambient_dim))
            frozen[s.name] = np.atleast_1d(
                np.asarray(val, dtype=float)).reshape(-1)
    return frozen


def _slot_of_tan(spec) -> list[str]:
    """Map each tangent index to its slot name (slot granularity keeps an
    SO(3) orientation atomic for closure/partitioning)."""
    m: list[str] = [""] * spec.tangent_dim
    for s in spec.slots:
        for i in range(s.tangent_offset, s.tangent_offset + s.tangent_dim):
            m[i] = s.name
    return m


# ---------------------------------------------------------------------------
# Per-sensor linearized measurement model
# ---------------------------------------------------------------------------

@dataclass
class SensorModel:
    """One chosen output's linearized measurement model, by name.

    Symbolic expressions are in the system's `(x_sym, u_sym, n_sym, dt_sym,
    t_sym)` symbols; `h_sym` is noise-zeroed, `h_noisy_sym` keeps the noise
    live (they coincide for a noiseless sensor). `h_fn`/`H_fn` are
    `(x,u,dt,t)` convenience functions for point evaluation/analysis."""
    full:          str
    dim:           int
    craft_name:    str
    part_name:     str
    out_name:      str
    h_sym:         ca.MX
    h_noisy_sym:   ca.MX
    H_sym:         ca.MX
    L_h_sym:       ca.MX | None
    observed_cols: np.ndarray
    h_fn:          ca.Function | None
    H_fn:          ca.Function | None


class LinearizedSystem:
    """Linearized recurrence + measurement model over a World's (sub)state.

    Args:
        world       — the model.
        track       — what to estimate/regulate. With `close_track=True`
                      (filter) a `{craft_name: SlotSet}` lower bound closed
                      under the dynamics; with `close_track=False`
                      (regulator) a verbatim slot-name list, the rest
                      frozen. `None` keeps the full state.
        sensors     — measurement full-names (or suffixes). `None` → none
                      when `close_track=False`, all when `track is None`,
                      else the tracked crafts' sensors.
        inputs      — live Input full-names (or suffixes). `None` keeps all;
                      excluded inputs freeze at their default.
        close_track — close `track` over `F`'s structural dependency.
        control     — also build `B_sym = ∂δ'/∂u`.
        ref         — operating-point overrides merged over the world's
                      initial state; the freeze reference.
    """

    def __init__(self, world, *,
                 track=None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 close_track: bool = True,
                 control: bool = False,
                 ref: dict | None = None) -> None:
        # --- world prep (idempotent) ------------------------------------
        if not world._planets_registered:
            for p in world._planets:
                p.register_disturbances(world)
            world._planets_registered = True
        world._resolve_planet_state_overrides()

        self.world = world
        self.crafts = tuple(world.crafts)
        if not self.crafts:
            raise ValueError("LinearizedSystem: world has no crafts.")

        full_spec = StateSpec.from_world(world)
        self.full_spec = full_spec

        # --- compile the tick + classify its I/O ------------------------
        from .tick import compile_world_tick, walk_tick_signature
        from .fields import CollisionField, FluidField, GravityField, MagField
        compiled = compile_world_tick(
            list(self.crafts), list(world._couplings),
            gravity_field=world.get_field(GravityField),
            fluid_field=world.get_field(FluidField),
            mag_field=world.get_field(MagField),
            collision_field=world.get_field(CollisionField),
        )
        self._cf = compiled.casadi_function
        self.tick = compiled
        self.sample_rates = getattr(compiled, "sample_rates", {})

        sig = walk_tick_signature(self._cf, world, full_spec)
        self.noise_specs = sig.noise
        self.n_noise = sum(s.dim for s in sig.noise)
        self.input_defaults = dict(sig.input_defaults)
        self._sensor_meta = {s.full: (s.craft.name, s.part.name, s.output_name)
                             for s in sig.sensors}

        # --- live-input subset (excluded inputs freeze at default) ------
        frozen: dict[str, Any] = {}
        if inputs is None:
            self.input_names = list(sig.input_names)
        else:
            chosen = {resolve_suffix(k, sig.input_names, label="input",
                                     who="LinearizedSystem") for k in inputs}
            for n in sig.input_names:
                if n not in chosen:
                    frozen[n] = self.input_defaults[n]
            self.input_names = [n for n in sig.input_names if n in chosen]
        self.u_defaults = np.array(
            [self.input_defaults[n] for n in self.input_names], dtype=float)

        # --- chosen sensors ---------------------------------------------
        all_sensor_names = sig.sensor_names
        if sensors is not None:
            chosen = {resolve_suffix(k, all_sensor_names, label="sensor",
                                     who="LinearizedSystem") for k in sensors}
            chosen_sensors = [n for n in all_sensor_names if n in chosen]
        elif not close_track:
            chosen_sensors = []                  # a regulator needs no h/H
        elif track is None:
            chosen_sensors = list(all_sensor_names)
        else:
            # Sensors on tracked crafts only — an untracked craft's sensor
            # would auto-expand the state and pull it back in.
            chosen_sensors = [n for n in all_sensor_names
                              if n.split(".", 1)[0] in track]

        # --- freeze reference (initial state + operating point) ---------
        ref_flat = flatten_nested(world._initial_state_dict())
        if ref is not None:
            ref_flat.update(flatten_nested(ref))
        self.ref_flat = ref_flat

        # --- carve the tracked sub-state --------------------------------
        if track is None:
            self.spec = full_spec
        elif close_track:
            craft_names = {c.name for c in self.crafts}
            unknown = set(track) - craft_names
            if unknown:
                raise KeyError(
                    f"LinearizedSystem: track references unknown craft(s) "
                    f"{sorted(unknown)}; world has {sorted(craft_names)}.")
            # Structure-only pass over the FULL spec: F's dependency
            # pattern + each chosen sensor's observed slots seed the
            # closure; the rest of the state freezes at the reference.
            d = self._differentiate(full_spec, dict(frozen), chosen_sensors,
                                    control=False, build_functions=False)
            sot = _slot_of_tan(full_spec)
            seed: set[str] = {sot[int(j)] for sm in d["sensors"].values()
                              for j in sm.observed_cols}
            for craft_name, slotset in track.items():
                seed |= resolve_slotset(craft_name, slotset)
            kept = self._closure(d["F_pattern"], sot, seed)
            self.spec = StateSpec.subset(full_spec, kept)
            freeze_complement(full_spec, kept, ref_flat, into=frozen)
        else:
            all_names = {s.name for s in full_spec.slots}
            unknown = set(track) - all_names
            if unknown:
                raise KeyError(
                    f"LinearizedSystem: track references unknown slot(s) "
                    f"{sorted(unknown)}; available {sorted(all_names)}.")
            kept = set(track)
            self.spec = StateSpec.subset(full_spec, kept)
            freeze_complement(full_spec, kept, ref_flat, into=frozen)

        self.frozen = frozen
        kept_names = {s.name for s in self.spec.slots}
        self.tracked = [s.name for s in full_spec.slots if s.name in kept_names]

        # --- differentiate over the (sub)spec ---------------------------
        d = self._differentiate(self.spec, frozen, chosen_sensors,
                                control=control, build_functions=True)
        self.x_sym, self.u_sym, self.n_sym = d["x"], d["u"], d["n"]
        self.dt_sym, self.t_sym = d["dt"], d["t"]
        self.x_new = d["x_new"]
        self.x_new_noisy = d["x_new_noisy"]
        self.F_sym = d["F_sym"]
        self.B_sym = d["B_sym"]
        self.L_sym = d["L_sym"]
        self.Sigma = d["Sigma"]
        self.sensors: dict[str, SensorModel] = d["sensors"]
        self.predict_fn = d["predict_fn"]
        self.F_fn = d["F_fn"]
        self.B_fn = d["B_fn"]
        self.L_fn = d["L_fn"]
        self.blocks = d["blocks"]

    # ------------------------------------------------------------------
    # The manifold-aware differentiation engine (private)
    # ------------------------------------------------------------------

    def _differentiate(self, spec, frozen, outputs, *, control: bool,
                       build_functions: bool) -> dict:
        """Linearize the compiled tick over `spec`: forward expressions,
        error-state Jacobians, and per-output measurement models. With
        `build_functions=False` only the structural artifacts (sparsity
        patterns, observed columns) are produced — the cheap pass that
        seeds state subsetting."""
        n_ambient, n_tangent = spec.ambient_dim, spec.tangent_dim
        n_u = len(self.input_names)

        x = ca.MX.sym("x", n_ambient, 1)
        u = ca.MX.sym("u", n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
        dt = ca.MX.sym("dt", 1, 1)
        t = ca.MX.sym("t", 1, 1)
        n = (ca.MX.sym("noise", self.n_noise, 1) if self.n_noise > 0
             else ca.MX.zeros(0, 1))
        zero_n = ca.MX.zeros(self.n_noise, 1)
        args, argn = [x, u, dt, t], ["x", "u", "dt", "t"]

        # forward (noise live + zeroed)
        outs_n = self._tick_outputs(spec, x, frozen, u, dt, t, n)
        x_new_n = self._gather_state(spec, outs_n)
        x_new_0 = ca.substitute(x_new_n, n, zero_n)

        # F = ∂δ'/∂δ at δ=0 (error-state recipe)
        delta_in = ca.MX.sym("delta_in", n_tangent, 1)
        outs_pert = self._tick_outputs(spec, spec.boxplus_sym(x, delta_in),
                                       frozen, u, dt, t, zero_n)
        x_pert_new = self._gather_state(spec, outs_pert)
        delta_out = spec.boxminus_sym(x_pert_new, x_new_0)
        F_sym = ca.substitute(ca.jacobian(delta_out, delta_in),
                              delta_in, ca.MX.zeros(n_tangent, 1))
        F_pattern = np.array(ca.DM(F_sym.sparsity()))

        # B = ∂δ'/∂u (regulators only)
        B_sym = None
        if control and n_u > 0:
            delta_u = ca.MX.sym("delta_u", n_u, 1)
            outs_pu = self._tick_outputs(spec, x, frozen, u + delta_u,
                                         dt, t, zero_n)
            delta_out_u = spec.boxminus_sym(
                self._gather_state(spec, outs_pu), x_new_0)
            B_sym = ca.substitute(ca.jacobian(delta_out_u, delta_u),
                                  delta_u, ca.MX.zeros(n_u, 1))

        # L = ∂δ'/∂noise, Σ
        L_sym, L_pattern, Sigma = None, None, None
        if self.n_noise > 0:
            delta_out_n = spec.boxminus_sym(x_new_n, x_new_0)
            L_sym = ca.substitute(ca.jacobian(delta_out_n, n), n, zero_n)
            L_pattern = np.array(ca.DM(L_sym.sparsity()))
            sig2: list[float] = []
            for ns in self.noise_specs:
                sig2.extend([ns.sigma ** 2] * ns.dim)
            Sigma = np.diag(sig2)

        # per-output measurement models
        sensors: dict[str, SensorModel] = {}
        h_supports: list[np.ndarray] = []
        for full in outputs:
            if full not in outs_n:
                raise RuntimeError(
                    f"LinearizedSystem: tick is missing output {full!r}.")
            dim = int(outs_n[full].numel())
            h_noisy = ca.reshape(outs_n[full], dim, 1)
            h_pert = ca.reshape(outs_pert[full], dim, 1)
            H_sym = ca.substitute(ca.jacobian(h_pert, delta_in),
                                  delta_in, ca.MX.zeros(n_tangent, 1))
            cols = np.flatnonzero(
                np.array(ca.DM(H_sym.sparsity())).any(axis=0))
            h_supports.append(cols)
            h_sym = ca.substitute(h_noisy, n, zero_n)
            L_h_sym = (ca.substitute(ca.jacobian(h_noisy, n), n, zero_n)
                       if self.n_noise > 0 else None)
            h_fn = H_fn = None
            if build_functions:
                safe = full.replace(".", "_")
                h_fn = ca.Function(f"h_{safe}", args, [h_sym], argn, ["h"])
                H_fn = ca.Function(f"H_{safe}", args, [H_sym], argn, ["H"])
            craft_name, part_name, out_name = self._sensor_meta[full]
            sensors[full] = SensorModel(
                full=full, dim=dim, craft_name=craft_name,
                part_name=part_name, out_name=out_name,
                h_sym=h_sym, h_noisy_sym=h_noisy, H_sym=H_sym,
                L_h_sym=L_h_sym, observed_cols=cols, h_fn=h_fn, H_fn=H_fn)

        predict_fn = F_fn = B_fn = L_fn = None
        blocks: list = []
        if build_functions:
            predict_fn = ca.Function("predict", args, [x_new_0],
                                     argn, ["x_new"])
            F_fn = ca.Function("F", args, [F_sym], argn, ["F"])
            if B_sym is not None:
                B_fn = ca.Function("B", args, [B_sym], argn, ["B"])
            if L_sym is not None:
                L_fn = ca.Function("L", args, [L_sym], argn, ["L"])
            blocks = self._partition(n_tangent, F_pattern, L_pattern,
                                     h_supports)

        return {"x": x, "u": u, "n": n, "dt": dt, "t": t,
                "x_new": x_new_0, "x_new_noisy": x_new_n,
                "F_sym": F_sym, "F_pattern": F_pattern, "B_sym": B_sym,
                "L_sym": L_sym, "Sigma": Sigma, "sensors": sensors,
                "predict_fn": predict_fn, "F_fn": F_fn, "B_fn": B_fn,
                "L_fn": L_fn, "blocks": blocks}

    def _tick_outputs(self, spec, x_sym, frozen, u_sym, dt_sym, t_sym,
                      n_sym) -> dict:
        """Evaluate the compiled tick on flat symbolic inputs → {output
        name: MX}. Spec slots slice from `x_sym`; frozen entries bake as
        constants; live inputs from `u_sym`; noise channels from `n_sym`."""
        cf = self._cf
        in_names = [cf.name_in(i) for i in range(cf.n_in())]
        out_names = [cf.name_out(i) for i in range(cf.n_out())]
        u_index = {name: i for i, name in enumerate(self.input_names)}
        noise_off: dict[str, tuple[int, int]] = {}
        off = 0
        for ns in self.noise_specs:
            noise_off[ns.full] = (off, ns.dim)
            off += ns.dim

        sliced: list = []
        for name in in_names:
            if name == "dt":
                sliced.append(dt_sym)
            elif name == "t":
                sliced.append(t_sym)
            elif name in spec:
                s = spec.slot(name)
                sliced.append(
                    x_sym[s.ambient_offset:s.ambient_offset + s.ambient_dim])
            elif name in frozen:
                val = np.atleast_1d(
                    np.asarray(frozen[name], dtype=float)).reshape(-1, 1)
                sliced.append(ca.DM(val))
            elif name in u_index:
                sliced.append(u_sym[u_index[name]])
            elif name in noise_off:
                start, dim = noise_off[name]
                sliced.append(n_sym[start:start + dim])
            else:
                raise RuntimeError(
                    f"LinearizedSystem: tick input {name!r} not handled.")

        result = cf(*sliced)
        if len(out_names) == 1:
            return {out_names[0]: result}
        return {nm: result[i] for i, nm in enumerate(out_names)}

    @staticmethod
    def _gather_state(spec, outputs_by_name: dict) -> ca.MX:
        """Concatenate state-slot outputs in spec order → ambient vector."""
        chunks = []
        for slot in spec.slots:
            r = outputs_by_name[slot.name]
            if r.shape != (slot.ambient_dim, 1):
                r = ca.reshape(r, slot.ambient_dim, 1)
            chunks.append(r)
        return ca.vertcat(*chunks)

    @staticmethod
    def _closure(F_pattern, sot, seed_slots) -> set[str]:
        """Expand `seed_slots` to the set closed under the dynamics:
        backward reachability on F's structural slot-dependency graph."""
        rows, cols = np.nonzero(F_pattern)
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
    def _partition(n_tangent, F_pattern, L_pattern, h_supports) -> list:
        """Partition the tangent state into independent subsystems (coupled
        by F, a shared noise column, or joint observation) — under it the
        covariance stays block-diagonal forever, so a predict can propagate
        each block independently (Σ O(n_b³) instead of O(n³))."""
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

        rows, cols = np.nonzero(F_pattern)
        for i, j in zip(rows.tolist(), cols.tolist()):
            union(i, j)
        if L_pattern is not None:
            for k in range(L_pattern.shape[1]):
                rws = np.flatnonzero(L_pattern[:, k])
                for r in rws[1:]:
                    union(int(rws[0]), int(r))
        for touched in h_supports:
            for c in touched[1:]:
                union(int(touched[0]), int(c))

        groups: dict[int, list[int]] = {}
        for i in range(n_tangent):
            groups.setdefault(find(i), []).append(i)
        blocks = [np.array(sorted(v)) for v in groups.values()]
        blocks.sort(key=lambda b: int(b[0]))
        return blocks

    # ------------------------------------------------------------------

    def pack_ref(self, spec) -> np.ndarray:
        """Pack the operating-point reference into `spec`'s ambient layout."""
        return spec.pack({k: v for k, v in self.ref_flat.items() if k in spec})

    def __repr__(self) -> str:
        return (f"<LinearizedSystem spec(tangent)={self.spec.tangent_dim} "
                f"inputs={self.input_names} sensors={list(self.sensors)}>")
