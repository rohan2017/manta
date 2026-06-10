"""LinearizedSystem — the linearized model every transform reads.

`Sim`, `EKF`, and `LQR` are all "math over a linearized world": the forward
recurrence for the sim, the Kalman recursion for the filter, the Riccati
solve for the regulator. Everything *before* that math is identical
plumbing — compile the world tick, classify its I/O, resolve user-supplied
input/sensor names, carve the tracked sub-state (closed under the dynamics
for a filter, verbatim for a regulator), freeze the complement — plus the
manifold-aware differentiation itself. `LinearizedSystem` orchestrates it;
the pieces live beside it:

  * `engine.TickLinearizer` — the differentiation recipe over the tick.
  * `partition`             — dependency closure + independent blocks.
  * `names`                 — suffix resolution + freeze helpers.

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
  parameters   `p_sym`, `param_specs`, `param_names`, `n_param`,
               `p_defaults` (promoted tunable Parameters; empty unless
               built with `parameters=[...]`)
  convenience  `predict_fn / F_fn / B_fn` and per-sensor `h_fn / H_fn`
               (`(x,u,dt,t)`-signature `ca.Function`s for point evaluation)

It contains NO filter or controller math — the Kalman recursion lives in
`manta.estimation.ekf`, Riccati in `manta.control.lqr`; each writes its
equations over these artifacts and emits a `Module`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ir.state_spec import StateSpec, flatten_nested, resolve_slotset
from .engine import SensorModel, TickLinearizer
from .names import freeze_complement, resolve_suffix, slot_of_tangent_index
from .partition import dependency_closure


class LinearizedSystem:
    """Linearized recurrence + measurement model over a World's (sub)state.

    Args:
        world       — the model.
        track       — what to estimate/regulate. With `track_mode=
                      "closure"` (filter) a `{craft_name: SlotSet}` lower
                      bound closed under the dynamics; with `track_mode=
                      "verbatim"` (regulator) an exact slot-name list, the
                      rest frozen. `None` keeps the full state.
        sensors     — measurement full-names (or suffixes). `None` → none
                      when `track_mode="verbatim"`, all when `track is
                      None`, else the tracked crafts' sensors.
        inputs      — live Input full-names (or suffixes). `None` keeps all;
                      excluded inputs freeze at their default.
        track_mode  — "closure" expands `track` over F's structural
                      dependencies (a filter must propagate everything the
                      tracked slots couple to); "verbatim" takes it
                      exactly as given (a regulator deliberately freezes
                      the rest).
        parameters  — promotable Parameter full-names (or suffixes) to
                      promote from baked graph constants to a live
                      parameter vector `p` (system ID). With any given,
                      `x_new`/`h` expressions carry a `p_sym` and every
                      convenience Function gains a `p` argument; see
                      `param_specs` / `p_defaults`. `None` (default) —
                      every parameter bakes as today.
        control     — also build `B_sym = ∂δ'/∂u`.
        ref         — operating-point overrides merged over the world's
                      initial state; the freeze reference.
        discretization — how `F_sym` discretizes the dynamics:
                      * "exact" (default) — F = ∂δ'/∂δ of the full
                        discrete tick. The reference jacobian, but its
                        code generation differentiates through the whole
                        integrator (both joint-space factorizations, the
                        momentum recovery, Rodrigues) — the dominant
                        kernel in a C++ deploy module.
                      * "euler" — F = I + dt·M with M = ∂²δ'/∂dt∂δ at
                        (δ=0, dt=0): the continuous-time linearization,
                        Euler-discretized. Equal to "exact" up to O(dt²)
                        — below EKF fidelity — including all manifold
                        transport terms (they fall out of the same
                        boxminus recipe). B/L stay exact.
    """

    def __init__(self, world, *,
                 track=None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 track_mode: str = "closure",
                 control: bool = False,
                 ref: dict | None = None,
                 discretization: str = "exact",
                 parameters: list[str] | None = None) -> None:
        if track_mode not in ("closure", "verbatim"):
            raise ValueError(
                f"LinearizedSystem: track_mode must be 'closure' or "
                f"'verbatim', got {track_mode!r}")
        if discretization not in ("exact", "euler"):
            raise ValueError(
                f"LinearizedSystem: discretization must be 'exact' or "
                f"'euler', got {discretization!r}")

        # The first transform finalizes the world: compile-time
        # registrations run once and the model locks (post-compile
        # additions raise instead of being silently invisible).
        world.finalize()

        self.world = world
        self.crafts = tuple(world.crafts)
        if not self.crafts:
            raise ValueError("LinearizedSystem: world has no crafts.")
        self.discretization = discretization
        self._validate_model(world)

        full_spec = StateSpec.from_world(world)
        self.full_spec = full_spec

        # --- compile the tick + classify its I/O ------------------------
        self._compile_tick(world, self._resolve_parameters(parameters))

        # --- live-input subset (excluded inputs freeze at default) ------
        frozen = self._choose_inputs(inputs)
        engine = TickLinearizer(self._cf, self.input_names,
                                self.noise_specs, discretization,
                                param_specs=self.param_specs)

        chosen_sensors = self._choose_sensors(sensors, track, track_mode)

        # --- freeze reference (initial state + operating point) ---------
        ref_flat = flatten_nested(world._initial_state_dict())
        if ref is not None:
            ref_flat.update(flatten_nested(ref))
        self.ref_flat = ref_flat

        self._carve_subspec(track, track_mode, chosen_sensors, frozen,
                            engine)
        self.frozen = frozen
        kept_names = {s.name for s in self.spec.slots}
        self.tracked = [s.name for s in full_spec.slots
                        if s.name in kept_names]

        # --- differentiate over the (sub)spec ---------------------------
        d = engine.differentiate(self.spec, frozen, chosen_sensors,
                                 control=control)
        self.x_sym, self.u_sym, self.n_sym = d["x"], d["u"], d["n"]
        self.p_sym = d["p"]
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
    # Construction passes
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model(world) -> None:
        """Verify per-part `requires_fields` / `requires_planet` against
        the world's registry — every transform passes through here."""
        for craft in world.crafts:
            for part in craft.parts:
                for req_cls in getattr(type(part), "requires_fields", []):
                    if not any(isinstance(f, req_cls)
                               for f in world.fields):
                        raise ValueError(
                            f"World '{world.name}': part "
                            f"{type(part).__name__}('{part.name}') requires "
                            f"a registered {req_cls.__name__} but none is "
                            f"attached to this world.")
                req_planet = getattr(type(part), "requires_planet", None)
                if req_planet is not None and not any(
                        isinstance(p, req_planet) for p in world._planets):
                    raise ValueError(
                        f"World '{world.name}': part "
                        f"{type(part).__name__}('{part.name}') requires a "
                        f"{req_planet.__name__} planet but none is "
                        f"registered with this world.")

    def _resolve_parameters(self, parameters) -> set[str]:
        """Resolve requested tunable-parameter names (full or suffix)
        against the model's promotable Parameter declarations."""
        if not parameters:
            return set()
        available = [f"{c.name}.{p.name}.{n}" for c in self.crafts
                     for p in c.parts
                     for n in p.promotable_parameter_declarations()]
        return {resolve_suffix(k, available, label="parameter",
                               who="LinearizedSystem") for k in parameters}

    def _compile_tick(self, world, tunable_params: set[str]) -> None:
        """Compile the shared world tick and walk its signature."""
        from ..tick import compile_world_tick, walk_tick_signature
        from ..fields import (CollisionField, FluidField, GravityField,
                              MagField)
        compiled = compile_world_tick(
            list(self.crafts), list(world._couplings),
            gravity_field=world.get_field(GravityField),
            fluid_field=world.get_field(FluidField),
            mag_field=world.get_field(MagField),
            collision_field=world.get_field(CollisionField),
            world=world,
            tunable_params=tunable_params,
        )
        self._cf = compiled.casadi_function
        self.tick = compiled
        self.sample_rates = getattr(compiled, "sample_rates", {})

        sig = walk_tick_signature(self._cf, world, self.full_spec)
        self._sig = sig
        self.noise_specs = sig.noise
        self.n_noise = sum(s.dim for s in sig.noise)
        self.input_defaults = dict(sig.input_defaults)
        # Promoted-parameter channels (cf-signature order = `p` layout).
        self.param_specs = sig.params
        self.param_names = sig.param_names
        self.n_param = sum(p.dim for p in sig.params)
        self.p_defaults = (np.concatenate([p.value for p in sig.params])
                           if sig.params else np.zeros(0))

    def _choose_inputs(self, inputs) -> dict[str, Any]:
        """Resolve the live-input subset; excluded inputs freeze at their
        default. Returns the (started) frozen map."""
        sig = self._sig
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
        return frozen

    def _choose_sensors(self, sensors, track, track_mode) -> list[str]:
        """Resolve the chosen measurement outputs."""
        all_sensor_names = self._sig.sensor_names
        if sensors is not None:
            chosen = {resolve_suffix(k, all_sensor_names, label="sensor",
                                     who="LinearizedSystem")
                      for k in sensors}
            return [n for n in all_sensor_names if n in chosen]
        if track_mode == "verbatim":
            return []                            # a regulator needs no h/H
        if track is None:
            return list(all_sensor_names)
        # Sensors on tracked crafts only — an untracked craft's sensor
        # would auto-expand the state and pull it back in.
        return [n for n in all_sensor_names if n.split(".", 1)[0] in track]

    def _carve_subspec(self, track, track_mode, chosen_sensors, frozen,
                       engine: TickLinearizer) -> None:
        """Set `self.spec` to the tracked sub-state and freeze the
        complement at the reference (into `frozen`, in place)."""
        full_spec = self.full_spec
        if track is None:
            self.spec = full_spec
            return
        if track_mode == "closure":
            craft_names = {c.name for c in self.crafts}
            unknown = set(track) - craft_names
            if unknown:
                raise KeyError(
                    f"LinearizedSystem: track references unknown craft(s) "
                    f"{sorted(unknown)}; world has {sorted(craft_names)}.")
            # Structure-only pass over the FULL spec: F's dependency
            # pattern + each chosen sensor's observed slots seed the
            # closure; the rest of the state freezes at the reference.
            # This pass cannot be merged with the real differentiation —
            # the subset it decides is exactly what the real pass runs
            # over.
            d = engine.structure(full_spec, dict(frozen), chosen_sensors)
            sot = slot_of_tangent_index(full_spec)
            seed: set[str] = {sot[int(j)] for sm in d["sensors"].values()
                              for j in sm.observed_cols}
            for craft_name, slotset in track.items():
                seed |= resolve_slotset(craft_name, slotset)
            kept = dependency_closure(d["F_pattern"], sot, seed)
        else:                                    # verbatim
            all_names = {s.name for s in full_spec.slots}
            unknown = set(track) - all_names
            if unknown:
                raise KeyError(
                    f"LinearizedSystem: track references unknown slot(s) "
                    f"{sorted(unknown)}; available {sorted(all_names)}.")
            kept = set(track)
        self.spec = StateSpec.subset(full_spec, kept)
        freeze_complement(full_spec, kept, self.ref_flat, into=frozen)

    # ------------------------------------------------------------------

    def pack_ref(self, spec) -> np.ndarray:
        """Pack the operating-point reference into `spec`'s ambient layout."""
        return spec.pack_any(self.ref_flat)

    def __repr__(self) -> str:
        return (f"<LinearizedSystem spec(tangent)={self.spec.tangent_dim} "
                f"inputs={self.input_names} sensors={list(self.sensors)}>")
