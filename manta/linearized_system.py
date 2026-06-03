"""LinearizedSystem — the pypose `System` analog for a manta World.

`EKF(world)` and `LQR(world)` are both "linearize the world tick over a
(sub)state, then do plain math on it" — the Kalman recursion for the filter,
the Riccati solve for the regulator. Everything *before* that plain math is
identical plumbing: walk the tick signature, resolve user-supplied
input/sensor names, carve the tracked sub-state (closed under the dynamics
for the filter, taken verbatim for the regulator), freeze the complement,
and hand the result to `Linearization`. `LinearizedSystem` owns ALL of that
— so the filter and regulator hold no slot/sensor/subset machinery and read
as the math they are.

It exposes, all indexed by name:

  * `spec`        — the tracked `StateSpec` (subset + frozen complement
                    already applied); `full_spec` is the un-subsetted layout.
  * `lin`         — the `Linearization` over `spec`: `f`, `F` (`= A`), `B`
                    (control Jacobian, when `control=True`), `L`/`Sigma`
                    (process noise), and the per-output `h`/`H`/`L_h`.
  * `sensors`     — `{full_name: {dim, h_fn, H_fn, L_h_fn, part, craft,
                    out_name, full}}` for the chosen measurements.
  * `input_names` / `u_defaults` / `input_defaults` — the control routing.
  * `tracked`     — tracked slot names in full-spec order.
  * `sample_rates`, `noise_specs`, `frozen`, `ref_flat`.

This is the home the design calls for: "ALL slot/sensor/subset machinery
lives here, NOT in the filter." It absorbs the former `linearized_world.py`
prologue helpers (now module-level functions below).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .linearization import Linearization
from .estimation.state_spec import StateSpec, resolve_slotset


# ---------------------------------------------------------------------------
# Name / state-dict helpers (the former linearized_world.py)
# ---------------------------------------------------------------------------

def resolve_suffix(key: str, candidates, *, label: str, who: str) -> str:
    """Resolve one user-supplied name against `candidates`.

    Accepts an exact match, else a unique `.<suffix>` match (craft-relative
    shorthand like ``"t.throttle"`` for ``"drone.t.throttle"``). Raises
    `KeyError` on an unknown or ambiguous key. `who` names the caller for
    the error (e.g. ``"EKF.predict"``); `label` names the kind (``"input"``,
    ``"sensor"``, ``"slot"``).
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
    (zeros if absent), as a flat numpy column. Returns the frozen dict
    (updating `into` in place when given)."""
    frozen = into if into is not None else {}
    for s in full_spec.slots:
        if s.name not in kept:
            val = init_flat.get(s.name, np.zeros(s.ambient_dim))
            frozen[s.name] = np.atleast_1d(
                np.asarray(val, dtype=float)).reshape(-1)
    return frozen


# ---------------------------------------------------------------------------
# LinearizedSystem
# ---------------------------------------------------------------------------

class LinearizedSystem:
    """Linearized recurrence + measurement model over a World's (sub)state.

    Args:
        world       — the model.
        track       — what to estimate/regulate. For the filter
                      (`close_track=True`) a `{craft_name: SlotSet}` *lower
                      bound* closed under the dynamics; for the regulator
                      (`close_track=False`) a verbatim list of slot names,
                      the rest frozen. `None` keeps the full state.
        sensors     — measurement full-names (or suffixes). `None` →
                      sensible default: none when not `close_track`, all
                      when `track is None`, else the tracked crafts' sensors.
        inputs      — live Input full-names (or suffixes). `None` keeps all;
                      excluded inputs freeze at their default.
        close_track — close `track` over `F`'s structural dependency (the
                      filter) vs. take it verbatim (the regulator).
        control     — also build the control Jacobian `B = ∂f/∂u`.
        ref         — operating-point overrides (nested or flat) merged over
                      the world's initial state; the freeze reference and the
                      point the regulator linearizes about.
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
        cf = compiled.casadi_function
        self.tick = compiled
        self.sample_rates = getattr(compiled, "sample_rates", {})

        sig = walk_tick_signature(cf, world, full_spec)
        self.noise_specs = sig.noise
        self.input_defaults = dict(sig.input_defaults)

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
        sensor_lookup = {s.full: (s.part, s.craft, s.output_name)
                         for s in sig.sensors}
        if sensors is not None:
            chosen = {resolve_suffix(k, all_sensor_names, label="sensor",
                                     who="LinearizedSystem") for k in sensors}
            chosen_sensor_names = [n for n in all_sensor_names if n in chosen]
        elif not close_track:
            chosen_sensor_names = []                 # regulator needs no h/H
        elif track is None:
            chosen_sensor_names = list(all_sensor_names)
        else:
            # Default to sensors on tracked crafts only — an untracked craft's
            # sensor would auto-expand the state and pull it back in.
            chosen_sensor_names = [n for n in all_sensor_names
                                   if n.split(".", 1)[0] in track]

        # --- freeze reference (initial state, plus any operating point) -
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
            # Structure-only full linearization → F's dependency pattern +
            # each chosen sensor's observed slots.
            full_lin = Linearization(
                cf, full_spec, frozen=dict(frozen),
                input_names=self.input_names, noise_specs=self.noise_specs,
                outputs=chosen_sensor_names, build_functions=False)
            seed: set[str] = set(full_lin.observed_slots)
            for craft_name, slotset in track.items():
                seed |= resolve_slotset(craft_name, slotset)
            kept = full_lin.dependency_closure(seed)
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

        # --- linearize over the (sub)spec -------------------------------
        self.lin = Linearization(
            cf, self.spec, frozen=frozen, input_names=self.input_names,
            noise_specs=self.noise_specs, outputs=chosen_sensor_names,
            control=control, build_functions=True)

        # --- sensor table (keyed by full name, in registration order) ---
        self.sensors: dict[str, dict[str, Any]] = {}
        for full in chosen_sensor_names:
            part, craft, out_name = sensor_lookup[full]
            o = self.lin.outputs[full]
            self.sensors[full] = {
                "dim": o.dim, "h_fn": o.h_fn, "H_fn": o.H_fn,
                "L_h_fn": o.L_h_fn, "part": part, "craft": craft,
                "out_name": out_name, "full": full,
            }

    # ---- convenience accessors (the linearized dynamics, by name) ------

    @property
    def f(self):
        """Predict `f(x,u,dt,t)` — next ambient state (noise-zeroed)."""
        return self.lin.predict_fn

    @property
    def F(self):
        """Tangent state-transition `∂f/∂δ` (the LQR's `A`)."""
        return self.lin.F_fn

    @property
    def B(self):
        """Control Jacobian `∂f/∂u` (None unless built with `control=True`)."""
        return self.lin.B_fn

    def pack_ref(self, spec) -> np.ndarray:
        """Pack the operating-point reference into `spec`'s ambient layout."""
        return spec.pack({k: v for k, v in self.ref_flat.items() if k in spec})

    def __repr__(self) -> str:
        return (f"<LinearizedSystem spec(tangent)={self.spec.tangent_dim} "
                f"inputs={self.input_names} sensors={list(self.sensors)}>")
