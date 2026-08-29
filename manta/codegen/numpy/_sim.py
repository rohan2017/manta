"""NumpySim — the THREADED simulation-oracle view."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..._validation import require_finite, require_positive
from ...ir._names import resolve_suffix
from ...ir.module import Role, StateRef
from ...ir.state_spec import flatten_nested
from ..target import for_role
from ._noise import NoiseCheckpoint, NoiseDriver
from ._runtime import NumpyRuntime, _split, finite_array, pack_fields


@dataclass(frozen=True)
class SimCheckpoint:
    """Complete deterministic restart point for a simulation runtime."""

    values: tuple[tuple[str, tuple[float, ...]], ...]
    outputs: tuple[tuple[str, tuple[float, ...]], ...]
    time: float
    artifact_id: str
    noise: NoiseCheckpoint | None
    models: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        def own(entries, *, who):
            try:
                entries = tuple(entries)
            except TypeError as exc:
                raise TypeError(f"SimCheckpoint.{who} must be an iterable") from exc
            owned = []
            names = set()
            for entry in entries:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise TypeError(
                        f"SimCheckpoint.{who} entries must be (name, values) pairs")
                name, values = entry
                if (not isinstance(name, str) or "." not in name
                        or any(not part.isidentifier()
                               for part in name.split("."))):
                    raise ValueError(
                        f"SimCheckpoint.{who}: invalid name {name!r}")
                if name in names:
                    raise ValueError(
                        f"SimCheckpoint.{who}: duplicate name {name!r}")
                names.add(name)
                arr = finite_array(
                    values, who=f"SimCheckpoint.{who} {name!r}").reshape(-1)
                if arr.size == 0:
                    raise ValueError(
                        f"SimCheckpoint.{who} {name!r}: values cannot be empty")
                owned.append((name, tuple(float(v) for v in arr)))
            return tuple(owned)

        object.__setattr__(self, "values", own(self.values, who="values"))
        object.__setattr__(self, "outputs", own(self.outputs, who="outputs"))
        object.__setattr__(self, "time", float(require_finite(
            self.time, name="SimCheckpoint.time")))
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("SimCheckpoint.artifact_id is required")
        if self.noise is not None and not isinstance(self.noise, NoiseCheckpoint):
            raise TypeError("SimCheckpoint.noise must be a NoiseCheckpoint or None")
        models = tuple(self.models)
        if any(not isinstance(item, (tuple, list)) or len(item) != 2
               for item in models):
            raise TypeError("SimCheckpoint.models entries must be (name, state) pairs")
        names = [item[0] for item in models]
        if (any(not isinstance(name, str) or not name.isidentifier()
                for name in names) or len(names) != len(set(names))):
            raise ValueError("SimCheckpoint.models needs unique identifier names")
        object.__setattr__(self, "models", tuple(
            (name, copy.deepcopy(state)) for name, state in models))


def _stepn_port_arg(role: Role, *, hold, u, noise, params, dt, t, n):
    """The folded `mapaccum` argument for one step-entry PortRef, by Role:
    per-call vectors (u, noise, params) held constant across the n substeps
    (via `hold`), dt repeated, t advancing per substep."""
    return for_role(role, {
        Role.CONTROL: lambda: hold(u),
        Role.NOISE: lambda: hold(noise),
        Role.PARAMETER: lambda: hold(params),
        Role.TIMESTEP: lambda: ca.repmat(ca.DM(float(dt)), 1, n),
        Role.TIME: lambda: ca.DM(np.array([[t + k * dt for k in range(n)]])),
    }, who="_stepn_port_arg")


class NumpySim(NumpyRuntime):
    """The simulation oracle. The runtime holds the nested state dict
    (`sim.state`); `step(dt, u={...})` applies the commands and advances it,
    realizing that step's sensor readings. The kernel is pure: rate gating is
    the downstream driving loop's job; declared rates remain Module metadata.
    Read sensors with `outputs()` (raw nested) or `reading(name)` (one, by
    name)."""

    def __init__(self, module) -> None:
        super().__init__(module)
        self._driver: NoiseDriver | None = None
        self._outputs: dict[str, dict[str, Any]] = {}
        self._sim_state: dict | None = None
        self._stepn_cache: dict[int, Any] = {}   # n → folded step kernel
        self._coupled_models: list[Any] = []

    # ---- held state ----------------------------------------------------

    def initial_state(self) -> dict[str, dict[str, Any]]:
        """Fresh nested initial state: the manifold slots' defaults plus
        input/noise placeholder entries (commands you may set; noise seeds
        that stay at zero — a `NoiseDriver` draw never enters the dict)."""
        nested = self._spec.to_nested(self.module.state.field("x").init)
        for f in self._u_fields():
            owner, rest = _split(f.name)
            nested.setdefault(owner, {}).setdefault(rest, f.default)
        if self._noise_port is not None:
            for f in self._noise_port.fields:
                owner, rest = _split(f.name)
                nested.setdefault(owner, {}).setdefault(
                    rest, np.zeros(f.dim) if f.dim > 1 else 0.0)
        return nested

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        """The held nested state (lazy-seeded; mutate in place to set
        commands or override slots).

        Aliasing rule: the OWNER DICTS stay live across steps (holding
        `st = sim.state['craft']` keeps working), but the slot VALUES are
        replaced each step — a reference to `sim.state['c']['position']`
        goes stale after `step()`; read through the dict, don't cache the
        array. Unknown keys are rejected at the next `step()` (a typo'd
        slot would otherwise be a silent no-op)."""
        if self._sim_state is None:
            self._sim_state = self.initial_state()
        return self._sim_state

    @state.setter
    def state(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError(
                f"{type(self).__name__}.state: expected a nested "
                f"{{owner: {{slot: value}}}} dict, got "
                f"{type(value).__name__}")
        flat = flatten_nested(value)
        missing = [s.name for s in self._spec.slots if s.name not in flat]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.state: assigned dict is missing "
                f"state slot(s) {missing}. Assignment replaces the whole "
                f"state — mutate `sim.state[owner][slot]` to override "
                f"individual slots.")
        self._check_state_keys(flat)
        self._sim_state = value

    @property
    def time(self) -> float:
        """Current logical time shared by the spatial and coupled models."""
        return self._t

    def model_state(self) -> dict[str, dict[str, Any]]:
        """Only manifold state slots, excluding commands/noise placeholders."""
        flat = flatten_nested(self.state)
        return self._spec.to_nested(self._spec.pack_projected(flat))

    def _known_state_keys(self) -> set[str]:
        """Every legitimate flat key of the state dict: manifold slots,
        command inputs, and noise placeholders."""
        if not hasattr(self, "_known_keys"):
            keys = {s.name for s in self._spec.slots}
            keys.update(f.name for f in self._u_fields())
            if self._noise_port is not None:
                keys.update(f.name for f in self._noise_port.fields)
            self._known_keys: set[str] = keys
        return self._known_keys

    def _check_state_keys(self, flat: dict) -> None:
        """Refuse unknown keys — a typo'd slot (`positon`) silently
        ignored by `pack_any` looks exactly like a physics bug."""
        unknown = set(flat) - self._known_state_keys()
        if unknown:
            raise KeyError(
                f"{type(self).__name__}: unknown state key(s) "
                f"{sorted(unknown)}. Known slots/inputs/noise: "
                f"{sorted(self._known_state_keys())}")

    # ---- step ----------------------------------------------------------

    def step(self, dt: float, *, t: float | None = None,
             u: dict[str, Any] | None = None
             ) -> dict[str, dict[str, Any]]:
        """Advance the held state by `dt`. `u` is `{input: value}` (full or
        suffix names) applied this step over the held `sim.state` inputs.
        Runs the oracle kernel (one noise draw) and returns the new state
        dict. Downstream code owns any actuator intake hold policy."""
        if isinstance(dt, dict):
            raise TypeError(
                "NumpySim.step: the functional step(state, dt) form was "
                "removed — pass commands as step(dt, u={...}).")
        dt = require_positive(dt, name="NumpySim.step dt")
        t0 = self._t if t is None else float(require_finite(t, name="NumpySim.step t"))
        next_t = float(require_finite(
            t0 + dt, name="NumpySim.step resulting time"))
        if not self._coupled_models:
            noise_before = self._driver.checkpoint() if self._driver else None
            x_before = self._state["x"].copy()
            try:
                self._sim_state = self._advance(self.state, dt, t0, u)
            except Exception:
                self._state["x"] = x_before
                if self._driver is not None and noise_before is not None:
                    self._driver.restore(noise_before)
                raise
            self._t = next_t
            return self._sim_state

        before = self.checkpoint()
        try:
            merged_u = self._coupled_inputs(u, t0, dt)
            self._sim_state = self._advance(self.state, dt, t0, merged_u)
            for model in self._coupled_models:
                model.post_step(self, next_t, dt)
        except Exception:
            self.restore(before)
            raise
        self._t = next_t
        return self._sim_state

    @staticmethod
    def _snapshot_values(nested: dict) -> tuple[tuple[str, tuple[float, ...]], ...]:
        flat = flatten_nested(nested)
        values = []
        for name in sorted(flat):
            raw = np.asarray(flat[name])
            if raw.dtype.kind not in "iuf":
                raise TypeError(f"checkpoint: {name!r} is not real numeric data")
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"checkpoint: {name!r} is non-finite")
            values.append((name, tuple(float(v) for v in arr)))
        return tuple(values)

    @staticmethod
    def _nested_values(values) -> dict[str, dict[str, Any]]:
        nested: dict[str, dict[str, Any]] = {}
        for full, data in values:
            owner, slot = _split(full)
            value = float(data[0]) if len(data) == 1 else np.asarray(data)
            nested.setdefault(owner, {})[slot] = value
        return nested

    def checkpoint(self) -> SimCheckpoint:
        self._check_state_keys(flatten_nested(self.state))
        models = []
        for model in self._coupled_models:
            state = model.checkpoint()
            models.append((model.name, state))
        return SimCheckpoint(
            self._snapshot_values(self.state),
            self._snapshot_values(self._outputs),
            float(self._t), self.module.artifact_id,
            self._driver.checkpoint() if self._driver else None,
            tuple(models))

    def restore(self, checkpoint: SimCheckpoint) -> None:
        if not isinstance(checkpoint, SimCheckpoint):
            raise TypeError("NumpySim.restore: expected SimCheckpoint")
        if checkpoint.artifact_id != self.module.artifact_id:
            raise ValueError("NumpySim.restore: checkpoint belongs to another artifact")
        if not np.isfinite(checkpoint.time):
            raise ValueError("NumpySim.restore: time must be finite")
        next_state = self._nested_values(checkpoint.values)
        flat = flatten_nested(next_state)
        self._check_state_keys(flat)
        if set(flat) != self._known_state_keys():
            raise ValueError("NumpySim.restore: checkpoint state layout is incomplete")
        self._spec.pack_projected(flat)
        self._pack_u(flat, None)
        self._noise_vec(flat)
        next_outputs = self._nested_values(checkpoint.outputs)
        if self._driver is None and checkpoint.noise is not None:
            raise ValueError("NumpySim.restore: checkpoint requires an attached driver")
        if self._driver is not None and checkpoint.noise is None:
            raise ValueError("NumpySim.restore: checkpoint has no attached-driver state")
        if self._driver is not None:
            self._driver.validate_checkpoint(checkpoint.noise)
        model_names = tuple(model.name for model in self._coupled_models)
        checkpoint_names = tuple(name for name, _ in checkpoint.models)
        if checkpoint_names != model_names:
            raise ValueError(
                "NumpySim.restore: coupled-model layout differs from checkpoint")
        for model, (_, state) in zip(self._coupled_models, checkpoint.models):
            model.validate_checkpoint(state)
        if self._driver is not None:
            self._driver.restore(checkpoint.noise)
        if self._sim_state is None:
            self._sim_state = next_state
        else:
            for owner in tuple(self._sim_state):
                if owner not in next_state:
                    del self._sim_state[owner]
            for owner, slots in next_state.items():
                live = self._sim_state.setdefault(owner, {})
                live.clear()
                live.update(slots)
        self._outputs = next_outputs
        self._t = float(checkpoint.time)
        self._state["x"] = self._spec.pack_projected(flat)
        for model, (_, state) in zip(self._coupled_models, checkpoint.models):
            model.restore(copy.deepcopy(state))

    def attach_model(self, model):
        """Attach physical non-spatial state to this simulation clock.

        A coupled model contributes pre-step Manta inputs and advances once
        after the corresponding physics tick. Checkpoint/restore and failures
        are atomic across the spatial model, noise driver, and every attached
        model.
        """
        required = ("name", "bind", "inputs", "post_step", "checkpoint",
                    "validate_checkpoint", "restore")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(f"coupled model is missing {missing}")
        if (not isinstance(model.name, str) or not model.name.isidentifier()
                or any(existing.name == model.name
                       for existing in self._coupled_models)):
            raise ValueError("coupled model name must be a unique identifier")
        model.bind(self)
        self._coupled_models.append(model)
        return model

    def _coupled_inputs(self, supplied, time: float, dt: float) -> dict:
        names = self._input_names()
        merged = {}
        owners = {}
        for key, value in (supplied or {}).items():
            full = resolve_suffix(key, names, label="input", who="NumpySim.step")
            merged[full] = value
            owners[full] = "caller"
        for model in self._coupled_models:
            values = model.inputs(self, time, dt)
            if not isinstance(values, dict):
                raise TypeError(f"coupled model {model.name!r} inputs must be a dict")
            for key, value in values.items():
                full = resolve_suffix(
                    key, names, label="input", who=f"coupled model {model.name}")
                if full in owners:
                    raise ValueError(
                        f"Manta input {full!r} has two owners: "
                        f"{owners[full]!r} and {model.name!r}")
                owners[full] = model.name
                merged[full] = value
        return merged

    def _pack_u(self, flat: dict, u: dict[str, Any] | None) -> np.ndarray:
        """The flat control vector: the held `sim.state` inputs overlaid with
        this step's `u` (suffix names resolved)."""
        fields = self._u_fields()
        named = {f.name: (flat.get(f.name, f.default))
                 for f in fields}
        if u:
            names = self._input_names()
            for k, v in u.items():
                named[resolve_suffix(k, names, label="input",
                                     who=type(self).__name__)] = v
        return pack_fields(fields, named, default=lambda f: f.default,
                           who="step")

    def _advance(self, state: dict, dt: float, t: float,
                 u: dict[str, Any] | None = None) -> dict:
        flat = flatten_nested(state)
        self._check_state_keys(flat)
        self._state["x"] = self._spec.pack_projected(flat)
        u = self._pack_u(flat, u)
        ep = self.module.entry("step")
        res = self._run(ep, {"u": u, "noise": self._noise_vec(flat),
                             "dt": dt, "t": t})
        readings = {name: res[name] for name in ep.returns}
        return self._commit_step(state, readings)

    def _commit_step(self, prev_state: dict, readings: dict) -> dict:
        """Write the freshly packed `self._state['x']` back into `prev_state`
        IN PLACE — manifold slots get this step's values; input-only entries
        (commands, noise placeholders — sensor readings deliberately stay OUT
        of the state dict) are untouched. Mutating in place keeps every dict
        reference a caller may hold (`st = sim.state['craft']`) live across
        steps. This step's `{full sensor name: reading}` is scattered into
        `self._outputs`."""
        for owner, slots in self._spec.to_nested(self._state["x"]).items():
            prev_state.setdefault(owner, {}).update(slots)
        self._outputs = {}
        for full, reading in readings.items():
            owner, slot = _split(full)
            self._outputs.setdefault(owner, {})[slot] = reading
        return prev_state

    def step_n(self, dt: float, n: int, *, t: float | None = None,
               u: dict[str, Any] | None = None
               ) -> dict[str, dict[str, Any]]:
        """Advance `n` substeps of `dt` in ONE folded call — `u` commands held
        (ZOH) for the block, state chained through a `mapaccum` of the step
        kernel. Output readings + state are bit-identical to `n` sequential
        `step(dt, u=u)` calls; compiled (`_enable_compile`) it runs the whole
        inner loop in C.

        Falls back to sequential stepping when a `NoiseDriver` is attached (a
        fresh stochastic draw per substep cannot be folded)."""
        dt = require_positive(dt, name="NumpySim.step_n dt")
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError(f"NumpySim.step_n n must be a non-negative integer, got {n!r}")
        n = int(n)
        if t is not None:
            t = float(require_finite(t, name="NumpySim.step_n t"))
        if n <= 1 or self._driver is not None or self._coupled_models:
            before = self.checkpoint()
            try:
                for k in range(n):
                    self.step(dt, t=None if t is None else t + k * dt, u=u)
            except Exception:
                self.restore(before)
                raise
            return self.state          # property: seeds when n == 0
        t0 = self._t if t is None else t
        next_t = float(require_finite(
            t0 + n * dt, name="NumpySim.step_n resulting time"))
        before = self.checkpoint()
        try:
            self._sim_state = self._advance_n(self.state, dt, n, t0, u)
        except Exception:
            self.restore(before)
            raise
        self._t = next_t
        return self._sim_state

    def _step_n_fn(self, n: int):
        if n not in self._stepn_cache:
            # accumulate output 0 (x_new) -> input 0 (x); u/noise/dt/t are
            # per-substep parameters.
            self._stepn_cache[n] = self._functions["step"].mapaccum(
                f"step_x{n}", n, [0], [0])
        return self._stepn_cache[n]

    def _advance_n(self, state: dict, dt: float, n: int, t: float,
                   u: dict[str, Any] | None = None) -> dict:
        flat = flatten_nested(state)
        self._check_state_keys(flat)
        x0 = np.asarray(self._spec.pack_projected(flat),
                        dtype=float).reshape(-1, 1)
        u = self._pack_u(flat, u)
        noise = self._noise_vec(flat)
        fn = self._step_n_fn(n)

        def _held(vec: np.ndarray):
            """A per-call vector held constant across the n substeps."""
            return (ca.repmat(ca.DM(vec.reshape(-1, 1)), 1, n) if vec.size
                    else ca.DM(0, n))

        ep = self.module.entry("step")
        params = self.param_vector()
        call_args: list = []
        for a in ep.args:
            if isinstance(a, StateRef):
                call_args.append(x0)
                continue
            call_args.append(_stepn_port_arg(
                self.module.port(a.name).role, hold=_held, u=u, noise=noise,
                params=params, dt=dt, t=t, n=n))
        res = fn(*call_args)
        outs = [res] if not isinstance(res, (list, tuple)) else list(res)
        expected = len(ep.writes) + len(ep.returns)
        if len(outs) != expected:
            raise RuntimeError(
                f"{self.module.name}.step_n: folded kernel produced "
                f"{len(outs)} outputs, expected {expected}")
        next_x = finite_array(
            np.asarray(outs[0])[:, -1], who="NumpySim.step_n state",
            size=self._spec.ambient_dim).reshape(-1)
        readings = {
            name: finite_array(
                np.asarray(outs[len(ep.writes) + i])[:, -1],
                who=f"NumpySim.step_n output {name!r}",
                size=self.module.port(name).size).reshape(-1)
            for i, name in enumerate(ep.returns)}
        self._state["x"] = next_x
        return self._commit_step(state, readings)

    def _noise_vec(self, flat: dict | None = None) -> np.ndarray:
        """The flat noise draw, in port-field order: a `NoiseDriver` sample
        takes precedence, else a channel value set directly in the state
        dict (deterministic tests), else zero."""
        port = self._noise_port
        if port is None:
            return np.zeros(0)
        draw = self._driver.sample() if self._driver is not None else {}
        held = flat if flat is not None else {}
        selected = {}
        for f in port.fields:
            if f.name in draw:                     # a draw wins over the dict
                selected[f.name] = draw[f.name]
            elif f.name in held:
                selected[f.name] = held[f.name]
        return pack_fields(port.fields, selected, default=0.0, who="noise")

    def outputs(self) -> dict[str, dict[str, Any]]:
        """Sensor readings from the most recent step (nested, realized
        with that step's noise draw)."""
        return self._outputs

    # ---- noise ----------------------------------------------------------

    def attach_driver(self, driver: NoiseDriver) -> NoiseDriver:
        """Attach a stochastic `NoiseDriver`: every active (σ>0) channel of
        the Module's NOISE port is sampled each step, so truth is noisy
        with the very σ the EKF reads for R/Q. Without one the sim is a
        noiseless oracle."""
        if self._noise_port is None:
            raise ValueError(
                f"{self.module.name}: module declares no noise port.")
        driver.bind(self._noise_port.fields)
        self._driver = driver
        return driver

    @property
    def driver(self) -> NoiseDriver | None:
        return self._driver

    # ---- reading sensors + rate tools ------------------------------------

    def reading(self, name: str) -> Any:
        """The latest raw reading for a sensor (full or suffix name) from the
        most recent step. Readings are realized every step; gate feeding
        downstream according to the measurement port's declared rate."""
        full = resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="output", who=type(self).__name__)
        owner, slot = _split(full)
        return self._outputs.get(owner, {}).get(slot)

    def __repr__(self) -> str:
        drv = "" if self._driver is None else f" +{self._driver!r}"
        return f"<NumpySim over {self.module!r}{drv}>"
