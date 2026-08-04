"""NumpySim — the THREADED simulation-oracle view."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...ir.module import Role, StateRef
from ...ir.state_spec import flatten_nested
from ...ir._names import resolve_suffix
from ...rates import CommandLatch, RateGate
from ..target import for_role
from ._noise import NoiseDriver
from ._runtime import NumpyRuntime, _split, pack_fields


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
    the loop's job — build a `rate_gate(...)` / `command_latch()` from the
    model's declared rates and apply them yourself (uniform across backends).
    Read sensors with `outputs()` (raw nested) or `reading(name)` (one, by
    name)."""

    def __init__(self, module) -> None:
        super().__init__(module)
        self._driver: NoiseDriver | None = None
        self._outputs: dict[str, dict[str, Any]] = {}
        self._sim_state: dict | None = None
        self._stepn_cache: dict[int, Any] = {}   # n → folded step kernel

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
        dict. Rate-gated actuators: pass a `command_latch()`-held `u`."""
        if isinstance(dt, dict):
            raise TypeError(
                "NumpySim.step: the functional step(state, dt) form was "
                "removed — pass commands as step(dt, u={...}).")
        t0 = self._t if t is None else t
        self._sim_state = self._advance(self.state, float(dt), t0, u)
        self._t = t0 + float(dt)
        return self._sim_state

    def _pack_u(self, flat: dict, u: dict[str, Any] | None) -> np.ndarray:
        """The flat control vector: the held `sim.state` inputs overlaid with
        this step's `u` (suffix names resolved)."""
        fields = self._u_fields()
        named = {f.name: (flat[f.name] if f.name in flat else f.default)
                 for f in fields}
        for k, v in (u or {}).items():
            named[resolve_suffix(k, self._input_names(), label="input",
                                 who=type(self).__name__)] = v
        return pack_fields(fields, named, default=lambda f: f.default,
                           who="step")

    def _advance(self, state: dict, dt: float, t: float,
                 u: dict[str, Any] | None = None) -> dict:
        flat = flatten_nested(state)
        self._check_state_keys(flat)
        self._state["x"] = self._spec.pack_any(flat)
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
        if n <= 1 or self._driver is not None:
            for k in range(int(n)):
                self.step(dt, t=None if t is None else t + k * dt, u=u)
            return self.state          # property: seeds when n == 0
        t0 = self._t if t is None else t
        self._sim_state = self._advance_n(self.state, float(dt), int(n), t0, u)
        self._t = t0 + n * float(dt)
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
        x0 = np.asarray(self._spec.pack_any(flat),
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
        self._state["x"] = np.asarray(outs[0])[:, -1].reshape(-1)
        readings = {
            name: np.asarray(outs[len(ep.writes) + i])[:, -1].reshape(-1)
            for i, name in enumerate(ep.returns)}
        return self._commit_step(state, readings)

    def _noise_vec(self, flat: dict | None = None) -> np.ndarray:
        """The flat noise draw, in port-field order: a `NoiseDriver` sample
        takes precedence, else a channel value set directly in the state
        dict (deterministic tests), else zero."""
        port = self._noise_port
        if port is None:
            return np.zeros(0)
        source = dict(flat) if flat is not None else {}
        if self._driver is not None:
            source.update(self._driver.sample())   # a draw wins over the dict
        return pack_fields(port.fields, source, default=0.0, who="noise")

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
        yourself with a `rate_gate(name)` if the sensor is rate-limited."""
        full = resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="output", who=type(self).__name__)
        owner, slot = _split(full)
        return self._outputs.get(owner, {}).get(slot)

    def rate_gate(self, name: str) -> RateGate:
        """A `RateGate` at sensor `name`'s declared rate (fires every step if
        the sensor has none). Apply it in your loop:
        `if g.due(t): ekf.update(name, sim.reading(name), u=u)`."""
        full = resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="output", who=type(self).__name__)
        return RateGate(self.module.port(full).rate)

    def command_latch(self) -> CommandLatch:
        """A `CommandLatch` over the actuators' declared intake rates (ZOH).
        Apply it once per step and pass the held `u` to BOTH `sim.step` and
        `ekf.predict`, so the filter predicts on the command truth acted on."""
        return CommandLatch({f.name: f.rate for f in self._u_fields()
                             if f.rate is not None})

    def __repr__(self) -> str:
        drv = "" if self._driver is None else f" +{self._driver!r}"
        return f"<NumpySim over {self.module!r}{drv}>"
