"""NumpySim — the THREADED simulation-oracle view."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...ir.module import Role, StateRef
from ...ir.state_spec import flatten_nested
from ...linearization import resolve_suffix
from ._noise import NoiseDriver
from ._runtime import NumpyRuntime, _split, pack_fields


def _stepn_port_arg(role: Role, *, hold, u, noise, params, dt, t, n):
    """The folded `mapaccum` argument for one step-entry PortRef, by Role:
    per-call vectors (u, noise, params) held constant across the n substeps
    (via `hold`), dt repeated, t advancing per substep."""
    if role is Role.CONTROL:
        return hold(u)
    if role is Role.NOISE:
        return hold(noise)
    if role is Role.PARAMETER:
        return hold(params)
    if role is Role.TIMESTEP:
        return ca.repmat(ca.DM(float(dt)), 1, n)
    if role is Role.TIME:
        return ca.DM(np.array([[t + k * dt for k in range(n)]]))
    raise NotImplementedError(
        f"step_n arg for role {role} — update _stepn_port_arg (and the "
        f"ARG_ROLES contract in manta.ir.module) for the new Role.")


class NumpySim(NumpyRuntime):
    """The simulation oracle. The runtime holds the nested state dict
    (`sim.state`); `step(dt)` advances it and realizes that step's sensor
    readings (`outputs()`), pulling wired command ports first and
    publishing readings to wired output ports after."""

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
        commands or override slots)."""
        if self._sim_state is None:
            self._sim_state = self.initial_state()
        return self._sim_state

    @state.setter
    def state(self, value: dict) -> None:
        self._sim_state = value

    # ---- step ----------------------------------------------------------

    def step(self, dt: float, *, t: float | None = None
             ) -> dict[str, dict[str, Any]]:
        """Advance the held state by `dt`: pull wired command ports (ZOH),
        run the oracle kernel (one noise draw), publish this step's
        readings to wired output ports. Returns the new state dict."""
        if isinstance(dt, dict):
            raise TypeError(
                "NumpySim.step: the functional step(state, dt) form was "
                "removed — mutate `sim.state` and call step(dt).")
        t0 = self._t if t is None else t
        st = self.state
        for full, v in self._ports.pull(t0).items():
            owner, rest = _split(full)
            st.setdefault(owner, {})[rest] = v
        self._sim_state = self._advance(st, float(dt), t0)
        self._t = t0 + float(dt)
        readings = {f"{o}.{s}": v for o, slots in self._outputs.items()
                    for s, v in slots.items()}
        self._ports.publish(readings, t0)
        return self._sim_state

    def _advance(self, state: dict, dt: float, t: float) -> dict:
        flat = flatten_nested(state)
        self._state["x"] = self._spec.pack_any(flat)
        u = pack_fields(self._u_fields(), flat,
                        default=lambda f: f.default, who="step")
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

    def step_n(self, dt: float, n: int, *, t: float | None = None
               ) -> dict[str, dict[str, Any]]:
        """Advance `n` substeps of `dt` in ONE folded call — commands held
        (ZOH), state chained through a `mapaccum` of the step kernel. Output
        readings + state are bit-identical to `n` sequential `step(dt)` calls;
        compiled (`_enable_compile`) it runs the whole inner loop in C.

        Falls back to sequential stepping when a `NoiseDriver` is attached (a
        fresh stochastic draw per substep cannot be folded)."""
        if n <= 1 or self._driver is not None:
            for k in range(int(n)):
                self.step(dt, t=None if t is None else t + k * dt)
            return self._sim_state
        t0 = self._t if t is None else t
        st = self.state
        for full, v in self._ports.pull(t0).items():
            owner, rest = _split(full)
            st.setdefault(owner, {})[rest] = v
        self._sim_state = self._advance_n(st, float(dt), int(n), t0)
        self._t = t0 + n * float(dt)
        readings = {f"{o}.{s}": v for o, slots in self._outputs.items()
                    for s, v in slots.items()}
        self._ports.publish(readings, t0)
        return self._sim_state

    def _step_n_fn(self, n: int):
        if n not in self._stepn_cache:
            # accumulate output 0 (x_new) -> input 0 (x); u/noise/dt/t are
            # per-substep parameters.
            self._stepn_cache[n] = self._functions["step"].mapaccum(
                f"step_x{n}", n, [0], [0])
        return self._stepn_cache[n]

    def _advance_n(self, state: dict, dt: float, n: int, t: float) -> dict:
        flat = flatten_nested(state)
        x0 = np.asarray(self._spec.pack_any(flat),
                        dtype=float).reshape(-1, 1)
        u = pack_fields(self._u_fields(), flat,
                        default=lambda f: f.default, who="step")
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

    # ---- ports -----------------------------------------------------------

    def out(self, name: str):
        """Producer port for a sensor reading (rate-gated sample-and-hold;
        published each step)."""
        full = resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="output", who=type(self).__name__)
        port = self.module.port(full)
        return self._ports.producer(full, dim=port.size, rate=port.rate)

    def command(self, name: str):
        """Latched (ZOH) consumer port for a control input, pulled before
        each step."""
        full = resolve_suffix(name, self._input_names(), label="input",
                              who=type(self).__name__)
        f = next(f for f in self._u_fields() if f.name == full)
        return self._ports.consumer(full, dim=f.dim, rate=f.rate)

    def __repr__(self) -> str:
        drv = "" if self._driver is None else f" +{self._driver!r}"
        return f"<NumpySim over {self.module!r}{drv}>"
