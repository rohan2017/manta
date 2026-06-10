"""NumpySim — the THREADED simulation-oracle view."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from ...ir.state_spec import flatten_nested
from ...linearization import resolve_suffix
from ._noise import NoiseDriver
from ._runtime import NumpyRuntime, _split


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
        spec = self._spec
        flat = flatten_nested(state)
        self._state["x"] = spec.pack_any(flat)
        u = np.array([float(np.asarray(flat.get(f.name, f.default)).ravel()[0])
                      for f in self._u_fields()])
        readings = self._run(self.module.entry("step"),
                             {"u": u, "noise": self._noise_vec(flat),
                              "dt": dt, "t": t})
        new_state = spec.to_nested(self._state["x"])
        # Preserve input-only entries (commands, noise placeholders); the
        # sensor readings deliberately stay OUT of the state dict.
        for owner, slots in state.items():
            new_state.setdefault(owner, {})
            for slot, val in slots.items():
                if slot not in new_state[owner]:
                    new_state[owner][slot] = val
        self._outputs = {}
        for full, reading in readings.items():
            owner, slot = _split(full)
            self._outputs.setdefault(owner, {})[slot] = reading
        return new_state

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
        cache = self.__dict__.setdefault("_stepn_cache", {})
        if n not in cache:
            # accumulate output 0 (x_new) -> input 0 (x); u/noise/dt/t are
            # per-substep parameters.
            cache[n] = self._functions["step"].mapaccum(f"step_x{n}", n,
                                                        [0], [0])
        return cache[n]

    def _advance_n(self, state: dict, dt: float, n: int, t: float) -> dict:
        spec = self._spec
        flat = flatten_nested(state)
        x0 = np.asarray(spec.pack_any(flat), dtype=float).reshape(-1, 1)
        u = np.array([float(np.asarray(flat.get(f.name, f.default)).ravel()[0])
                      for f in self._u_fields()])
        noise = self._noise_vec(flat)
        fn = self._step_n_fn(n)
        U = ca.repmat(ca.DM(u.reshape(-1, 1)), 1, n) if u.size else ca.DM(0, n)
        NO = (ca.repmat(ca.DM(noise.reshape(-1, 1)), 1, n) if noise.size
              else ca.DM(0, n))
        DT = ca.repmat(ca.DM(float(dt)), 1, n)
        T = ca.DM(np.array([[t + k * dt for k in range(n)]]))
        res = fn(x0, U, NO, DT, T)
        outs = [res] if not isinstance(res, (list, tuple)) else list(res)
        self._state["x"] = np.asarray(outs[0])[:, -1].reshape(-1)
        new_state = spec.to_nested(self._state["x"])
        for owner, slots in state.items():
            new_state.setdefault(owner, {})
            for slot, val in slots.items():
                if slot not in new_state[owner]:
                    new_state[owner][slot] = val
        ep = self.module.entry("step")
        self._outputs = {}
        for i, name in enumerate(ep.returns):
            owner, slot = _split(name)
            self._outputs.setdefault(owner, {})[slot] = (
                np.asarray(outs[len(ep.writes) + i])[:, -1].reshape(-1))
        return new_state

    def _noise_vec(self, flat: dict | None = None) -> np.ndarray:
        """The flat noise draw, in port-field order: a `NoiseDriver` sample
        takes precedence, else a channel value set directly in the state
        dict (deterministic tests), else zero."""
        port = self._noise_port
        if port is None:
            return np.zeros(0)
        vec = np.zeros(port.size)
        samples = self._driver.sample() if self._driver is not None else {}
        off = 0
        for f in port.fields:
            if f.name in samples:
                vec[off:off + f.dim] = np.asarray(samples[f.name]).ravel()
            elif flat is not None and f.name in flat:
                vec[off:off + f.dim] = np.asarray(flat[f.name]).ravel()
            off += f.dim
        return vec

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
