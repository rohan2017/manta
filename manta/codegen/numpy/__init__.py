"""TargetNumpy — the native-Python backend: ONE runtime for every Module.

`TargetNumpy(x)` lowers a typed `Module` (or any transform exposing
`.module()` — `Sim`, `EKF`, `LQR`, a recurrence block) to a single
`NumpyRuntime`. There are no per-transform runtime classes: the runtime's
surface is *derived from the Module's structure* —

  * THREADED + an oracle ``step`` entry (a Sim)   → dict-state functional
    `step(state, dt)` / bus `step(dt)`, `outputs()`, `out`/`command` ports,
    `attach_driver` (feeds the NOISE port).
  * HELD + MEASUREMENT ports (a filter)           → `predict`/`update`
    (by sensor *name*), `feed`/`step` (the measurement bus), `x`/`P`,
    `reset`, `state_dict`, `meas`/`command`/`estimate` ports.
  * HELD + an OUTPUT port (a recurrence)          → `step(dt, **inputs)`,
    `outputs()`, `reset`, `input`/`output` ports, `compute`.
  * THREADED + a ``control`` entry (a regulator)  → `u(x)`,
    `control(state_dict)`, `estimate`/`command` ports, `compute`.

Everything is driven by the IR's types: slot names come from the manifold
spec, input/sensor/noise names + defaults/σ/rates from the Ports. The
backend never mentions a transform. `call(method, values)` is the generic
escape hatch onto any entry point.
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from ...bus import MeasurementBus, PortSet
from ...ir.module import Hosting, Module, PortRef, Role, StateRef
from ...linearized_system import resolve_suffix


def _split(full: str) -> tuple[str, str]:
    owner, rest = full.split(".", 1)
    return owner, rest


class NumpyRuntime:
    """The one native-Python runtime over a typed `Module`."""

    def __init__(self, module: Module) -> None:
        self.module = module
        self._spec = module.spec
        self._state: dict[str, np.ndarray] = {}
        for f in module.state.fields:
            a = np.asarray(f.init, dtype=float)
            self._state[f.name] = (a.reshape(f.shape) if f.kind == "matrix"
                                   else a.reshape(-1).copy())

        ctrl = module.ports_by_role(Role.CONTROL)
        self._u_port = ctrl[0] if ctrl else None
        noi = module.ports_by_role(Role.NOISE)
        self._noise_port = noi[0] if noi else None
        self._meas_ports_ir = module.ports_by_role(Role.MEASUREMENT)
        out = module.ports_by_role(Role.OUTPUT)
        self._y_port = out[0] if out else None
        st = module.ports_by_role(Role.STATE)
        self._x_port = st[0] if st else None
        self._methods = {e.method for e in module.entry_points}

        self._t = 0.0
        self._driver: "NoiseDriver | None" = None
        self._outputs: dict[str, dict[str, Any]] = {}
        self._bus_state: dict | None = None
        self._ports = PortSet()
        self._y = (np.zeros(self._y_port.size) if self._y_port is not None
                   else None)

        # Filter surface: the measurement bus (mailboxes per MEASUREMENT
        # port — possibly none — + ZOH commands + the estimate port). A
        # held module with a `predict` entry IS a filter; a held module
        # with an OUTPUT port is a recurrence.
        self._bus: MeasurementBus | None = None
        if module.hosting is Hosting.HELD and "predict" in self._methods:
            self._bus = MeasurementBus(
                self,
                sensors={p.name: {"dim": p.size, "key": p.name}
                         for p in self._meas_ports_ir},
                input_names=self._input_names(),
                sample_rates={f.name: f.rate for f in self._u_fields()
                              if f.rate is not None},
                estimate_dim=self._spec.ambient_dim,
                estimate_layout={s.name: (s.ambient_offset, s.ambient_dim)
                                 for s in self._spec.slots})

    # ------------------------------------------------------------------
    # The kernel engine (typed-arg gather → call → scatter)
    # ------------------------------------------------------------------

    def call(self, method: str, values: dict[str, Any] | None = None,
             **kw) -> dict[str, np.ndarray]:
        """Run one entry point. `values`/kwargs are keyed by port name
        (use the dict for dotted names); TIME ports default to 0."""
        vals = dict(values or {})
        vals.update(kw)
        return self._run(self.module.entry(method), vals)

    def _run(self, ep, values: dict[str, Any]) -> dict[str, np.ndarray]:
        m = self.module
        args = []
        for a in ep.args:
            if isinstance(a, StateRef):
                args.append(self._state[a.name])
            elif isinstance(a, PortRef):
                if a.name in values:
                    args.append(values[a.name])
                elif m.port(a.name).role is Role.TIME:
                    args.append(0.0)
                else:
                    raise KeyError(
                        f"{m.name}.{ep.method}: missing value for port "
                        f"{a.name!r}.")
            else:                                  # pragma: no cover
                raise TypeError(f"unknown kernel arg {a!r}")
        fn = m.functions[ep.fn]
        res = fn(*args)
        outs = [res] if fn.n_out() == 1 else list(res)
        for i, w in enumerate(ep.writes):
            fld = m.state.field(w)
            arr = np.asarray(outs[i], dtype=float)
            self._state[w] = (arr.reshape(fld.shape) if fld.kind == "matrix"
                              else arr.reshape(-1))
        ret: dict[str, np.ndarray] = {}
        for name, o in zip(ep.returns, outs[len(ep.writes):]):
            port = m.port(name)
            a = np.asarray(o, dtype=float)
            ret[name] = (a.reshape(port.shape) if len(port.shape) == 2
                         else a.reshape(-1))
        return ret

    # ------------------------------------------------------------------
    # Shared metadata helpers (all from the Module)
    # ------------------------------------------------------------------

    def _u_fields(self):
        return self._u_port.fields if self._u_port is not None else ()

    def _input_names(self) -> list[str]:
        return [f.name for f in self._u_fields()]

    def build_u(self, u: dict[str, Any] | None) -> np.ndarray:
        """Resolve a `{name: value}` dict (full or suffix names) to the
        flat control vector over the Module's declared defaults."""
        fields = self._u_fields()
        out = np.array([float(np.asarray(f.default).ravel()[0])
                        for f in fields])
        if u:
            names = self._input_names()
            idx = {n: i for i, n in enumerate(names)}
            for k, v in u.items():
                full = resolve_suffix(k, names, label="input",
                                      who=type(self).__name__)
                out[idx[full]] = float(np.asarray(v).ravel()[0])
        return out

    @property
    def spec(self):
        return self._spec

    @property
    def input_names(self) -> list[str]:
        """The Module's declared control-input names, in order."""
        return self._input_names()

    # ==================================================================
    # Sim surface (THREADED oracle: a `step` entry consuming NOISE)
    # ==================================================================

    def initial_state(self) -> dict[str, dict[str, Any]]:
        """Fresh nested initial state: the manifold slots' defaults plus
        input/noise placeholder entries (commands you may set; noise seeds
        that stay at zero — a `NoiseDriver` draw never enters the dict)."""
        spec = self._spec
        x0 = self.module.state.field("x").init
        nested: dict[str, dict[str, Any]] = {}
        for full, val in spec.unpack(np.asarray(x0, dtype=float)).items():
            owner, slot = _split(full)
            nested.setdefault(owner, {})[slot] = val
        for f in self._u_fields():
            owner, rest = _split(f.name)
            nested.setdefault(owner, {}).setdefault(rest, f.default)
        if self._noise_port is not None:
            for f in self._noise_port.fields:
                owner, rest = _split(f.name)
                nested.setdefault(owner, {}).setdefault(
                    rest, np.zeros(f.dim) if f.dim > 1 else 0.0)
        return nested

    def step(self, state=None, dt=None, t: float | None = None, **inputs):
        """Polymorphic step, by Module shape:

        * Sim (threaded oracle) — functional `step(state, dt, t=0)` → next
          state dict (readings via `outputs()`), or bus `step(dt[, t=…])`
          advancing the held `self.state`.
        * Filter (held + measurements) — `step(dt[, t=…[, Q=…]])`: fold
          fresh measurements, then predict (the measurement bus).
        * Recurrence (held + readouts) — `step(dt, **inputs)` → readouts.
        """
        if self._bus is not None:                      # filter
            return self._bus.step(dt if dt is not None else state,
                                  t=t, **inputs)
        if self.module.hosting is Hosting.HELD:        # recurrence
            return self._recurrence_step(state, t=t, **inputs)
        if isinstance(state, dict):                    # sim, functional
            return self._functional_step(state, dt,
                                         0.0 if t is None else t)
        bus_dt = dt if state is None else state        # sim, bus form
        if bus_dt is None:
            raise TypeError(f"{type(self).__name__}.step: provide dt")
        return self._sim_bus_step(float(bus_dt), t)

    def _functional_step(self, state: dict, dt: float,
                         t: float = 0.0) -> dict[str, dict[str, Any]]:
        spec = self._spec
        flat: dict[str, Any] = {}
        for owner, slots in state.items():
            for slot, val in slots.items():
                flat[f"{owner}.{slot}"] = val
        self._state["x"] = spec.pack(
            {k: v for k, v in flat.items() if k in spec})
        u = np.array([float(np.asarray(flat.get(f.name, f.default)).ravel()[0])
                      for f in self._u_fields()])
        readings = self._run(self.module.entry("step"),
                             {"u": u, "noise": self._noise_vec(flat),
                              "dt": dt, "t": t})
        new_state: dict[str, dict[str, Any]] = {k: {} for k in state}
        for full, val in spec.unpack(self._state["x"]).items():
            owner, slot = _split(full)
            new_state.setdefault(owner, {})[slot] = val
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

    @property
    def state(self):
        """Held state. Sim bus mode: the nested dict (lazy-seeded, mutate
        in place). Recurrence: `{slot: value}` by name."""
        if self.module.hosting is Hosting.HELD:
            return self._spec.unpack(self._state["x"])
        if self._bus_state is None:
            self._bus_state = self.initial_state()
        return self._bus_state

    @state.setter
    def state(self, value: dict) -> None:
        if self.module.hosting is Hosting.HELD:
            raise AttributeError("held state: use reset()")
        self._bus_state = value

    def _sim_bus_step(self, dt: float, t: float | None):
        t0 = self._t if t is None else t
        st = self.state
        for full, v in self._ports.pull(t0).items():
            owner, rest = _split(full)
            st.setdefault(owner, {})[rest] = v
        new = self._functional_step(st, dt, t0)
        self._bus_state = new
        self._t = t0 + dt
        readings = {f"{o}.{s}": v for o, slots in self._outputs.items()
                    for s, v in slots.items()}
        self._ports.publish(readings, t0)
        return new

    def attach_driver(self, driver: "NoiseDriver") -> "NoiseDriver":
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
    def driver(self) -> "NoiseDriver | None":
        return self._driver

    def out(self, name: str):
        """Producer port for a sensor reading (rate-gated sample-and-hold;
        published each bus step)."""
        full = resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="output", who=type(self).__name__)
        port = self.module.port(full)
        return self._ports.producer(full, dim=port.size, rate=port.rate)

    def command(self, name: str):
        """Port for a control input. On a sim/filter: a latched (ZOH)
        consumer pulled before each step. On a regulator: the producer
        `compute()` publishes to."""
        if self._bus is not None:
            return self._bus.command(name)
        full = resolve_suffix(name, self._input_names(), label="input",
                              who=type(self).__name__)
        if "control" in self._methods:                 # regulator: producer
            return self._ports.producer(full, dim=1)
        f = next(f for f in self._u_fields() if f.name == full)
        return self._ports.consumer(full, dim=f.dim, rate=f.rate)

    # ==================================================================
    # Filter surface (HELD + MEASUREMENT ports)
    # ==================================================================

    @property
    def x(self) -> np.ndarray:
        return self._state["x"].copy()

    @property
    def P(self) -> np.ndarray:
        return self._state["P"].copy()

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner."""
        nested: dict[str, dict[str, Any]] = {}
        for full, val in self._spec.unpack(self._state["x"]).items():
            owner, slot = _split(full)
            nested.setdefault(owner, {})[slot] = val
        return nested

    def reset(self, state: dict | None = None, *,
              P: np.ndarray | None = None) -> None:
        """Reset held state. `state` is a nested/flat dict merged over
        the Module's declared initial values, or a flat ambient vector
        taken verbatim; `P` resets the covariance."""
        x_field = self.module.state.field("x")
        if isinstance(state, np.ndarray):
            self._state["x"] = np.asarray(state, dtype=float).reshape(-1).copy()
        elif state is not None:
            base = self._spec.unpack(np.asarray(x_field.init, dtype=float))
            for k, v in state.items():
                if isinstance(v, dict):
                    for slot, val in v.items():
                        base[f"{k}.{slot}"] = val
                else:
                    base[k] = v
            self._state["x"] = self._spec.pack(
                {k: v for k, v in base.items() if k in self._spec})
        elif P is None:
            self._state["x"] = np.asarray(
                x_field.init, dtype=float).reshape(-1).copy()
            if "P" in self.module.state:
                pf = self.module.state.field("P")
                self._state["P"] = np.asarray(
                    pf.init, dtype=float).reshape(pf.shape).copy()
            if self._y is not None:
                self._y = np.zeros(self._y_port.size)
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (self._spec.tangent_dim, self._spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"reset: P shape {P.shape} doesn't match tangent dim "
                    f"{expected}")
            self._state["P"] = P.copy()
        if self._bus is not None:
            self._bus.publish_estimate()

    def predict(self, dt: float, *, t: float = 0.0,
                u: dict[str, Any] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the estimate by `dt`. Process noise: the model's baked
        `L Σ Lᵀ` unless an explicit `Q` overrides it."""
        u_vec = self.build_u(u)
        if Q is None:
            self._run(self.module.entry("predict"),
                      {"u": u_vec, "dt": dt, "t": t})
        else:
            self._run(self.module.entry("predict_with_Q"),
                      {"Q": np.asarray(Q, dtype=float), "u": u_vec,
                       "dt": dt, "t": t})

    def update(self, target, z=None, R=None, *, t: float = 0.0,
               u: dict[str, Any] | None = None) -> None:
        """Fold one measurement.

        * `update("gps.position", z)` — by sensor name (full or suffix),
          through the baked Joseph-update kernel.
        * `update(h_sym, z, R=R)` — a caller-supplied `h(x)` callable +
          measurement covariance (custom measurements; numpy-only).
        """
        if callable(target):
            if z is None or R is None:
                raise TypeError("update(h_sym, z, R=...): z and R required")
            return self._update_low_level(target, z, R)
        self.fold(self._resolve_sensor(target),
                  z, self.build_u(u if u is not None else
                                  (self.inputs if self.inputs else None)),
                  t=t)

    def _resolve_sensor(self, name: str) -> str:
        return resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="sensor", who=type(self).__name__)

    def fold(self, full: str, z, u_vec: np.ndarray, *,
             t: float = 0.0) -> None:
        """Fold one measurement (full sensor name) — the bus's hook."""
        port = self.module.port(full)
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != port.size:
            raise ValueError(
                f"update {full}: expected z of size {port.size}, got "
                f"{z_arr.size}.")
        self._run(self.module.entry(f"update_{full.replace('.', '_')}"),
                  {full: z_arr, "u": u_vec, "t": t})

    @property
    def estimate_vector(self) -> np.ndarray:
        return self._state["x"]

    def _update_low_level(self, h_sym: Callable, z, R) -> None:
        """Joseph update for a caller-supplied `h(x)` — built on the spec's
        manifold ops; the one genuinely runtime-defined measurement."""
        spec = self._spec
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"update: R shape {R.shape} doesn't match z size {z.size}")
        x_sym = ca.MX.sym("x", spec.ambient_dim, 1)
        h_mx = h_sym(x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta", spec.tangent_dim, 1)
        h_pert = h_sym(spec.boxplus_sym(x_sym, delta))
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(ca.jacobian(h_pert, delta), delta,
                              ca.MX.zeros(spec.tangent_dim, 1))
        x_now = self._state["x"]
        h_x = np.asarray(ca.Function("h", [x_sym], [h_mx])(x_now)).reshape(-1)
        H = np.asarray(ca.Function("H", [x_sym], [H_sym])(x_now))
        if h_x.size != z.size:
            raise ValueError(
                f"update: h(x) size {h_x.size} doesn't match z size {z.size}")
        P = self._state["P"]
        S = H @ P @ H.T + R
        K = np.linalg.solve(S.T, (P @ H.T).T).T
        self._state["x"] = spec.boxplus_num(x_now, K @ (z - h_x))
        IKH = np.eye(spec.tangent_dim) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T
        self._state["P"] = 0.5 * (P + P.T)

    # bus passthroughs (filter only)
    @property
    def inputs(self) -> dict:
        return self._bus.inputs if self._bus is not None else {}

    @inputs.setter
    def inputs(self, value: dict) -> None:
        self._bus.inputs = value

    @property
    def Q(self):
        return self._bus.Q

    @Q.setter
    def Q(self, value) -> None:
        self._bus.Q = value

    def feed(self, name: str, z, *, t: float | None = None) -> None:
        self._bus.feed(name, z, t=t)

    def meas(self, name: str):
        return self._bus.meas(name)

    @property
    def estimate(self):
        """Producer port: the estimate as a flat ambient vector (filter),
        or the consumer port a regulator reads."""
        if self._bus is not None:
            return self._bus.estimate
        from ...signal import Signal
        if getattr(self, "_est_in", None) is None:
            self._est_in = Signal("estimate", dim=None)
        return self._est_in

    # ==================================================================
    # Recurrence surface (HELD + an OUTPUT port)
    # ==================================================================

    def _recurrence_step(self, dt, *, t=None, **inputs):
        tt = self._t if t is None else t
        u = np.zeros(self._u_port.size)
        off = 0
        for f in self._u_fields():
            if f.name not in inputs:
                raise KeyError(
                    f"step: missing input {f.name!r}; required: "
                    f"{self._input_names()}")
            v = np.atleast_1d(np.asarray(inputs[f.name],
                                         dtype=float)).reshape(-1)
            if v.size != f.dim:
                raise ValueError(
                    f"step: input {f.name!r} expects dim {f.dim}, got "
                    f"{v.size}.")
            u[off:off + f.dim] = v
            off += f.dim
        ret = self._run(self.module.entry("step"),
                        {"u": u, "dt": dt, "t": tt})
        self._y = ret[self._y_port.name]
        self._t = tt + dt
        return self.readouts()

    def readouts(self) -> dict[str, Any]:
        """Last-computed readouts by output-field name (scalars unwrapped)."""
        out: dict[str, Any] = {}
        off = 0
        for f in self._y_port.fields:
            seg = self._y[off:off + f.dim]
            out[f.name] = float(seg[0]) if f.dim == 1 else seg.copy()
            off += f.dim
        return out

    def input(self, name: str):
        """Consumer port for a recurrence input (latched / ZOH)."""
        names = self._input_names()
        if name not in names:
            raise KeyError(f"unknown input port {name!r}. Available: {names}")
        f = next(f for f in self._u_fields() if f.name == name)
        return self._ports.consumer(name, dim=f.dim)

    def output(self, name: str):
        """Producer port for a recurrence readout."""
        names = [f.name for f in (self._y_port.fields if self._y_port else ())]
        if name not in names:
            raise KeyError(f"unknown output port {name!r}. Available: {names}")
        f = next(f for f in self._y_port.fields if f.name == name)
        return self._ports.producer(name, dim=f.dim)

    # ==================================================================
    # Regulator surface (THREADED + a `control` entry)
    # ==================================================================

    def u(self, x_flat) -> np.ndarray:
        """Control vector for a flat ambient state (full-spec layout)."""
        return self.call("control",
                         {"x": np.asarray(x_flat, dtype=float)})["u"]

    def control(self, state: dict) -> dict[str, float]:
        """Map a state estimate (nested or flat dict) → `{input: value}`,
        merged over the Module's reference point."""
        spec = self._x_port.manifold
        base = spec.unpack(np.asarray(self._x_port.init, dtype=float))
        for k, v in state.items():
            if isinstance(v, dict):
                for slot, val in v.items():
                    base[f"{k}.{slot}"] = val
            else:
                base[k] = v
        x = spec.pack({k: v for k, v in base.items() if k in spec})
        u_vec = self.u(x)
        return {n: float(u_vec[i])
                for i, n in enumerate(self._input_names())}

    def compute(self, dt: float | None = None, *,
                t: float | None = None) -> dict[str, Any]:
        """Pull wired inputs, evaluate, publish to output ports.

        Regulator: read the wired `estimate`, apply the control law,
        publish each command. Recurrence: pull input ports, `step(dt)`,
        publish readouts (stamped at start-of-step)."""
        if "control" in self._methods:
            est = self.estimate.read()
            if est is None:
                raise RuntimeError(
                    "compute: estimate input is empty — wire a producer "
                    "into `.estimate` (or set it) first.")
            if isinstance(est, dict):
                u = self.control(est)
            else:
                self._ensure_gather()
                x_full = np.asarray(self._x_port.init, dtype=float).copy()
                x_src = np.asarray(est, dtype=float).reshape(-1)
                for foff, dim, soff in self._gather:
                    x_full[foff:foff + dim] = x_src[soff:soff + dim]
                u_vec = self.u(x_full)
                u = {n: float(u_vec[i])
                     for i, n in enumerate(self._input_names())}
            for full, v in u.items():
                self.command(full).set(v)
            return u
        tt = self._t if t is None else t
        out = self._recurrence_step(dt, t=tt, **self._ports.pull(tt))
        self._ports.publish(out, t=tt)        # start-of-step stamp
        return out

    def _ensure_gather(self) -> None:
        if getattr(self, "_gather", None) is not None:
            return
        spec = self._x_port.manifold
        src = self.estimate.source
        layout = getattr(src, "layout", None) if src is not None else None
        if layout is None:
            raise RuntimeError(
                "compute: wired estimate carries no layout; wire a filter's "
                "`estimate` port (a flat vector + layout) or set a dict.")
        self._gather = [
            (spec.slot(name).ambient_offset, sdim, soff)
            for name, (soff, sdim) in layout.items() if name in spec]

    # ------------------------------------------------------------------

    def observability(self, **kwargs):       # pragma: no cover - thin
        raise AttributeError(
            "observability is an IR-level analysis — call it on the EKF "
            "transform: EKF(world, ...).observability(...)")

    def __repr__(self) -> str:
        drv = "" if self._driver is None else f" +{self._driver!r}"
        return f"<NumpyRuntime over {self.module!r}{drv}>"


# ---------------------------------------------------------------------------
# NoiseDriver — stochastic source for a Module's NOISE port
# ---------------------------------------------------------------------------

class NoiseDriver:
    """Draws the per-step samples that make an oracle Module's noise live.

    Binds to the NOISE port's fields (name, dim, σ); each `sample()` is an
    independent `N(0, σ²)` draw per active channel. Deliberately thin and
    swappable — kept out of the pure kernels — and simply omitted on a
    deploy target.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._channels: list[tuple[str, int, float]] = []

    def bind(self, fields) -> None:
        self._channels = [(f.name, f.dim, float(f.sigma or 0.0))
                          for f in fields]

    def sample(self) -> dict[str, np.ndarray]:
        return {name: self._rng.normal(0.0, sigma, dim)
                for name, dim, sigma in self._channels if sigma > 0.0}

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def __repr__(self) -> str:
        active = sum(1 for _, _, s in self._channels if s > 0.0)
        return f"<NoiseDriver seed={self._seed} active={active}>"


# ---------------------------------------------------------------------------
# TargetNumpy
# ---------------------------------------------------------------------------

def TargetNumpy(x) -> NumpyRuntime:
    """Lower a typed `Module` — or any transform exposing `.module()`
    (`Sim`, `EKF`, `LQR`, a recurrence block) — to the one native-Python
    `NumpyRuntime`."""
    from ..target import as_module
    return NumpyRuntime(as_module(x, "TargetNumpy"))
