"""TargetNumpy — the native-Python backend.

ONE kernel engine + four thin typed views. `NumpyRuntime` is the engine:
the generic typed-arg gather → `ca.Function` call → scatter over a
Module's entry points (`call(method, values)`), plus the shared port
metadata helpers. Each view subclasses it and exposes exactly the surface
its Module shape implies — nothing else:

  * `NumpySim`        — THREADED + an oracle ``step`` entry. Held state:
                        `sim.state` (nested dict), `step(dt)`,
                        `outputs()`, `out`/`command` ports,
                        `attach_driver` (feeds the NOISE port).
  * `NumpyFilter`     — HELD + a ``predict`` entry. `predict`/`update`
                        (by sensor *name*), `feed`+`step` (the
                        measurement bus), `x`/`P`, `reset`, `state_dict`,
                        `meas`/`command`/`estimate` ports.
  * `NumpyRecurrence` — HELD + an OUTPUT port. `step(dt, **inputs)`,
                        `readouts()`, `reset`, `input`/`output` ports,
                        `compute`.
  * `NumpyRegulator`  — THREADED + a ``control`` entry. `u(x)`,
                        `control(state_dict)`, `estimate`/`command`
                        ports, `compute`.

`TargetNumpy(x)` inspects the Module once and returns the matching view;
a Module that matches no view (e.g. the Sim's noiseless deploy bundle)
comes back as the bare engine — `call()` works on any entry point.

Everything is driven by the IR's types: slot names come from the manifold
spec, input/sensor/noise names + defaults/σ/rates from the Ports. The
backend never mentions a transform.
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from ...bus import MeasurementBus, PortSet
from ...ir.module import Hosting, Module, PortRef, Role, StateRef
from ...ir.state_spec import flatten_nested
from ...linearized_system import resolve_suffix


def _split(full: str) -> tuple[str, str]:
    owner, rest = full.split(".", 1)
    return owner, rest


# ---------------------------------------------------------------------------
# The kernel engine
# ---------------------------------------------------------------------------

class NumpyRuntime:
    """The generic engine over a typed `Module`: state storage + the
    typed-arg gather → kernel call → scatter. Views subclass it."""

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
        self._ports = PortSet()

    # ---- kernel engine (typed-arg gather → call → scatter) -----------

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

    # ---- shared metadata helpers (all from the Module) ----------------

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

    def observability(self, **kwargs):       # pragma: no cover - thin
        raise AttributeError(
            "observability is an IR-level analysis — call it on the EKF "
            "transform: EKF(world, ...).observability(...)")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} over {self.module!r}>"


# ---------------------------------------------------------------------------
# Sim view — THREADED oracle, held by the runtime (bus-driven)
# ---------------------------------------------------------------------------

class NumpySim(NumpyRuntime):
    """The simulation oracle. The runtime holds the nested state dict
    (`sim.state`); `step(dt)` advances it and realizes that step's sensor
    readings (`outputs()`), pulling wired command ports first and
    publishing readings to wired output ports after."""

    def __init__(self, module: Module) -> None:
        super().__init__(module)
        self._driver: "NoiseDriver | None" = None
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


# ---------------------------------------------------------------------------
# Filter view — HELD + a predict entry
# ---------------------------------------------------------------------------

class NumpyFilter(NumpyRuntime):
    """A predict/update filter. Held `x`/`P`; baked per-sensor updates by
    NAME; the measurement bus (`feed` + `step`, wired `meas`/`command`/
    `estimate` ports) on top."""

    def __init__(self, module: Module) -> None:
        super().__init__(module)
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

    # ---- estimate access -------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._state["x"].copy()

    @property
    def P(self) -> np.ndarray:
        return self._state["P"].copy()

    @property
    def estimate_vector(self) -> np.ndarray:
        return self._state["x"]

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner."""
        return self._spec.to_nested(self._state["x"])

    def reset(self, state: dict | None = None, *,
              P: np.ndarray | None = None) -> None:
        """Reset held state. `state` is a nested/flat dict merged over
        the Module's declared initial values, or a flat ambient vector
        taken verbatim; `P` resets the covariance."""
        x_field = self.module.state.field("x")
        if state is not None:
            self._state["x"] = self._spec.pack_any(state, base=x_field.init)
        elif P is None:
            self._state["x"] = np.asarray(
                x_field.init, dtype=float).reshape(-1).copy()
            pf = self.module.state.field("P")
            self._state["P"] = np.asarray(
                pf.init, dtype=float).reshape(pf.shape).copy()
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (self._spec.tangent_dim, self._spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"reset: P shape {P.shape} doesn't match tangent dim "
                    f"{expected}")
            self._state["P"] = P.copy()
        self._bus.publish_estimate()

    # ---- predict / update ------------------------------------------------

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

    # ---- the measurement bus ----------------------------------------------

    def step(self, dt: float, *, t: float | None = None,
             Q: np.ndarray | None = None) -> None:
        """Fold fresh measurements (interval start), then predict by `dt`."""
        return self._bus.step(dt, t=t, Q=Q)

    def feed(self, name: str, z, *, t: float | None = None) -> None:
        self._bus.feed(name, z, t=t)

    def meas(self, name: str):
        return self._bus.meas(name)

    def command(self, name: str):
        return self._bus.command(name)

    @property
    def estimate(self):
        """Producer port: the estimate as a flat ambient vector."""
        return self._bus.estimate

    @property
    def inputs(self) -> dict:
        return self._bus.inputs

    @inputs.setter
    def inputs(self, value: dict) -> None:
        self._bus.inputs = value

    @property
    def Q(self):
        return self._bus.Q

    @Q.setter
    def Q(self, value) -> None:
        self._bus.Q = value


# ---------------------------------------------------------------------------
# Recurrence view — HELD + an OUTPUT port
# ---------------------------------------------------------------------------

class NumpyRecurrence(NumpyRuntime):
    """A stateful dataflow block (PID, Madgwick, …): `step(dt, **inputs)`
    advances the held state and computes the readouts."""

    def __init__(self, module: Module) -> None:
        super().__init__(module)
        self._y = np.zeros(self._y_port.size)

    @property
    def state(self) -> dict[str, Any]:
        """Held state, `{slot: value}` by name."""
        return self._spec.unpack(self._state["x"])

    def reset(self) -> None:
        """Reset the held state to the Module's declared initial values."""
        x_field = self.module.state.field("x")
        self._state["x"] = np.asarray(
            x_field.init, dtype=float).reshape(-1).copy()
        self._y = np.zeros(self._y_port.size)

    def step(self, dt: float, *, t: float | None = None,
             **inputs) -> dict[str, Any]:
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

    # ---- ports -----------------------------------------------------------

    def input(self, name: str):
        """Consumer port for a recurrence input (latched / ZOH)."""
        names = self._input_names()
        if name not in names:
            raise KeyError(f"unknown input port {name!r}. Available: {names}")
        f = next(f for f in self._u_fields() if f.name == name)
        return self._ports.consumer(name, dim=f.dim)

    def output(self, name: str):
        """Producer port for a recurrence readout."""
        names = [f.name for f in self._y_port.fields]
        if name not in names:
            raise KeyError(f"unknown output port {name!r}. Available: {names}")
        f = next(f for f in self._y_port.fields if f.name == name)
        return self._ports.producer(name, dim=f.dim)

    def compute(self, dt: float, *, t: float | None = None) -> dict[str, Any]:
        """Pull wired input ports, `step(dt)`, publish readouts (stamped
        at start-of-step)."""
        tt = self._t if t is None else t
        out = self.step(dt, t=tt, **self._ports.pull(tt))
        self._ports.publish(out, t=tt)
        return out


# ---------------------------------------------------------------------------
# Regulator view — THREADED + a control entry
# ---------------------------------------------------------------------------

class NumpyRegulator(NumpyRuntime):
    """A stateless control law: map a state estimate to commands.

    Holds the live reference point `x_ref` (seeded from the Module's
    built operating point); `retarget()` moves it at runtime."""

    def __init__(self, module: Module) -> None:
        super().__init__(module)
        self._est_in = None
        self._gather = None
        st = module.ports_by_role(Role.STATE)
        self._ref_port = st[1] if len(st) > 1 else None
        self._x_ref = (np.asarray(self._ref_port.init, dtype=float)
                       .reshape(-1).copy() if self._ref_port is not None
                       else np.asarray(self._x_port.init,
                                       dtype=float).reshape(-1).copy())

    def retarget(self, state: dict) -> None:
        """Move the reference point the law regulates to (nested or flat
        dict, merged over the CURRENT reference). The gain K is NOT
        re-solved: exact wherever the dynamics are invariant along the
        moved direction (e.g. translating a hover setpoint); build a new
        LQR for a genuinely different operating point (new A/B or trim).
        """
        if self._ref_port is None:
            raise AttributeError(
                f"{type(self).__name__}: module {self.module.name!r} has "
                "no reference port — its control law is not retargetable.")
        self._x_ref = self._ref_port.manifold.pack_any(state,
                                                       base=self._x_ref)

    @property
    def x_ref(self) -> np.ndarray:
        """The live reference point (flat ambient vector)."""
        return self._x_ref.copy()

    def u(self, x_flat) -> np.ndarray:
        """Control vector for a flat ambient state (full-spec layout)."""
        vals = {"x": np.asarray(x_flat, dtype=float)}
        if self._ref_port is not None:
            vals[self._ref_port.name] = self._x_ref
        return self.call("control", vals)["u"]

    def control(self, state: dict) -> dict[str, float]:
        """Map a state estimate (nested or flat dict) → `{input: value}`,
        merged over the live reference point (unsupplied slots sit at
        the reference, i.e. zero error)."""
        x = self._x_port.manifold.pack_any(state, base=self._x_ref)
        u_vec = self.u(x)
        return {n: float(u_vec[i])
                for i, n in enumerate(self._input_names())}

    # ---- ports -----------------------------------------------------------

    def command(self, name: str):
        """Producer port `compute()` publishes this command to."""
        full = resolve_suffix(name, self._input_names(), label="input",
                              who=type(self).__name__)
        return self._ports.producer(full, dim=1)

    @property
    def estimate(self):
        """Consumer port: wire a filter's `estimate` (flat vector +
        layout) into it, or `set()` a state dict directly."""
        from ...signal import Signal
        if self._est_in is None:
            self._est_in = Signal("estimate", dim=None)
        return self._est_in

    def compute(self) -> dict[str, Any]:
        """Read the wired `estimate`, apply the control law, publish each
        command to its port. Returns `{input: value}`."""
        est = self.estimate.read()
        if est is None:
            raise RuntimeError(
                "compute: estimate input is empty — wire a producer "
                "into `.estimate` (or set it) first.")
        if isinstance(est, dict):
            u = self.control(est)
        else:
            self._ensure_gather()
            x_full = self._x_ref.copy()
            x_src = np.asarray(est, dtype=float).reshape(-1)
            for foff, dim, soff in self._gather:
                x_full[foff:foff + dim] = x_src[soff:soff + dim]
            u_vec = self.u(x_full)
            u = {n: float(u_vec[i])
                 for i, n in enumerate(self._input_names())}
        for full, v in u.items():
            self.command(full).set(v)
        return u

    def _ensure_gather(self) -> None:
        if self._gather is not None:
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
    (`Sim`, `EKF`, `LQR`, a recurrence block) — to the matching
    native-Python view (sim / filter / recurrence / regulator), or the
    bare kernel engine when no view matches."""
    from ..target import as_module
    m = as_module(x, "TargetNumpy")
    methods = {e.method for e in m.entry_points}
    if m.hosting is Hosting.HELD:
        if "predict" in methods:
            return NumpyFilter(m)
        if m.ports_by_role(Role.OUTPUT):
            return NumpyRecurrence(m)
    else:
        if "control" in methods:
            return NumpyRegulator(m)
        if "step" in methods:
            return NumpySim(m)
    return NumpyRuntime(m)
