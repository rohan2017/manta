"""TargetNumpy — native-Python runtime evaluator backend.

`TargetNumpy(ir)` dispatches by IR type:

  * `TargetNumpy(Sim)` → `NumpyWorld` with `.step()` and
    `.initial_state()`.
  * `TargetNumpy(EKF)`           → `NumpyEKF` with `.predict()`,
    `.update()`, `.state_dict()`, `.reset()`, `.x`, `.P`.

The IR objects themselves (`Sim`, `EKF`) describe the
compiled symbolic graph but are not directly callable for ticking /
predicting — choose a target to get the runtime.
"""

from __future__ import annotations

from typing import Any, Callable

import casadi as ca
import numpy as np

from ..target import Target

# Output-shape → vector dimension, for sizing sensor-output ports.
_SHAPE_DIM = {"scalar": 1, "vec3": 3, "vec4": 4}


# ---------------------------------------------------------------------------
# NumpyWorld — runtime for Sim
# ---------------------------------------------------------------------------

class NumpyWorld:
    """Native-Python evaluator wrapping a `Sim` IR.

    Provides the user-facing tick API: `initial_state()` and
    `step(state, dt, t=0)`. Internally calls the IR's CasADi function
    after translating between the user's nested state dict and the
    flat-prefixed casadi-input names.
    """

    def __init__(self, cw) -> None:
        from ...sim import Sim
        if not isinstance(cw, Sim):
            raise TypeError(
                f"NumpyWorld: expected Sim, got "
                f"{type(cw).__name__}")
        self._cw = cw
        # Optional stochastic noise source (see attach_driver). None ⇒
        # the sim is a noiseless oracle: Noise inputs stay at their
        # zero seed, so a sensor reading equals its mean model.
        self._driver = None
        # Signal-bus plumbing (lazy). `_bus_state` is the held state for
        # the no-arg `step(dt)` form; `_out_ports` publish sensor
        # readings; `_cmd_in_ports` receive actuator commands.
        self._bus_state: dict | None = None
        self._bus_t = 0.0
        self._out_ports: dict[str, Any] = {}
        self._cmd_in_ports: dict[str, Any] = {}
        self._sig = None

    # ---- IR passthroughs ----------------------------------------------

    @property
    def world(self):
        return self._cw.world

    @property
    def crafts(self):
        return self._cw.crafts

    @property
    def tick(self):
        return self._cw.tick

    # ---- Noise driver --------------------------------------------------

    def attach_driver(self, driver: "NoiseDriver") -> "NoiseDriver":
        """Attach a stochastic `NoiseDriver` so `step()` injects sensor +
        process noise per the model's `Noise` channels.

        Without a driver the sim is a noiseless oracle (the on-device
        default — there you run against real sensors). With one, every
        active (σ>0) Noise channel is sampled each step and fed into its
        tick input, so truth is genuinely noisy *and* matches the very σ
        the EKF uses to size R/Q. Returns the driver for one-liners::

            sim.attach_driver(NoiseDriver(seed=7))
        """
        from ...estimation.state_spec import StateSpec
        from ...tick_signature import walk_tick_signature
        cf  = self._cw.tick.casadi_function
        sig = walk_tick_signature(
            cf, self._cw.world, StateSpec.from_world(self._cw.world))
        driver.bind(sig.noise)
        self._driver = driver
        return driver

    @property
    def driver(self) -> "NoiseDriver | None":
        return self._driver

    # ---- State + step --------------------------------------------------

    def initial_state(self) -> dict[str, dict[str, Any]]:
        """Fresh copy of the per-owner initial state."""
        import copy
        return copy.deepcopy(self._cw._initial)

    def step(self, state=None, dt=None, t: float | None = None):
        """Advance the world by `dt`. Returns a fresh state dict.

        Two call forms:

          * **functional** — `step(state, dt, t=0)`: pure, returns a new
            state dict; nothing is held on the runtime. `state` may be a
            craft or a field disturbance keyed by `<owner>.<slot>`.
          * **bus** — `step(dt)` / `step(dt, t=...)`: advances the
            runtime's *held* state (`sim.state`), pulling any wired
            command ports in first and publishing sensor readings to the
            wired output ports after. Use with `wire(...)`.

        `t` is the world-clock time at the start of this step. In bus
        mode, omitting it advances the runtime's internal clock by `dt`
        each call (so published measurements carry an honest, increasing
        timestamp the estimator's staleness check can trust).
        """
        if isinstance(state, dict):
            return self._functional_step(state, dt, 0.0 if t is None else t)
        bus_dt = dt if state is None else state
        if bus_dt is None:
            raise TypeError("NumpyWorld.step: provide dt")
        return self._bus_step(float(bus_dt), t)

    def _functional_step(self,
                         state: dict[str, dict[str, Any]],
                         dt: float,
                         t: float = 0.0) -> dict[str, dict[str, Any]]:
        flat: dict[str, Any] = {"dt": dt, "t": t}
        for owner_name, owner_state in state.items():
            for slot, val in owner_state.items():
                flat[f"{owner_name}.{slot}"] = val

        # Overlay fresh stochastic draws onto the (zero-seeded) Noise
        # inputs. The samples live only in this tick's `flat` — they are
        # never written back to `state`, so each step draws afresh.
        if self._driver is not None:
            for name, sample in self._driver.sample().items():
                flat[name] = sample

        out = self._cw.tick(**flat)

        new_state: dict[str, dict[str, Any]] = {k: {} for k in state}
        for key, val in out.items():
            owner_name, slot = key.split(".", 1)
            if owner_name not in new_state:
                new_state[owner_name] = {}
            new_state[owner_name][slot] = val
        # Preserve input-only slots the tick doesn't write back.
        for owner_name, owner_state in state.items():
            for slot, val in owner_state.items():
                if slot not in new_state[owner_name]:
                    new_state[owner_name][slot] = val
        return new_state

    # ---- Signal-bus ports + held-state stepping ------------------------

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        """The held state for bus-mode `step(dt)` — lazily seeded from
        `initial_state()`. Returned by reference, so you can mutate it
        in place (e.g. `sim.state["c"]["position"] = ...`)."""
        if self._bus_state is None:
            self._bus_state = self.initial_state()
        return self._bus_state

    @state.setter
    def state(self, value: dict) -> None:
        self._bus_state = value

    def out(self, name: str):
        """Producer port for a sensor Output (`"craft.part.output"` or an
        unambiguous suffix). Published each bus `step()` — at the part's
        declared sample rate (`ctx.sample`) if any, else every tick."""
        from ...signal import Signal
        full = self._resolve(name, self._signature().sensor_names, "output")
        if full not in self._out_ports:
            self._out_ports[full] = Signal(
                full, dim=self._output_dim(full), rate=self._rate(full))
        return self._out_ports[full]

    def command(self, name: str):
        """Consumer port for an actuator Input. Pulled into the held
        state before each bus `step()` (latched / zero-order-hold) — at
        the part's declared intake rate (`ctx.hold`) if any."""
        from ...signal import Signal
        full = self._resolve(name, self._signature().input_names, "input")
        if full not in self._cmd_in_ports:
            self._cmd_in_ports[full] = Signal(
                full, dim=self._input_dim(full), latched=True,
                rate=self._rate(full))
        return self._cmd_in_ports[full]

    def _rate(self, full: str):
        return getattr(self._cw.tick, "sample_rates", {}).get(full)

    def _bus_step(self, dt: float, t: float | None) -> dict[str, dict[str, Any]]:
        t0 = self._bus_t if t is None else t  # start-of-step world time
        st = self.state                       # lazy-seed the held state
        # Pull wired command ports into the held state (ZOH, rate-gated).
        for full, port in self._cmd_in_ports.items():
            v = port.latched_value(t0)
            if v is not None:
                owner, rest = full.split(".", 1)
                st.setdefault(owner, {})[rest] = v
        new = self._functional_step(st, dt, t0)
        self._bus_state = new
        self._bus_t = t0 + dt
        # Publish sensor readings, stamped at the start-of-step time (the
        # instant the reading observes), to the wired output ports —
        # gated by each port's sample rate (sample-and-hold in between).
        for full, port in self._out_ports.items():
            owner, slot = full.split(".", 1)
            if owner in new and slot in new[owner] and port.due(t0):
                port.set(np.asarray(new[owner][slot]).ravel(), t=t0)
        return new

    # ---- Port resolution helpers --------------------------------------

    def _signature(self):
        if self._sig is None:
            from ...estimation.state_spec import StateSpec
            from ...tick_signature import walk_tick_signature
            cf = self._cw.tick.casadi_function
            self._sig = walk_tick_signature(
                cf, self._cw.world, StateSpec.from_world(self._cw.world))
        return self._sig

    @staticmethod
    def _resolve(name: str, candidates: list[str], label: str) -> str:
        if name in candidates:
            return name
        matches = [c for c in candidates if c.endswith("." + name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"NumpyWorld: ambiguous {label} {name!r}; matches {matches}. "
                f"Use the fully-qualified form.")
        raise KeyError(
            f"NumpyWorld: unknown {label} {name!r}. Available: "
            f"{sorted(candidates)}")

    def _output_dim(self, full: str) -> int | None:
        for s in self._signature().sensors:
            if s.full == full:
                shape = s.part.output_declarations()[s.output_name].shape
                return _SHAPE_DIM.get(shape)
        return None

    def _input_dim(self, full: str) -> int | None:
        for ic in self._signature().inputs:
            if ic.full == full:
                d = ic.default
                return 1 if np.ndim(d) == 0 else int(np.size(d))
        return None

    def __repr__(self) -> str:
        drv = "" if self._driver is None else f" +{self._driver!r}"
        return f"<NumpyWorld over {self._cw!r}{drv}>"


# ---------------------------------------------------------------------------
# NoiseDriver — stochastic source for a Sim's Noise channels
# ---------------------------------------------------------------------------

class NoiseDriver:
    """Draws the random samples that make a `Sim` genuinely noisy.

    A model's `Noise` channels (WhiteNoise / RandomWalkNoise) only
    declare a *distribution*; the compiled tick exposes each as an
    ordinary input seeded to zero. Left alone, the sim is a noiseless
    oracle. Attach a driver and every active (σ>0) channel is sampled
    each `step()` and fed into its input — so the same σ that the EKF
    reads to size R/Q now also shapes the truth it estimates.

    Both kinds sample `N(0, σ²)` per component: WhiteNoise adds the draw
    straight into the reading; RandomWalkNoise feeds its `_driver` input
    and the kernel applies the √dt random-walk scaling. The driver is a
    deliberately thin, swappable layer (kept *out* of the pure kernel),
    so it lowers to a few lines in any target language and is simply
    omitted for on-device deployment.

    Usage::

        sim = TargetNumpy(Sim(world))
        sim.attach_driver(NoiseDriver(seed=7))
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng  = np.random.default_rng(seed)
        # (input_name, dim, sigma) per channel; filled by bind().
        self._channels: list[tuple[str, int, float]] = []

    def bind(self, channels) -> None:
        """Wire the driver to a list of `NoiseChannel` specs (from
        `walk_tick_signature`). Called by `NumpyWorld.attach_driver`."""
        self._channels = [(c.full, c.dim, c.sigma) for c in channels]

    def sample(self) -> dict[str, np.ndarray]:
        """One independent `N(0, σ²)` draw per active channel.

        Returns `{input_name: vec}`; inactive (σ=0) channels are omitted
        so they keep their zero seed."""
        out: dict[str, np.ndarray] = {}
        for name, dim, sigma in self._channels:
            if sigma > 0.0:
                out[name] = self._rng.normal(0.0, sigma, dim)
        return out

    def reset(self) -> None:
        """Re-seed to the construction seed for a reproducible rerun."""
        self._rng = np.random.default_rng(self._seed)

    def __repr__(self) -> str:
        active = sum(1 for _, _, s in self._channels if s > 0.0)
        return f"<NoiseDriver seed={self._seed} active={active}>"


# ---------------------------------------------------------------------------
# NumpyEKF — runtime for EKF (mutable state + predict/update)
# ---------------------------------------------------------------------------

class NumpyEKF:
    """Native-Python evaluator wrapping an `EKF` IR.

    Holds mutable `_x` (ambient state vector) and `_P` (tangent
    covariance). Calls the IR's compiled predict / measurement
    functions during `predict()` and `update()`.
    """

    def __init__(self, ekf) -> None:
        from ...estimation.ekf import EKF
        if not isinstance(ekf, EKF):
            raise TypeError(
                f"NumpyEKF: expected EKF, got {type(ekf).__name__}")
        self._ekf = ekf
        self._x   = self._initial_x()
        self._P   = np.eye(ekf.spec.tangent_dim) * 1e-2

        # Measurement bus: one mailbox per registered sensor, in
        # registration order (= the order step() applies them). The user
        # drops a reading in via feed(); step() consumes fresh ones.
        self._meas: dict[str, dict[str, Any]] = {}
        for key, spec_o in ekf._sensors.items():
            self._meas[spec_o["full"]] = {
                "value": None, "fresh": False, "t": None,
                "dim": spec_o["dim"], "key": key,
            }
        # Latched (zero-order-hold) control inputs, read by step()'s
        # predict. Full or craft-relative names; resolved per predict.
        self.inputs: dict[str, float] = {}
        # Filter clock — advances by dt each step (or to the supplied t).
        self._t = 0.0
        # Default process noise used by step() when none is passed.
        self.Q: np.ndarray | None = None
        # Signal-bus ports (lazy): measurement + command consumers and a
        # state-estimate producer. `_seen_meas` tracks the last upstream
        # version folded in, so a measurement is consumed once per fresh
        # sample (multi-rate safe).
        self._meas_ports: dict[str, Any] = {}
        self._cmd_ports:  dict[str, Any] = {}
        self._seen_meas:  dict[str, int] = {}
        self._est_port = None

    def _initial_x(self) -> np.ndarray:
        """Pack the world's nested initial-state dict into the spec's
        flat ambient layout."""
        ekf = self._ekf
        init_flat: dict[str, Any] = {}
        for owner_name, owner_state in ekf.world._initial_state_dict().items():
            for k, v in owner_state.items():
                init_flat[f"{owner_name}.{k}"] = v
        init_for_pack = {k: v for k, v in init_flat.items() if k in ekf.spec}
        return ekf.spec.pack(init_for_pack)

    # ---- IR passthroughs ----------------------------------------------

    @property
    def ekf(self):
        """The wrapped EKF IR (carries the symbolic functions,
        sensor table, etc.)."""
        return self._ekf

    @property
    def spec(self):
        return self._ekf.spec

    @property
    def world(self):
        return self._ekf.world

    @property
    def crafts(self):
        return self._ekf.crafts

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    # ---- State view + reset --------------------------------------------

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner — same shape as
        `NumpyWorld.initial_state()`.
        """
        flat = self._ekf.spec.unpack(self._x)
        nested: dict[str, dict[str, Any]] = {}
        for full_name, val in flat.items():
            owner_name, slot = full_name.split(".", 1)
            nested.setdefault(owner_name, {})[slot] = val
        return nested

    def reset(self, *,
              state: dict | None = None,
              P: np.ndarray | None = None) -> None:
        """Reset the EKF state and/or covariance.

        `state` accepts either:
          * Nested `{owner: {slot: value}}` (canonical, matches
            `state_dict()`).
          * Flat dict keyed by full slot name (`"<owner>.<slot>"`).
        Missing entries fall back to the world's add_craft defaults.
        """
        ekf = self._ekf
        if state is not None:
            full_flat: dict[str, Any] = {}
            for owner_name, owner_state in ekf.world._initial_state_dict().items():
                for k, v in owner_state.items():
                    full_flat[f"{owner_name}.{k}"] = v
            for k, v in state.items():
                if isinstance(v, dict):
                    for slot, val in v.items():
                        full_flat[f"{k}.{slot}"] = val
                else:
                    full_flat[k] = v
            full_flat = {k: v for k, v in full_flat.items() if k in ekf.spec}
            self._x = ekf.spec.pack(full_flat)
        if P is not None:
            P = np.asarray(P, dtype=float)
            expected = (ekf.spec.tangent_dim, ekf.spec.tangent_dim)
            if P.shape != expected:
                raise ValueError(
                    f"NumpyEKF.reset: P shape {P.shape} doesn't match "
                    f"tangent dim {expected}")
            self._P = P.copy()
        if self._est_port is not None:
            self._est_port.set(self._x.copy())

    # ---- Signal-bus ports ----------------------------------------------

    def meas(self, name: str):
        """Consumer port for a sensor measurement (`"craft.part.output"`
        or an unambiguous suffix). Wire a sim output into it, or `set()`
        it yourself; `step()` folds it in once per fresh sample."""
        from ...signal import Signal
        full = self._resolve_meas_name(name)
        if full not in self._meas_ports:
            self._meas_ports[full] = Signal(full, dim=self._meas[full]["dim"])
        return self._meas_ports[full]

    def command(self, name: str):
        """Consumer port for a known control input (drives the predict).
        Latched / zero-order-hold, gated at the part's declared intake
        rate (`ctx.hold`) so predict sees the same held command as truth."""
        from ...signal import Signal
        full = self._resolve_input_full(name)
        if full not in self._cmd_ports:
            rate = getattr(self._ekf, "_sample_rates", {}).get(full)
            self._cmd_ports[full] = Signal(full, latched=True, rate=rate)
        return self._cmd_ports[full]

    @property
    def estimate(self):
        """Producer port carrying the current state estimate as the flat
        ambient vector (EKF spec layout), refreshed after every
        `step()`/`reset()`. The port's `layout` ({slot: (offset, dim)})
        lets a consumer build a fixed name-keyed gather into its own
        layout — codegen-honest (a flat vector + a static index map, not
        a nested dict). Wire it into `lqr.estimate`."""
        from ...signal import Signal
        if self._est_port is None:
            self._est_port = Signal("estimate", dim=self.spec.ambient_dim)
            self._est_port.layout = {
                s.name: (s.offset, s.dim) for s in self.spec.slots}
            self._est_port.set(self._x.copy())
        return self._est_port

    def _resolve_input_full(self, name: str) -> str:
        names = self._ekf._input_names
        if name in names:
            return name
        matches = [n for n in names if n.endswith("." + name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"NumpyEKF.command: ambiguous input {name!r}; matches "
                f"{matches}. Use the fully-qualified form.")
        raise KeyError(
            f"NumpyEKF.command: unknown input {name!r}. Available: "
            f"{sorted(names)}")

    # ---- Predict -------------------------------------------------------

    def predict(self,
                dt: float,
                *,
                t: float = 0.0,
                u: dict[str, float] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the nominal state and tangent covariance by `dt`.

        Args mirror `EKF.predict` from the old monolithic class:
        `dt` (timestep), `t` (world-clock time), `u` (per-tick part
        Input override dict), `Q` (process-noise override; default is
        auto-assembly from registered Noise channels).

        The covariance is propagated per independent-subsystem block
        (`ekf._blocks`): with one block this is the usual dense
        `F P Fᵀ + Q`; with several (uncoupled crafts estimated jointly)
        each block propagates on its own — Σ O(n_b³) instead of O(n³).
        Off-diagonal (cross-block) covariance stays zero by construction
        of the partition, so it is never touched.
        """
        ekf = self._ekf
        u_vec = ekf._build_u(u)
        x_new = np.asarray(ekf._f_fn(self._x, u_vec, dt, t)).reshape(-1)
        F     = np.asarray(ekf._F_fn(self._x, u_vec, dt, t))
        L = None
        if Q is None and ekf._L_fn is not None:
            L = np.asarray(ekf._L_fn(self._x, u_vec, dt, t))
        P = self._P
        for idx in ekf._blocks:
            sub = np.ix_(idx, idx)
            Fbb = F[sub]
            if Q is not None:
                Qbb = Q[sub]
            elif L is not None:
                Lb  = L[idx, :]
                Qbb = Lb @ ekf._Sigma @ Lb.T
            else:
                Qbb = 0.0
            P[sub] = Fbb @ P[sub] @ Fbb.T + Qbb
        self._x = x_new
        self._P = 0.5 * (P + P.T)

    # ---- Update --------------------------------------------------------

    def update(self, *args, **kwargs) -> None:
        """Polymorphic update.

        Sensor form (auto h, H, R from Noise declarations):
            ekf.update(part, gyro=z_gyro, accel=z_accel)

        Low-level form (caller-supplied h, R):
            ekf.update(h_sym, z, R)
            ekf.update(h_sym, z=z, R=R)
        """
        if args and callable(args[0]):
            h_sym = args[0]
            try:
                z = args[1] if len(args) > 1 else kwargs["z"]
                R = args[2] if len(args) > 2 else kwargs["R"]
            except KeyError as e:
                raise TypeError(
                    f"NumpyEKF.update: low-level form needs h_sym + z + R; "
                    f"missing {e}") from None
            return self._update_low_level(h_sym, z, R)
        if len(args) == 1 and not callable(args[0]):
            return self._update_sensor(args[0], kwargs)
        raise TypeError(
            "NumpyEKF.update: pass (h_sym, z, R) for the low-level form "
            "or (part, **{output_name: z}) for the sensor form.")

    def _update_sensor(self, part, measurements: dict[str, Any]) -> None:
        if not measurements:
            raise ValueError(
                f"NumpyEKF.update: no measurements provided for "
                f"{type(part).__name__}('{part.name}').")
        ekf = self._ekf
        for out_name, z in measurements.items():
            key = (id(part), out_name)
            if key not in ekf._sensors:
                avail = [k[1] for k in ekf._sensors if k[0] == id(part)]
                raise KeyError(
                    f"NumpyEKF.update: part '{part.name}' has no "
                    f"registered output {out_name!r} (available: {avail}).")
            self._apply_sensor_update(ekf._sensors[key], z,
                                      ekf._u_defaults)

    def _apply_sensor_update(self, spec_o: dict[str, Any], z,
                             u_vec: np.ndarray) -> None:
        """Evaluate one sensor's cached h/H/R at the current state + `u_vec`
        and fold the measurement `z` in via the Joseph-form update."""
        ekf = self._ekf
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != spec_o["dim"]:
            raise ValueError(
                f"NumpyEKF: {spec_o['full']}: expected z of size "
                f"{spec_o['dim']}, got {z_arr.size}.")
        h_val = np.asarray(spec_o["h_fn"](self._x, u_vec, 0.0, 0.0)
                            ).reshape(-1)
        H     = np.asarray(spec_o["H_fn"](self._x, u_vec, 0.0, 0.0))
        if spec_o["L_h_fn"] is not None:
            L_h = np.asarray(spec_o["L_h_fn"](self._x, u_vec, 0.0, 0.0))
            R   = L_h @ ekf._Sigma @ L_h.T
        else:
            R   = np.zeros((spec_o["dim"], spec_o["dim"]))
        self._apply_update(z_arr, h_val, H, R)

    # ---- Measurement bus + step ----------------------------------------

    def _resolve_meas_name(self, name: str) -> str:
        """Resolve a full or craft-relative sensor name to its full key."""
        if name in self._meas:
            return name
        matches = [n for n in self._meas if n.endswith("." + name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"NumpyEKF.feed: ambiguous sensor name {name!r}; matches "
                f"{matches}. Use the fully-qualified form.")
        raise KeyError(
            f"NumpyEKF.feed: unknown sensor {name!r}. Registered: "
            f"{sorted(self._meas)}")

    def feed(self, name: str, z, *, t: float | None = None) -> None:
        """Drop a measurement into the bus and mark it fresh.

        `name` is a registered sensor's full `"craft.part.output"` name or
        an unambiguous suffix. `t` is an optional measurement timestamp
        used for staleness rejection in `step()`. The reading is consumed
        (and the fresh flag cleared) by the next `step()`.
        """
        m = self._meas[self._resolve_meas_name(name)]
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != m["dim"]:
            raise ValueError(
                f"NumpyEKF.feed: {name}: expected z of size {m['dim']}, "
                f"got {z_arr.size}.")
        m["value"] = z_arr
        m["fresh"] = True
        m["t"]     = t

    def step(self, dt: float, *, t: float | None = None,
             Q: np.ndarray | None = None) -> None:
        """Fold the interval-start measurements, then predict by `dt`.

        The sim emits sensor outputs from the *input* state, so a reading
        published over the step `[t, t+dt]` is sampled at its start `t`.
        It is therefore folded against the current (pre-predict) state —
        **update-then-predict** — and only then is the state propagated
        over `[t, t+dt]`. (Folding an interval-start reading *after* a full
        predict — the earlier order — meets it with the `t+dt` state, which
        biases rate-derived states like orientation by O(dt): during a
        transient the innovation carries the rate's change over `dt`.)
        After the call the held state is the prediction at `t+dt` — the
        estimate to act on at that instant.

        Pulls wired ports first: latched `command` ports refresh
        `self.inputs` (ZOH), and each `meas` port whose upstream advanced
        is dropped into the mailbox. `Q` overrides the process noise (else
        `self.Q`, else model auto-assembly). A measurement stamped after
        the current time is held for a later step; full out-of-sequence
        handling is out of scope.
        """
        t_start = t if t is not None else self._t
        # Pull wired command ports (ZOH, rate-gated) into the latched
        # input dict — gated on the same clock as the sim so predict and
        # truth hold the identical command.
        for full, port in self._cmd_ports.items():
            v = port.latched_value(t_start)
            if v is not None:
                self.inputs[full] = float(v) if np.ndim(v) == 0 else v
        # Pull wired/own measurement ports: fold in once per fresh sample.
        for full, port in self._meas_ports.items():
            ver = port.cur_version
            if ver > self._seen_meas.get(full, 0):
                self._seen_meas[full] = ver
                self.feed(full, port.read(), t=port.cur_t)

        u_dict  = self.inputs if self.inputs else None
        u_vec = self._ekf._build_u(u_dict)
        # Update first: fold the interval-start measurements at the current
        # (pre-predict) state, so the reading meets the state it observed.
        for full, m in self._meas.items():
            if not m["fresh"]:
                continue
            m["fresh"] = False
            if m["t"] is not None and m["t"] < t_start - 1e-9:
                continue  # stale: predates the current (pre-predict) state
            self._apply_sensor_update(self._ekf._sensors[m["key"]],
                                      m["value"], u_vec)
        # Then predict over [t_start, t_start + dt].
        self.predict(dt, t=t_start, u=u_dict,
                     Q=Q if Q is not None else self.Q)
        self._t = t_start + dt
        if self._est_port is not None:
            self._est_port.set(self._x.copy())

    def _update_low_level(self,
                          h_sym: Callable[[ca.MX], ca.MX],
                          z: np.ndarray,
                          R: np.ndarray) -> None:
        """Caller-supplied h(x) callable + R. Used for direct-slot
        measurements (`measurement_slot` helpers)."""
        ekf = self._ekf
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        if R.shape != (z.size, z.size):
            raise ValueError(
                f"NumpyEKF.update: R shape {R.shape} doesn't match z size {z.size}")

        h_mx = h_sym(ekf._x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta_h", ekf.spec.tangent_dim, 1)
        x_pert = ekf.spec.boxplus_sym(ekf._x_sym, delta)
        h_pert = h_sym(x_pert)
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(
            ca.jacobian(h_pert, delta),
            delta, ca.MX.zeros(ekf.spec.tangent_dim, 1))

        h_fn = ca.Function("h_low", [ekf._x_sym], [h_mx])
        H_fn = ca.Function("H_low", [ekf._x_sym], [H_sym])
        h_x  = np.asarray(h_fn(self._x)).reshape(-1)
        H    = np.asarray(H_fn(self._x))
        if h_x.size != z.size:
            raise ValueError(
                f"NumpyEKF.update: h(x) size {h_x.size} doesn't match z "
                f"size {z.size}")
        self._apply_update(z, h_x, H, R)

    def _apply_update(self,
                      z: np.ndarray,
                      h_x: np.ndarray,
                      H: np.ndarray,
                      R: np.ndarray) -> None:
        y = z - h_x
        S = H @ self._P @ H.T + R
        K = np.linalg.solve(S.T, (self._P @ H.T).T).T
        delta_x = K @ y
        self._x = self._ekf.spec.boxplus_num(self._x, delta_x)
        I = np.eye(self._ekf.spec.tangent_dim)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    def observability(self, **kwargs):
        """Local observability report for this EKF's sensor set at an
        operating point (defaults to the world's initial state). See
        `manta.estimation.observability`. Pass `sensors=[…]` to ask what a
        subset of sensors would observe."""
        from ...estimation.observability import observability
        return observability(self._ekf, **kwargs)

    def __repr__(self) -> str:
        return f"<NumpyEKF over {self._ekf!r}>"


# ---------------------------------------------------------------------------
# NumpyLQR — runtime for LQR
# ---------------------------------------------------------------------------

class NumpyLQR:
    """Native-Python evaluator wrapping an `LQR` IR.

    Stateless: the gain is baked at construction, so `control` is a pure
    function of the supplied state estimate."""

    def __init__(self, lqr) -> None:
        from ...control.lqr import LQR
        if not isinstance(lqr, LQR):
            raise TypeError(
                f"NumpyLQR: expected LQR, got {type(lqr).__name__}")
        self._lqr = lqr
        # Signal-bus ports (lazy): a state-estimate consumer + one
        # command producer per control input.
        self._est_in = None
        self._cmd_ports: dict[str, Any] = {}
        # Fixed gather from the wired estimate's layout → this LQR's full
        # ambient layout: list of (full_offset, dim, src_offset). Built
        # once on first compute() (None until then).
        self._gather: "list[tuple[int, int, int]] | None" = None

    @property
    def K(self) -> np.ndarray:
        return self._lqr.K

    @property
    def spec(self):
        return self._lqr.spec

    @property
    def input_names(self) -> list:
        return self._lqr.input_names

    def u(self, x_flat) -> np.ndarray:
        """Control vector for a flat ambient state (spec layout)."""
        return np.asarray(
            self._lqr.control_fn(np.asarray(x_flat, dtype=float))).reshape(-1)

    def control(self, state: dict) -> dict[str, float]:
        """Map a state estimate → `{input_name: value}`.

        `state` is nested `{owner: {slot: value}}` (e.g.
        `NumpyEKF.state_dict()` or a `NumpyWorld` state) or flat
        `{"owner.slot": value}`. Slots not supplied fall back to the
        world's initial state.
        """
        lqr = self._lqr
        flat: dict[str, Any] = {}
        for owner_name, owner_state in lqr.world._initial_state_dict().items():
            for k, v in owner_state.items():
                flat[f"{owner_name}.{k}"] = v
        for k, v in state.items():
            if isinstance(v, dict):
                for slot, val in v.items():
                    flat[f"{k}.{slot}"] = val
            else:
                flat[k] = v
        x = lqr.spec.pack({k: v for k, v in flat.items() if k in lqr.spec})
        u_vec = self.u(x)
        return {name: float(u_vec[i])
                for i, name in enumerate(lqr.input_names)}

    # ---- Signal-bus ports ----------------------------------------------

    @property
    def estimate(self):
        """Consumer port for the state estimate this regulator acts on.
        Wire `ekf.estimate` into it."""
        from ...signal import Signal
        if self._est_in is None:
            self._est_in = Signal("estimate", dim=None)
        return self._est_in

    def command(self, name: str):
        """Producer port for one control input (`"craft.part.input"` or an
        unambiguous suffix). `compute()` writes the latest command here;
        wire it into `sim.command(...)` / `ekf.command(...)`."""
        from ...signal import Signal
        full = self._resolve_input_full(name)
        if full not in self._cmd_ports:
            self._cmd_ports[full] = Signal(full, dim=1, latched=True)
        return self._cmd_ports[full]

    def compute(self) -> dict[str, float]:
        """Read the wired estimate, evaluate the gain, and publish each
        control to its `command` port. Returns the `{input: value}` dict.

        The estimate is the producer's flat ambient vector; its slots are
        scattered (by name, via a fixed precomputed gather) into this
        LQR's full ambient layout over the operating-point base, then the
        baked control law reads the tracked slots from it. A nested dict
        is still accepted (e.g. an unwired estimate set by hand)."""
        import numpy as np
        est = self.estimate.read()
        if est is None:
            raise RuntimeError(
                "NumpyLQR.compute: estimate input is empty — wire "
                "`ekf.estimate` into `lqr.estimate` (or set it) first.")
        if isinstance(est, dict):
            u = self.control(est)                    # legacy / hand-set dict
        else:
            self._ensure_gather()
            x_full = self._lqr.x_ref.copy()          # operating-point base
            x_src  = np.asarray(est, dtype=float).reshape(-1)
            for foff, dim, soff in self._gather:
                x_full[foff:foff + dim] = x_src[soff:soff + dim]
            u_vec = self.u(x_full)
            u = {n: float(u_vec[i])
                 for i, n in enumerate(self._lqr.input_names)}
        for full, v in u.items():
            self.command(full).set(v)
        return u

    def _ensure_gather(self) -> None:
        """Build the fixed estimate→full-ambient gather from the wired
        producer's `layout`. Each slot the producer exposes that also
        exists in this LQR's full spec is copied by name; slots the
        producer lacks keep their operating-point value (and untracked
        slots are ignored by the control law anyway)."""
        if self._gather is not None:
            return
        full = self._lqr.spec
        src = self.estimate.source
        layout = getattr(src, "layout", None) if src is not None else None
        if layout is None:
            raise RuntimeError(
                "NumpyLQR.compute: wired estimate carries no layout; wire "
                "`ekf.estimate` (a flat vector + layout) or set a dict.")
        pairs: list[tuple[int, int, int]] = []
        for name, (soff, sdim) in layout.items():
            if name in full:
                pairs.append((full.slot(name).offset, sdim, soff))
        self._gather = pairs

    def _resolve_input_full(self, name: str) -> str:
        names = self._lqr.input_names
        if name in names:
            return name
        matches = [n for n in names if n.endswith("." + name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"NumpyLQR.command: ambiguous input {name!r}; matches "
                f"{matches}. Use the fully-qualified form.")
        raise KeyError(
            f"NumpyLQR.command: unknown input {name!r}. Available: "
            f"{sorted(names)}")

    def __repr__(self) -> str:
        return f"<NumpyLQR over {self._lqr!r}>"


# ---------------------------------------------------------------------------
# NumpyRecurrence — runtime for any `recurrence` block (PID, Madgwick, IMU)
# ---------------------------------------------------------------------------

class NumpyRecurrence:
    """Native-Python evaluator for a `recurrence` block.

    Holds the ambient state `_x`; `step()` evaluates the block's single
    baked `update_fn` and stores the result. There is **no boxplus at
    runtime** — the manifold integration is inside the kernel — so this one
    class drives every recurrence block (PID, Madgwick/Mahony, the IMU
    integrator) unchanged. Exposes a direct call form (`step(dt, **inputs)`)
    and signal-bus ports (`input(...)` / `output(...)` / `compute(...)`).
    """

    def __init__(self, block) -> None:
        from ...recurrence import RecurrenceBlock
        if not isinstance(block, RecurrenceBlock):
            raise TypeError(
                f"NumpyRecurrence: expected a RecurrenceBlock, got "
                f"{type(block).__name__}")
        self._b = block
        self._x = block.x0.copy()
        self._y = np.zeros(block.output_dim)
        self._t = 0.0
        self._in_ports:  dict[str, Any] = {}
        self._out_ports: dict[str, Any] = {}

    # ---- IR passthroughs ----------------------------------------------

    @property
    def block(self):
        return self._b

    @property
    def spec(self):
        return self._b.spec

    # ---- State + step --------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Current state by slot name (e.g. `{"integral": …}`)."""
        return self._b.spec.unpack(self._x)

    def reset(self, state: dict | None = None) -> None:
        """Reset to the block's initial state, or overlay `state` (a
        `{slot_name: value}` dict) over it."""
        if state is None:
            self._x = self._b.x0.copy()
        else:
            base = self._b.spec.unpack(self._b.x0)
            base.update(state)
            self._x = self._b.spec.pack(base)
        self._y = np.zeros(self._b.output_dim)

    def _pack_u(self, inputs: dict) -> np.ndarray:
        u = np.zeros(self._b.input_dim)
        off = 0
        for p in self._b.inputs:
            if p.name not in inputs:
                raise KeyError(
                    f"NumpyRecurrence.step: missing input {p.name!r}; "
                    f"required: {[pp.name for pp in self._b.inputs]}")
            v = np.atleast_1d(np.asarray(inputs[p.name], dtype=float)).reshape(-1)
            if v.size != p.dim:
                raise ValueError(
                    f"NumpyRecurrence.step: input {p.name!r} expects dim "
                    f"{p.dim}, got {v.size}.")
            u[off:off + p.dim] = v
            off += p.dim
        return u

    def step(self, dt: float, *, t: float | None = None,
             **inputs) -> dict[str, Any]:
        """Advance one step and return the readouts by output-port name.

        `inputs` supplies every declared input port by name. `t` is the
        step-start time (defaults to the runtime's internal clock).
        """
        tt = self._t if t is None else t
        u = self._pack_u(inputs)
        x_next, y = self._b.update_fn(self._x, u, dt, tt)
        self._x = np.asarray(x_next).reshape(-1)
        self._y = np.asarray(y).reshape(-1)
        self._t = tt + dt
        return self.outputs()

    def outputs(self) -> dict[str, Any]:
        """Last-computed readouts by port name (scalars unwrapped)."""
        out: dict[str, Any] = {}
        off = 0
        for p in self._b.outputs:
            seg = self._y[off:off + p.dim]
            out[p.name] = float(seg[0]) if p.dim == 1 else seg.copy()
            off += p.dim
        return out

    # ---- Signal-bus ports ----------------------------------------------

    def input(self, name: str):
        """Consumer port for an input (latched / zero-order-hold). Wire a
        producer into it, or `set()` it yourself; `compute()` reads it."""
        from ...signal import Signal
        port = self._port(name, [p.name for p in self._b.inputs], "input")
        if port not in self._in_ports:
            dim = next(p.dim for p in self._b.inputs if p.name == port)
            self._in_ports[port] = Signal(port, dim=dim, latched=True)
        return self._in_ports[port]

    def output(self, name: str):
        """Producer port for a readout, refreshed after each `compute()`."""
        from ...signal import Signal
        port = self._port(name, [p.name for p in self._b.outputs], "output")
        if port not in self._out_ports:
            dim = next(p.dim for p in self._b.outputs if p.name == port)
            self._out_ports[port] = Signal(port, dim=dim)
        return self._out_ports[port]

    def compute(self, dt: float, *, t: float | None = None) -> dict[str, Any]:
        """Read the wired input ports, step, and publish to output ports."""
        tt = self._t if t is None else t
        inputs: dict[str, Any] = {}
        for name, port in self._in_ports.items():
            v = port.latched_value(tt)
            if v is not None:
                inputs[name] = v
        out = self.step(dt, t=tt, **inputs)
        for name, port in self._out_ports.items():
            port.set(np.atleast_1d(np.asarray(out[name])).ravel(), t=self._t)
        return out

    @staticmethod
    def _port(name: str, candidates: list[str], label: str) -> str:
        if name in candidates:
            return name
        raise KeyError(
            f"NumpyRecurrence: unknown {label} port {name!r}. Available: "
            f"{candidates}")

    def __repr__(self) -> str:
        return f"<NumpyRecurrence over {self._b!r}>"


# ---------------------------------------------------------------------------
# TargetNumpy factory
# ---------------------------------------------------------------------------

class _NumpyBackend(Target):
    """Native-Python backend: all three transforms lower to in-memory
    runtime evaluators."""

    name = "TargetNumpy"

    def lower_sim(self, sim, **opts):
        return NumpyWorld(sim)

    def lower_ekf(self, ekf, **opts):
        return NumpyEKF(ekf)

    def lower_lqr(self, lqr, **opts):
        return NumpyLQR(lqr)

    def lower_recurrence(self, block, **opts):
        return NumpyRecurrence(block)


def TargetNumpy(ir):
    """Native-Python backend factory.

    Lowers a compiled-model IR to a Python-native runtime evaluator.
    Dispatches by IR type:

      * `Sim` → `NumpyWorld` (`.step()`, `.initial_state()`).
      * `EKF`           → `NumpyEKF`   (`.predict()`, `.update()`,
                                        `.state_dict()`, `.reset()`,
                                        `.x`, `.P`).
      * `LQR`           → `NumpyLQR`   (`.control()`, `.u()`, `.K`).
    """
    return _NumpyBackend().lower_block(ir)
