"""EKF — Error-state (manifold-aware) Extended Kalman Filter IR.

`EKF(world)` walks the world's crafts + fields, building:
  * a `StateSpec` (joint state across every craft + every state-bearing
    disturbance). `track=` carves a subset out of this — see below.
  * a `Linearization` over the world tick (`manta.linearization`), which
    produces the tangent-space `F`, process-noise gain `L`, and per-output
    `h`/`H`/`L_h` (the Jacobian machinery — also the seam LQR/iLQR reuse).
  * **the full Kalman recursion as fused `ca.Function`s** via
    `Linearization.kalman_functions`: a predict step `_predict_fn`
    `(x,P,Q,u,dt,t) → (x',P')`, a process-noise kernel `_process_noise_fn`
    `(x,u,dt,t) → Q = L Σ Lᵀ`, and a per-sensor Joseph update `_update_fns`
    `(x,P,z,u,t) → (x',P')`. The linear algebra lives HERE, once — both
    backends (numpy + emitted C++) just *evaluate* these kernels; neither
    reimplements the predict / Joseph update. The EKF still does the model
    introspection (Inputs vs Noise, which outputs are sensors) and exposes
    the building-block Jacobians for analysis (`observability`).
  * Per-sensor measurement bundles: `_sensors[(id(part), output_name)]`
    holds the sensor's `update_fn` (+ the h/H/L_h Jacobians for analysis),
    the sensor dim, and a back-ref to the owning part + craft.

The result is an IR — a description of the symbolic predict/measurement
graph. Lower to a backend to actually run::

    from manta import TargetNumpy, EKF

    cw  = TargetNumpy(Sim(sim_world))
    ekf = TargetNumpy(EKF(est_world))
    for _ in range(N):
        state = cw.step(state, t=t, dt=dt)
        ekf.predict(t=t, dt=dt, u={"thrust.throttle": cmd})
        ekf.update(est_imu, gyro=measured_gyro, accel=measured_accel)
        ekf.update(est_gps, position=measured_pos)
        t += dt

State subsetting + measurement bus (the friendlier path)::

    from manta import EKF, TargetNumpy, POSE, TWIST

    # Estimate only craft "chaser"; ignore the rest. Use just these
    # sensors; treat the thruster command as a known input.
    ekf = TargetNumpy(EKF(world,
                          track={"chaser": POSE | TWIST},
                          sensors=["chaser.imu.gyro", "chaser.gps.position"],
                          inputs=["chaser.thruster.throttle"]))
    for _ in range(N):
        ekf.inputs["chaser.thruster.throttle"] = cmd      # latched (ZOH)
        ekf.feed("chaser.imu.gyro", gyro_z, t=t)          # 400 Hz
        if new_fix:
            ekf.feed("chaser.gps.position", pos_z, t=t)   # 1 Hz
        ekf.step(dt, t=t)            # predict + fold in fresh measurements

`track` is a *lower bound*: the framework expands it (and any slots the
chosen sensors observe) to a set closed under the dynamics — via the
structural sparsity of F — and freezes the rest at their initial values.
A fully independent craft you don't track drops out of the O(n³) predict
entirely; a craft you're coupled to is pulled back in automatically.

Auto-assembly contracts:

  * Q is built from every Noise channel that affects the next-tick
    state (via autodiff). The runtime can override with an explicit
    `Q=` argument to predict.

  * R is built per sensor output from the Noise channels feeding that
    output. `ekf.update(part, **measurements)` routes each measurement
    through its sensor's cached (h_fn, H_fn, R_builder).

  * Initial state seeds from `world._initial_state_dict()` (already
    PlanetState-resolved by `Sim(world)`). The runtime instantiates
    `_x` and `_P` from this seed.

Scope notes:
  * Couplings between crafts are honored by the world-tick (which the
    EKF reuses) — F propagates through them automatically.
  * RW bias channels are first-class (`RandomWalkNoise` declarations on
    Parts or Disturbances synthesize a state slot + a driver input with
    `bias_next = bias + sqrt(dt)·driver`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..linearized_system import LinearizedSystem, resolve_suffix
from .state_spec import StateSpec


class EKF:
    """Error-state EKF wrapping a `World`.

    Compiles its own world tick (separate from the sim's `cw.tick`)
    using the same fields + planets + couplings. State spans every
    craft in the world (`StateSpec.from_world`).
    """

    # Lowerable-block kind (see manta.codegen.block.KIND_EKF): a backend
    # lowers an EKF through its `lower_ekf` handler.
    RUNTIME_KIND = "ekf"

    def __init__(self, world, *,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None) -> None:
        """Build the EKF IR over `world`.

        Args:
            track:   `{craft_name: SlotSet}` — the *lower bound* of what
                     to estimate per craft. The framework expands this
                     (and any slots the chosen sensors observe) to a set
                     closed under the dynamics, freezing the rest at
                     their initial values. `None` (default) keeps the
                     full state for every craft — the legacy behavior.
            sensors: full `"craft.part.output"` names (or unambiguous
                     suffixes) the EKF may use as measurements. `None`
                     keeps every Part output as a sensor.
            inputs:  full `"craft.part.input"` names the EKF is aware of.
                     `None` keeps every Part input; excluded inputs are
                     frozen at their default.
        """
        # All slot/sensor/subset machinery — world prep, tick compile,
        # signature, input/state subsetting, the dependency closure, and the
        # Linearization — lives in `LinearizedSystem`. The EKF is then just
        # "the Kalman recursion over a linearized system": read the math off
        # it (predict/F/L + per-sensor h/H) and fuse it into kernels.
        sys = LinearizedSystem(world, track=track, sensors=sensors,
                               inputs=inputs, close_track=True)
        self.sys     = sys
        self.world   = world
        self.crafts  = sys.crafts
        self.spec    = sys.spec
        self._sample_rates = sys.sample_rates
        self._noise_specs  = sys.noise_specs
        self._input_names  = sys.input_names
        self._u_defaults   = sys.u_defaults

        lin = sys.lin
        self._x_sym  = lin.x_sym
        self._f_fn   = lin.predict_fn
        self._F_fn   = lin.F_fn
        self._L_fn   = lin.L_fn
        self._Sigma  = lin.Sigma
        # Independent-subsystem partition of the tangent state (block-diagonal
        # predict — Σ O(n_b³) instead of O(n³); see `Linearization`).
        self._blocks = lin.blocks

        # Sensor table re-keyed by `(id(part), out_name)` for
        # `ekf.update(part, **measurements)` routing (sys keys by full name).
        self._sensors: dict[tuple[int, str], dict[str, Any]] = {}
        for full, o in sys.sensors.items():
            self._sensors[(id(o["part"]), o["out_name"])] = {
                "dim":    o["dim"],
                "h_fn":   o["h_fn"],
                "H_fn":   o["H_fn"],
                "L_h_fn": o["L_h_fn"],
                "part":   o["part"],
                "craft":  o["craft"],
                "full":   o["full"],
            }

        # The full Kalman recursion, symbolic + once (see
        # `Linearization.kalman_functions`): predict + per-sensor Joseph
        # update as fused ca.Functions. Both backends EVALUATE these —
        # neither reimplements the linear algebra. `_update_fns` is keyed by
        # the sensor's full name; pair it with `_sensors` for routing.
        self._predict_fn, self._process_noise_fn, _upd = lin.kalman_functions()
        self._update_fns = _upd
        for key, o in self._sensors.items():
            o["update_fn"] = _upd[o["full"]]

        # Mutable runtime state (`_x`, `_P`) lives on the backend
        # evaluator (`NumpyEKF`), not on the IR. The IR keeps the
        # symbolic functions + sensor table; backends instantiate
        # their own state from `world._initial_state_dict()`.

    @property
    def n_blocks(self) -> int:
        """Number of independent subsystems in the tangent state. >1 means
        the predict propagates each block separately (see `_blocks`)."""
        return len(self._blocks)

    # ------------------------------------------------------------------
    # Name resolution / input vector (see manta.linearized_system)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_names(user_names, candidates, label: str) -> set[str]:
        """Resolve user-supplied names (full or unambiguous suffix) against
        a candidate list; raise on unknown/ambiguous."""
        return {resolve_suffix(k, candidates, label=label, who="EKF")
                for k in user_names}

    def _build_u(self, u: dict[str, float] | None) -> np.ndarray:
        """Resolve `u` to a flat input vector in `_input_names` order.

        Accepts either full names (`"drone.t.throttle"`) or craft-relative
        shorthand (`"t.throttle"`) when the shorthand uniquely identifies
        one input across all crafts.
        """
        if not self._input_names:
            return np.zeros(0)
        if u is None:
            return self._u_defaults.copy()
        out = self._u_defaults.copy()
        index = {n: i for i, n in enumerate(self._input_names)}
        for user_key, value in u.items():
            full = resolve_suffix(user_key, self._input_names,
                                  label="input", who="EKF.predict")
            out[index[full]] = float(value)
        return out

    # Predict / update / state_dict / reset / x / P live on the
    # runtime evaluator (`NumpyEKF`), not the IR. The split keeps
    # EKF as a pure compile-time description that any backend
    # (numpy, C++, …) can consume. Build a runtime via:
    #     from manta import TargetNumpy
    #     ekf = TargetNumpy(EKF(world))
    #     ekf.predict(dt=dt, t=t)
    #     ekf.update(part, gyro=z)
    # The IR holds the baked Kalman kernels (_predict_fn,
    # _process_noise_fn, _update_fns) the backends evaluate.



# ---------------------------------------------------------------------------
# Low-level h_sym helpers (kept for tests / advanced custom measurements)
# ---------------------------------------------------------------------------

def measurement_slot(spec: StateSpec, name: str):
    """Return an h_sym callable that reads slot `name` directly from x."""
    slot = spec.slot(name)
    def h_sym(x):
        return x[slot.ambient_offset : slot.ambient_offset + slot.ambient_dim]
    return h_sym


def measurement_component(spec: StateSpec, name: str, component: int):
    """Return an h_sym callable for a single component of a slot."""
    slot = spec.slot(name)
    if not (0 <= component < slot.ambient_dim):
        raise IndexError(
            f"measurement_component: slot {name!r} has dim {slot.ambient_dim}, "
            f"component {component} out of range")
    def h_sym(x):
        return x[slot.ambient_offset + component]
    return h_sym
