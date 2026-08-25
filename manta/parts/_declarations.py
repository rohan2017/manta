"""Declaration sentinels — the class-scope vocabulary of a Part.

A part (or disturbance) declares its interface as class attributes:

  * `Parameter(default)`        — frozen-at-config-time value (promotable
                                  to a live graph input for system ID when
                                  declared with `manifold=`).
  * `State(init=, manifold=)`   — per-tick state slot.
  * `Input(default)`            — per-tick external command.
  * `Output()`                  — per-tick emitted observable.
  * `WhiteNoise` / `RandomWalkNoise` / `GaussMarkovNoise`
                                — stochastic channels, including their
                                  CasADi `synthesize()` plumbing.

`DeclarationHost` (`manta.parts.base`) resolves these onto each instance
at construction. This module also hosts `PartUpdate` — the bundle
`Part.update()` returns, naming the same declared slots — and shared
construction-time validators (`unit_axis`).
"""

from __future__ import annotations

import math
from typing import Any


class _Declaration:
    """Base for class-attribute declarations. Stored on the class; replaced
    by the resolved value on each instance during `Part.__init__`."""

    __slots__ = ("default",)

    def __init__(self, default: Any) -> None:
        self.default = default


class Parameter(_Declaration):
    """Frozen-at-config-time value. Set when the user constructs a Part,
    used as a constant during graph tracing.

    Concrete attribute types are deduced from the default value at __init__
    time — a `Parameter(1.0)` becomes a Python float; a
    `Parameter((1.0, 0.0, 0.0))` stays a tuple until the part's update()
    promotes it to an IR vector (`Vec3[F].constant` / `Vec3[F].coerce`).

    Args:
        manifold — optional `Manifold` instance or shortcut string
                   (``"R1"``, ``"R3"``; same vocabulary as `Noise`).
                   Declaring it makes the parameter *promotable*: a
                   transform constructed with `parameters=[...]` (system
                   identification — see `manta.fit`) can promote it from
                   a baked graph constant to a live graph input named
                   `<craft>.<part>.<param>`. Inside `update()` a promoted
                   parameter reads as an IR value (the trace binds it),
                   so parts consume promotable parameters through the
                   `.coerce` factory, which accepts both forms.
                   `None` (default) — a plain Python config value.
        frame    — Frame tag, consumed when `manifold` is a shortcut
                   resolving to a vector manifold. The promoted input's
                   frame; must match what `update()` composes it with.
        numeric  — whether the declaration participates in the framework's
                   finite-number validation. Set false only for typed object
                   configuration whose protocol the owning Part validates.
    """

    __slots__ = ("allow_infinite", "manifold", "numeric")

    def __init__(self, default: Any, *, manifold=None, frame=None,
                 allow_infinite: bool = False,
                 numeric: bool = True) -> None:
        super().__init__(default)
        if not isinstance(numeric, bool):
            raise TypeError("Parameter.numeric must be a bool")
        if not numeric and (manifold is not None or allow_infinite):
            raise ValueError(
                "non-numeric Parameters cannot declare a manifold or "
                "allow_infinite")
        self.allow_infinite = allow_infinite
        self.numeric = numeric
        if manifold is None:
            self.manifold = None
        else:
            from ..ir.manifold import Manifold, manifold_from_shortcut
            self.manifold = (manifold if isinstance(manifold, Manifold)
                             else manifold_from_shortcut(manifold,
                                                         frame=frame))


class Output(_Declaration):
    """Per-tick value produced by a part (sensor reading, derived quantity,
    telemetry signal).

    Declared at class scope. The part writes its computed value via
    `PartUpdate.outputs["<name>"] = <Vec3 | Scalar | …>`. The framework
    emits the value as a graph output named "<part_name>.<name>"; tick
    callers read it from the result dict (read-only, doesn't round-trip
    back as next-tick state). The output's shape is whatever the part
    writes — nothing downstream needs it declared.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(default=None)


class Input(_Declaration):
    """Per-tick external value.

    Declared at class scope on a Part. The framework:
      * Creates a graph input named "<part_name>.<input_name>" each compile.
      * Rebinds the part attribute to the symbolic node before calling
        `update()`, so `self.<input_name>` reads the current value.
      * Initial state from Craft.initial_state() includes the input slot
        seeded with the declaration's `default` (or the construction-time
        override if the user passed one).
      * Inputs pass through Sim.step's merge — they persist
        between steps until the user overrides. This makes per-tick
        commands ergonomic: set once, tick repeatedly, change when you
        want.

    Args:
        default — Python value used to seed the initial state. May be
                  overridden at construction (`Motor("m", torque_cmd=0.5)`)
                  in which case the override becomes the seed.

    The semantic distinction from `Parameter`: Parameter values are
    frozen into the compiled graph as constants; Input values are
    re-evaluated each tick from the state dict.
    """


class SynthesizedNoise:
    """Return value of `Noise.synthesize()` describing the IR plumbing
    one occurrence of a Noise channel contributes to a tick.

    Attrs:
        signal_sym    — IR symbol bound onto `part.<name>` so that user
                        code reading `self.<name>` inside `update()`
                        sees the symbolic current value.
        state_update  — Optional `(slot_full_name, next_value_sym)`; the
                        compiler appends this to the tick's state-output
                        list (the slot evolves over time).
    """

    __slots__ = ("signal_sym", "state_update")

    def __init__(self, signal_sym, state_update=None):
        self.signal_sym   = signal_sym
        self.state_update = state_update


class Noise(_Declaration):
    """Abstract base for noise-channel declarations.

    Subclasses set class-level metadata (`kind`, `contributes_state`)
    and implement `synthesize()` (the per-tick IR plumbing). Backends
    key on `signal_manifold.kind` via their own registry — no
    `isinstance(WhiteNoise)` dispatch anywhere in the codebase.

    Concrete subclasses:

      * `WhiteNoise` — per-tick i.i.d. Gaussian. The framework creates a
        graph input named `<part>.<noise_name>`, rebinds the part
        attribute to that input, and the part adds it directly into its
        sensor reading (or process expression). σ is the per-tick
        measurement stddev. `kind = "white"`.

      * `RandomWalkNoise` — random-walk bias. The framework synthesizes:
          - A **state slot** `<part>.<noise_name>` holding the bias.
          - A driver noise input `<part>.<noise_name>_driver`.
          - A state update each tick:
                bias_next = bias + sqrt(dt) · driver,    driver ~ N(0, σ²).
        Inside `update()`, `self.<noise_name>` reads the bias state
        (the slowly-drifting current value). σ has continuous σ/√Hz
        semantics; per-tick bias variance is dt·σ². `kind = "random_walk"`.

      * `GaussMarkovNoise` — first-order Gauss–Markov (exponentially
        correlated) error with correlation time τ and stationary
        variance σ². Same state-slot / driver plumbing as the random
        walk, with the exact discrete transition
                φ = exp(-dt/τ),
                e_next = φ · e + sqrt(1 − φ²) · driver,   driver ~ N(0, σ²),
        so the slot's variance stays at σ² in steady state and the
        auto-assembled process noise is `(1 − φ²)·σ²` per tick — no
        Euler approximation anywhere. `kind = "gauss_markov"`.

    Args:
        signal_manifold — `Manifold` instance OR shortcut string. The
                          manifold of the symbol user code reads as
                          `self.<name>`. Shortcuts: ``"R1"`` (scalar),
                          ``"R3"`` (combine with ``frame=``). Default
                          ``"R3"``. Same vocabulary as `State(manifold=)`.
        frame           — Frame class, only consumed when
                          `signal_manifold` is a shortcut and resolves
                          to a vector-typed manifold. Ignored otherwise.
        sigma           — 1-σ standard deviation, scalar (isotropic
                          across axes). See subclass docstrings for
                          unit conventions.
    """

    # Class-level metadata. Subclasses MUST set both.
    kind:              str  = None    # type: ignore[assignment]
    contributes_state: bool = False

    __slots__ = ("sigma", "signal_manifold")

    def __init__(self, signal_manifold="R3", *, frame=None,
                 sigma: float = 0.0) -> None:
        super().__init__(default=None)
        from ..ir.manifold import manifold_from_shortcut
        self.signal_manifold = manifold_from_shortcut(
            signal_manifold, frame=frame)
        self.sigma = float(sigma)
        if self.sigma < 0.0:
            raise ValueError(
                f"{type(self).__name__}: sigma must be >= 0, "
                f"got {sigma!r}")

    # ---- Manifolds (resolved at synthesis time for the late-bound
    # frame default; the canonical form just returns self.signal_manifold
    # and only the legacy R3Manifold(frame=None) case substitutes) -----

    def resolved_signal_manifold(self, *, default_frame=None):
        """Return `self.signal_manifold` with any unresolved frame
        substituted from `default_frame`. Used at IR synthesis time;
        the unresolved form keeps `R3Manifold(frame=None)` legal so
        a part can declare a noise without committing to a frame
        until the compiler knows which one it's in (CraftFrame for
        parts, WorldFrame for disturbances)."""
        from ..ir.manifold import R3Manifold
        if isinstance(self.signal_manifold, R3Manifold) \
                and self.signal_manifold.frame is None:
            return R3Manifold(frame=default_frame)
        return self.signal_manifold

    def state_manifold(self, *, default_frame=None):
        """Manifold of the synthesized state slot, or None. For RW
        the state lives in the same space as the per-tick signal."""
        if not self.contributes_state:
            return None
        return self.resolved_signal_manifold(default_frame=default_frame)

    # ---- Per-instance runtime attributes ------------------------------

    def runtime_attributes(self, name: str) -> dict[str, tuple[float, bool]]:
        """The per-instance attributes this channel exposes on its owner,
        `attr -> (default, allow_zero)`. `DeclarationHost` seeds them from
        the declaration and accepts constructor overrides of the same
        names; `is_active` / `synthesize` read them back at compile time.
        Every channel has `<name>_sigma`; subclasses add their own
        (`GaussMarkovNoise` adds `<name>_tau`)."""
        return {f"{name}_sigma": (self.sigma, True)}

    # ---- Activity gate ------------------------------------------------

    def is_active(self, owner, name: str) -> bool:
        """Is this channel currently producing nonzero output? Reads
        the runtime `<name>_sigma` attribute on the owner."""
        return float(getattr(owner, f"{name}_sigma")) > 0.0

    # ---- Input-name classification (tick_signature.py / ekf.py) -------

    def driver_input_name(self, name: str) -> str:
        """The name of this channel's per-tick stochastic input. For
        White noise the signal IS the driver (same name); for RW the
        driver is a separate `<name>_driver` input distinct from the
        bias state name."""
        return name

    # ---- Initial state seeding ---------------------------------------

    def initial_state_entries(self, name: str, owner) -> dict[str, object]:
        """Names → zero values this channel contributes to the seed
        state dict (state_spec.unpack-compatible). Inert RW channels
        return an empty dict; everyone else seeds at least the signal
        slot."""
        return {name: self._zero_value()}

    # ---- IR synthesis (the world-tick compiler's entry point) --------

    def synthesize(self, *, base_name: str, name: str, dt, default_frame,
                   owner) -> SynthesizedNoise:
        """Build one tick's worth of IR plumbing for this channel.
        Subclasses implement; the world-tick compiler calls this once
        per noise declaration per owner."""
        raise NotImplementedError(
            f"{type(self).__name__}.synthesize must be implemented.")

    # ---- Helpers ------------------------------------------------------

    def _zero_value(self):
        """Numpy-side zero for seeding the input dict. Reads off the
        manifold so any shape (scalar, vec3, vec6, quat, …) Just Works."""
        return self.signal_manifold.default_value()


class WhiteNoise(Noise):
    """Per-tick i.i.d. Gaussian noise channel. σ is the per-tick stddev."""

    kind              = "white"
    contributes_state = False

    __slots__ = ()

    def synthesize(self, *, base_name, name, dt, default_frame, owner):
        mfd = self.resolved_signal_manifold(default_frame=default_frame)
        return SynthesizedNoise(signal_sym=mfd.ir_input(base_name))


class RandomWalkNoise(Noise):
    """Random-walk bias channel. σ has σ/√Hz drift-density units."""

    kind              = "random_walk"
    contributes_state = True

    __slots__ = ()

    def driver_input_name(self, name: str) -> str:
        return f"{name}_driver"

    def initial_state_entries(self, name, owner):
        # Inert when σ=0: no slot, no driver, no seed.
        if not self.is_active(owner, name):
            return {}
        zero = self._zero_value()
        return {name: zero, f"{name}_driver": zero}

    def synthesize(self, *, base_name, name, dt, default_frame, owner):
        import casadi as ca
        mfd = self.resolved_signal_manifold(default_frame=default_frame)
        if not self.is_active(owner, name):
            # Inert: bind zero, no slot, no driver.
            return SynthesizedNoise(signal_sym=mfd.ir_zero())
        # Active: bias-state input + driver input + state update.
        driver_name = f"{base_name}_driver"
        sqrt_dt    = ca.sqrt(dt._mx)
        bias_sym   = mfd.ir_input(base_name)
        driver_sym = mfd.ir_input(driver_name)
        bias_next  = mfd.ir_add(bias_sym, sqrt_dt * driver_sym._mx)
        return SynthesizedNoise(
            signal_sym=bias_sym,
            state_update=(base_name, bias_next),
        )


class GaussMarkovNoise(Noise):
    """First-order Gauss–Markov error channel: correlation time `tau`
    (seconds, > 0) and stationary 1-σ `sigma` (signal units).

    The synthesized state slot holds the correlated error itself; the
    exact discrete transition `φ = exp(-dt/τ)` keeps the slot's variance
    at σ² for any tick length. With `sigma == 0` the channel is inert
    (no slot, no driver), exactly like `RandomWalkNoise`.
    """

    kind              = "gauss_markov"
    contributes_state = True

    __slots__ = ("tau",)

    def __init__(self, signal_manifold="R3", *, frame=None,
                 sigma: float = 0.0, tau: float) -> None:
        super().__init__(signal_manifold, frame=frame, sigma=sigma)
        self.tau = _finite_positive_tau(tau, who=type(self).__name__)

    def runtime_attributes(self, name):
        return {**super().runtime_attributes(name),
                f"{name}_tau": (self.tau, False)}

    def driver_input_name(self, name: str) -> str:
        return f"{name}_driver"

    def initial_state_entries(self, name, owner):
        if not self.is_active(owner, name):
            return {}
        zero = self._zero_value()
        return {name: zero, f"{name}_driver": zero}

    def synthesize(self, *, base_name, name, dt, default_frame, owner):
        import casadi as ca
        mfd = self.resolved_signal_manifold(default_frame=default_frame)
        if not self.is_active(owner, name):
            return SynthesizedNoise(signal_sym=mfd.ir_zero())
        tau = _finite_positive_tau(getattr(owner, f"{name}_tau"),
                                   who=f"{type(owner).__name__}"
                                       f"({getattr(owner, 'name', '?')!r})."
                                       f"{name}_tau")
        phi = ca.exp(-dt._mx / tau)
        gain = ca.sqrt(1.0 - phi * phi)
        driver_name = f"{base_name}_driver"
        error_sym  = mfd.ir_input(base_name)
        driver_sym = mfd.ir_input(driver_name)
        # φ·e + sqrt(1-φ²)·w  ==  e ⊕ ((φ-1)·e + sqrt(1-φ²)·w): the
        # manifold's own `ir_add` keeps the typed value, as for RW.
        error_next = mfd.ir_add(
            error_sym, (phi - 1.0) * error_sym._mx + gain * driver_sym._mx)
        return SynthesizedNoise(
            signal_sym=error_sym,
            state_update=(base_name, error_next),
        )


def _finite_positive_tau(value, *, who: str) -> float:
    try:
        tau = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{who}: tau must be a positive finite number of "
                        f"seconds, got {value!r}") from None
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"{who}: tau must be a positive finite number of "
                         f"seconds, got {value!r}")
    return tau


class State(_Declaration):
    """Per-tick state slot.

    Declared at class scope. The framework:
      * Creates a graph input named "<part_name>.<state_name>" each compile.
      * Rebinds the part attribute to that input node before calling
        `update()`, so `self.<state_name>` reads the symbolic current value.
      * Reads the new value from `PartUpdate.new_state["<state_name>"]`
        and emits it as a graph output of the same name. Omitted states
        pass through unchanged.

    Args:
        init      Python value (default initial value across compiles).
                  For R1 a float; for R3 a length-3 tuple / ndarray; for
                  SO(3) a length-4 quaternion (w, x, y, z).
        manifold  String shortcut ('R1', 'R3') or a `Manifold` instance.
                  SO(3) state is fully supported — pass an explicit
                  `SO3Manifold(from_frame=..., to_frame=...)` instance
                  (the string shortcut is intentionally disallowed because
                  SO(3) needs the dual-frame parametrization). The slot
                  then evolves on the manifold: the part integrates it
                  with `manifold.boxplus(q, ω·dt)`, the framework keeps it
                  unit-normalized, and the EKF/LQR linearization gives it
                  a 3-dim tangent automatically. See `tests/test_so3_state`.
        frame     Frame tag for R3 state. Default `CraftFrame`. Ignored
                  for R1 and SO(3) (the latter's frames live on the
                  manifold). Folded into the Manifold instance.

    `state.manifold` always reads back as a `Manifold` instance; the
    string form is normalized at construction.
    """

    __slots__ = ("frame", "init", "manifold")

    def __init__(self, init, manifold="R1", frame=None) -> None:
        from ..ir.manifold import (
            Manifold,
            R3Manifold,
            SO3Manifold,
            manifold_from_shortcut,
        )
        if isinstance(manifold, Manifold):
            mfd = manifold
        else:
            # The shared shortcut grammar (`manifold_from_shortcut`) accepts
            # any R<n>, but State slots are deliberately restricted to the
            # manifolds the integrator/EKF/LQR plumbing is exercised on:
            # R1, R3, and SO(3). SO(3) is excluded from the shortcut form
            # (not the restriction) because it needs explicit
            # from_frame/to_frame.
            if manifold not in ("R1", "R3"):
                raise NotImplementedError(
                    f"State.manifold={manifold!r}: state manifolds are "
                    f"deliberately restricted to 'R1' / 'R3' (and SO(3)) "
                    f"for now, though the shared shortcut grammar is wider. "
                    f"For SO(3), pass an explicit "
                    f"SO3Manifold(from_frame=..., to_frame=...) instance "
                    f"to capture the dual-frame parametrization.")
            mfd = manifold_from_shortcut(manifold, frame=frame)
        if isinstance(mfd, R3Manifold):
            try:
                t = tuple(float(x) for x in init)
            except (TypeError, ValueError):
                raise ValueError(
                    f"State(manifold='R3'): init must be a 3-element "
                    f"sequence, got {init!r}")
            if len(t) != 3:
                raise ValueError(
                    f"State(manifold='R3'): init must be length-3, got "
                    f"{init!r}")
            init = t
        elif isinstance(mfd, SO3Manifold):
            if mfd.from_frame is None or mfd.to_frame is None:
                raise ValueError(
                    f"State(SO3Manifold): from_frame and to_frame must "
                    f"both be specified — got from_frame={mfd.from_frame!r}, "
                    f"to_frame={mfd.to_frame!r}.")
            try:
                t = tuple(float(x) for x in init)
            except (TypeError, ValueError):
                raise ValueError(
                    f"State(SO3Manifold): init must be a length-4 "
                    f"quaternion (w, x, y, z), got {init!r}")
            if len(t) != 4:
                raise ValueError(
                    f"State(SO3Manifold): init must be length-4, got "
                    f"{init!r}")
            init = t
        super().__init__(default=init)
        self.init     = init
        self.manifold = mfd
        # Keep the explicit `frame` attribute for read sites that
        # consult it directly; for R3 it mirrors the manifold's frame.
        # For SO3 the dual frames live on the manifold itself.
        self.frame = (mfd.frame if isinstance(mfd, R3Manifold) else frame)


# ---------------------------------------------------------------------------
# PartUpdate — return type for Part.update()
# ---------------------------------------------------------------------------

class PartUpdate:
    """Bundle returned by `Part.update(ctx)` describing this tick's
    contributions: a wrench (force + torque on parent in CraftFrame), new
    values for any declared State slots, any declared Output values the
    part produces, and the rates its I/O runs at.

    Construction::

        return PartUpdate(wrench, {"angle": a})
        return PartUpdate(wrench=w, new_state={"angle": a, "rate": r})
        return PartUpdate(wrench=w, outputs={"gyro": gyro_vec},
                          rates={"gyro": self.rate})

    `rates` maps this part's Output slots and/or Input attribute names to
    a rate in Hz (`None` ⇒ every tick). It is metadata only — the
    compiled tick stays a pure function (no sample-and-hold state enters
    the kernel, so it never complicates autodiff or the EKF/LQR
    linearization). The runtimes gate the matching *port*: an Output is
    published once per 1/rate window and held in between; an Input
    command is latched (ZOH) once per window, so truth and the
    estimator's predict see the *same* held command.

    Stateless parts can return a bare `Wrench` instead — the framework
    wraps it as `PartUpdate(wrench=w)` automatically.
    """

    __slots__ = ("new_state", "outputs", "rates", "wrench")

    def __init__(self,
                 wrench=None,
                 new_state: dict | None = None,
                 outputs: dict | None = None,
                 rates: dict | None = None) -> None:
        if wrench is None:
            raise TypeError("PartUpdate: wrench is required")
        self.wrench = wrench
        self.new_state = dict(new_state) if new_state else {}
        self.outputs   = dict(outputs)   if outputs   else {}
        self.rates     = dict(rates)     if rates     else {}


# ---------------------------------------------------------------------------
# Validators — shared construction-time guards for declared values
# ---------------------------------------------------------------------------

def unit_axis(value, *, who: str, what: str) -> tuple:
    """Validate a length-3, nonzero axis parameter; return it normalized.

    Construction-time guard for parts whose math assumes a unit axis
    (joint DOF axes, airfoil chord/normal): a zero axis is a config
    error and a non-unit one would silently scale the physics."""
    try:
        axis = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{who}: {what} must be a length-3 vector, got {value!r}")
    if len(axis) != 3:
        raise ValueError(
            f"{who}: {what} must be length-3, got {value!r}")
    n = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
    if n == 0.0:
        raise ValueError(f"{who}: {what} must be nonzero.")
    return (axis[0] / n, axis[1] / n, axis[2] / n)
