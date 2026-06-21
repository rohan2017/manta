"""FluidField + concrete fluid disturbances.

A FluidField returns `FluidState(density, pressure, temperature,
velocity)` at a queried world-frame point. The value is compound — three
scalars (density kg/m³, pressure Pa, temperature K) plus a bulk flow
velocity (Vec3[WorldFrame], m/s).

Disturbances combine by role (their `combining` flag), per component:

  * **baseline** — a regime medium (an ocean, an atmosphere). Baselines
    LAYER by their spatial `membership` rather than summing: in insertion
    order, `base ← (1 − w)·base + w·value`. So a global air background
    overlaid by an ocean pocket (membership = wet-fraction) gives
    `(1 − wet)·air + wet·ocean` — an alpha-composite override with no
    dilution, and vacuum (zero) where no baseline is active. This is the
    "which fluid am I in" selection: you don't *sum* 1025 + 1.225 kg/m³.

  * **averaged** — an estimation overlay (overlapping wind bubbles). The
    averaged disturbances are combined into a membership-weighted mean
    *among themselves* (`Σ wᵢ·valueᵢ / Σ wᵢ`), so two crafts' wind
    estimates agree where their bubbles overlap.

  * **additive** — a perturbation on top of the regime (a current, a
    thruster wake, a wing vortex, an explosion blast). Plain
    membership-weighted sum. (A future fluid-thruster must query the
    ambient velocity first and contribute only its delta, so two nearby
    thrusters don't double-count — that is the part's job, not the
    field's.)

`result = baseline + averaged + perturbations`, per component. This is a
deliberately simple linear-superposition rule set — NOT a Navier-Stokes
solve; advection and true currents are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..smoothing import hermite_blend, soft_norm
from .base import Disturbance, SuperposedField


_VEC3_W = Vec3[WorldFrame]


@dataclass(frozen=True)
class FluidState:
    """Local fluid properties at a world-frame point.

    density      — kg/m³. CasADi-MX scalar (composes with symbolic state).
    pressure     — Pa. MX scalar.
    temperature  — K. MX scalar.
    viscosity    — dynamic viscosity μ, Pa·s. MX scalar. An independent
                   property (like density): a gas baseline fills it from
                   temperature via Sutherland's law, while water sets it
                   directly. Drives the Reynolds number a foil sees.
    velocity     — bulk fluid velocity at the point, Vec3[WorldFrame].

    Disturbances and `FluidField.value_at_sym` return / consume this
    type. `__add__` (per-component sum) backs the additive pool;
    `scaled` (per-component scalar multiply) backs the membership
    weighting in the baseline / averaged blends.
    """
    density: ca.MX                  # scalar MX
    pressure: ca.MX                 # scalar MX
    temperature: ca.MX              # scalar MX
    velocity: "Vec3"                # Vec3[WorldFrame]
    viscosity: ca.MX = None         # scalar MX (defaults to 0 = unset)

    def __post_init__(self):
        # Tolerate older 4-component construction: an unset viscosity is a
        # zero MX, combined like the other scalars.
        if self.viscosity is None:
            object.__setattr__(self, "viscosity", ca.MX(0.0))

    def __add__(self, other: "FluidState") -> "FluidState":
        return FluidState(
            density     = self.density + other.density,
            pressure    = self.pressure + other.pressure,
            temperature = self.temperature + other.temperature,
            viscosity   = self.viscosity + other.viscosity,
            velocity    = self.velocity + other.velocity,
        )

    def scaled(self, s) -> "FluidState":
        """This state with every component multiplied by the MX (or
        float) scalar `s` — the membership weight in the blends."""
        return FluidState(
            density     = s * self.density,
            pressure    = s * self.pressure,
            temperature = s * self.temperature,
            viscosity   = s * self.viscosity,
            velocity    = _VEC3_W.from_mx(s * self.velocity._mx),
        )


class FluidField(SuperposedField):
    """Fluid density + pressure + temperature + bulk velocity over the
    world frame.

    The field value at a point is a `FluidState`. Concrete sources are
    added as `Disturbance` subclasses with a `combining` role (see the
    module docstring): `baseline` regime media, `averaged` estimation
    overlays, and `additive` perturbations.

    The `add_uniform` builder returns self for chaining; other
    disturbances attach via `field.add(CurrentFlow(...))`. The constructor
    accepts `density=`/`pressure=`/`temperature=` for the common
    single-uniform case — equivalent to one `add_uniform` baseline.
    """

    # Sentinel for the type-check in Field.add — any disturbance whose
    # `field_value_shape` is FluidState may be added.
    value_shape = FluidState

    def __init__(self,
                 density: float | None = None,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 *,
                 pressure: float = 0.0,
                 temperature: float = 0.0,
                 viscosity: float | None = None,
                 ) -> None:
        super().__init__()
        if density is not None:
            self.add_uniform(density, velocity,
                             pressure=pressure, temperature=temperature,
                             viscosity=viscosity)

    def _zero_value(self) -> FluidState:
        return FluidState(
            density     = ca.MX(0.0),
            pressure    = ca.MX(0.0),
            temperature = ca.MX(0.0),
            velocity    = _VEC3_W.constant((0.0, 0.0, 0.0)),
        )

    def value_at_sym(self, point: "Vec3", t) -> FluidState:
        """Combine every registered disturbance at `point`/`t` into one
        `FluidState`, per the baseline / averaged / additive rule set
        (module docstring). Returns the zero value if none are
        registered.
        """
        if not self._disturbances:
            return self._zero_value()

        baselines = [d for d in self._disturbances if d.combining == "baseline"]
        averaged  = [d for d in self._disturbances if d.combining == "averaged"]
        additive  = [d for d in self._disturbances if d.combining == "additive"]

        # Baseline regimes: layered alpha-composite override in insertion
        # order. base ← (1 − w)·base + w·contribution.
        base = self._zero_value()
        for d in baselines:
            w = d.membership(point, t)
            c = d.contribute_at_sym(point, t)
            base = base.scaled(1.0 - w) + c.scaled(w)

        # Averaged overlays: membership-weighted self-mean.
        avg = self._zero_value()
        if averaged:
            num = self._zero_value()
            den = ca.MX(0.0)
            for d in averaged:
                w = d.membership(point, t)
                num = num + d.contribute_at_sym(point, t).scaled(w)
                den = den + w
            avg = num.scaled(1.0 / (den + 1e-9))

        # Additive perturbations: membership-weighted sum on top.
        pert = self._zero_value()
        for d in additive:
            w = d.membership(point, t)
            pert = pert + d.contribute_at_sym(point, t).scaled(w)

        return base + avg + pert

    def add_uniform(self,
                    density: float,
                    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    *,
                    pressure: float = 0.0,
                    temperature: float = 0.0,
                    viscosity: float | None = None,
                    ) -> "FluidField":
        """Attach a uniform baseline medium (density + optional pressure,
        temperature, viscosity, flow). Viscosity defaults to Sutherland's
        law for air; pass it explicitly for a liquid. Returns self."""
        return self.add(UniformFluid(density, velocity,
                                     pressure=pressure,
                                     temperature=temperature,
                                     viscosity=viscosity))


# ---------------------------------------------------------------------------
# Membership helpers — smooth spatial supports (C¹), returned as callables
# `(point: Vec3, t) -> ca.MX in [0, 1]` to pass as `membership=`.
# ---------------------------------------------------------------------------

def below_surface(surface_height, width: float):
    """Membership = 1 below a surface, 0 above, blended over ±`width`.

    `surface_height(point, t)` returns the signed height of `point` above
    the surface (negative = submerged), as an MX. Used for an ocean
    regime whose top is a (possibly waving) sea surface; the complement
    `1 − membership` is the matching atmosphere region.
    """
    def _membership(point, t):
        return 1.0 - hermite_blend(surface_height(point, t), width)
    return _membership


def within_sphere(center: tuple[float, float, float],
                  radius: float, width: float):
    """Membership = 1 inside a world-frame sphere, 0 outside, blended
    over ±`width` about the radius. For a localized pocket (a wind
    bubble, an explosion); for a craft-following centre, override
    `Disturbance.membership` instead (see `CraftWindBubble`)."""
    c = _VEC3_W.constant(tuple(float(x) for x in center))

    def _membership(point, t):
        d = soft_norm(point._mx - c._mx)
        return 1.0 - hermite_blend(d - radius, width)
    return _membership


# ---------------------------------------------------------------------------
# Disturbance subclasses
# ---------------------------------------------------------------------------

class UniformFluid(Disturbance):
    """Position-independent baseline medium: constant density (+ optional
    pressure, temperature, flow).

    A `baseline` regime — where it is active (its membership, global by
    default) it *defines* the ambient fluid rather than adding to it.

    Args:
        density      — kg/m³. Common: ~1.225 (air), ~1025 (seawater),
                       ~1000 (fresh water).
        velocity     — bulk flow vector in WorldFrame, m/s. Default zero.
        pressure     — Pa. Default 0 (unset).
        temperature  — K. Default 0 (unset).
        viscosity    — dynamic viscosity μ, Pa·s. Default `None` → filled
                       from Sutherland's law for air at `temperature`
                       (or ISA sea level if temperature is unset), giving
                       ~1.79e-5 for a bare air medium. Liquids and exotic
                       gases pass μ explicitly (seawater ≈ 1.35e-3).
    """

    field_value_shape = FluidState
    combining = "baseline"

    def __init__(self,
                 density: float,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 *,
                 pressure: float = 0.0,
                 temperature: float = 0.0,
                 viscosity: float | None = None,
                 name: str | None = None,
                 combining: str | None = None,
                 membership=None) -> None:
        super().__init__(name=name, combining=combining, membership=membership)
        self.density     = float(density)
        self.pressure    = float(pressure)
        self.temperature = float(temperature)
        self.velocity    = tuple(float(x) for x in velocity)
        if self.density < 0.0:
            raise ValueError(
                f"UniformFluid: density must be >= 0, got {density!r}")
        if len(self.velocity) != 3:
            raise ValueError(
                f"UniformFluid: velocity must be length-3, got {velocity!r}")
        # Default-fill viscosity from Sutherland(T) for air — a lazy import
        # keeps the fields↔planets dependency one-directional at load time.
        if viscosity is None:
            from ..planets.atmosphere import T0_ISA, sutherland_viscosity
            T = self.temperature if self.temperature > 0.0 else T0_ISA
            viscosity = float(sutherland_viscosity(T))
        self.viscosity = float(viscosity)
        if self.viscosity < 0.0:
            raise ValueError(
                f"UniformFluid: viscosity must be >= 0, got {viscosity!r}")

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density     = ca.MX(self.density),
            pressure    = ca.MX(self.pressure),
            temperature = ca.MX(self.temperature),
            viscosity   = ca.MX(self.viscosity),
            velocity    = _VEC3_W.constant(self.velocity),
        )

    def __repr__(self) -> str:
        return (f"<UniformFluid density={self.density} "
                f"pressure={self.pressure} temperature={self.temperature} "
                f"viscosity={self.viscosity:.3e} velocity={self.velocity}>")


class CurrentFlow(Disturbance):
    """Localized current — an additive velocity perturbation that leaves
    density / pressure / temperature untouched.

    v1 ships the simplest non-spatial model: a constant velocity
    contribution everywhere (bound it with a `membership=` for a pocket).
    Future versions will accept a Gaussian envelope or a tabulated map.

    Args:
        velocity — world-frame velocity contribution, m/s.
    """

    field_value_shape = FluidState
    combining = "additive"

    def __init__(self,
                 velocity: tuple[float, float, float],
                 *,
                 name: str | None = None,
                 combining: str | None = None,
                 membership=None) -> None:
        super().__init__(name=name, combining=combining, membership=membership)
        self.velocity = tuple(float(x) for x in velocity)
        if len(self.velocity) != 3:
            raise ValueError(
                f"CurrentFlow: velocity must be length-3, got {velocity!r}")

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density     = ca.MX(0.0),
            pressure    = ca.MX(0.0),
            temperature = ca.MX(0.0),
            velocity    = _VEC3_W.constant(self.velocity),
        )

    def __repr__(self) -> str:
        return f"<CurrentFlow velocity={self.velocity}>"


class WeatherPatch(Disturbance):
    """Local thermodynamic perturbation — additive temperature / pressure
    (and optional density) deltas layered on top of the ambient regime.

    The planet's `Atmosphere` / `Ocean` baseline already gives a sane
    *average* (T, P, ρ) everywhere; a `WeatherPatch` is how a user paints
    a custom LOCAL curve over it — a warm thermal, a low-pressure cell, a
    surface inversion — without touching the baseline. Bind it to a
    region with `membership=` (e.g. `within_sphere(...)`); the default is
    global.

    Each of `temperature` (K), `pressure` (Pa) and `density` (kg/m³) is
    either a constant or a callable `(point: Vec3[WorldFrame], t) ->
    ca.MX` for a position/time-varying field — so a curve is just a
    Python function of the query point. They are **deltas**: they add to
    whatever the baseline (and any other patches) already report.

    Note: this perturbs the (T, P, ρ) components independently — it does
    NOT re-impose the ideal-gas tie between them. That is deliberate (the
    user is authoring the curve they want); pass whichever components you
    care about and leave the rest at 0.

    Args:
        temperature — K delta. Constant or `(point, t) -> MX`. Default 0.
        pressure    — Pa delta. Constant or `(point, t) -> MX`. Default 0.
        density     — kg/m³ delta. Constant or `(point, t) -> MX`. Default 0.
    """

    field_value_shape = FluidState
    combining = "additive"

    def __init__(self,
                 *,
                 temperature=0.0,
                 pressure=0.0,
                 density=0.0,
                 name: str | None = None,
                 combining: str | None = None,
                 membership=None) -> None:
        super().__init__(name=name, combining=combining, membership=membership)
        self.temperature = temperature
        self.pressure    = pressure
        self.density     = density

    @staticmethod
    def _eval(field, point, t):
        """A constant or a `(point, t) -> MX` callable → MX scalar."""
        if callable(field):
            return field(point, t)
        return ca.MX(float(field))

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density     = self._eval(self.density, point, t),
            pressure    = self._eval(self.pressure, point, t),
            temperature = self._eval(self.temperature, point, t),
            velocity    = _VEC3_W.constant((0.0, 0.0, 0.0)),
        )

    def __repr__(self) -> str:
        def _s(v):
            return "fn" if callable(v) else repr(v)
        return (f"<WeatherPatch temperature={_s(self.temperature)} "
                f"pressure={_s(self.pressure)} density={_s(self.density)}>")
