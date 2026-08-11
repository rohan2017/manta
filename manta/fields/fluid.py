"""FluidField + concrete fluid disturbances.

A FluidField returns `FluidState(density, pressure, temperature,
viscosity, velocity)` at a queried world-frame point. The value is
compound — four scalars (density kg/m³, pressure Pa, temperature K,
dynamic viscosity Pa·s) plus a bulk flow velocity (Vec3[WorldFrame],
m/s). Velocity comes last in every signature.

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
from ..smoothing import hermite_blend, smooth_max0, soft_norm
from .base import Disturbance, SuperposedField
from .fluid_props import T0_ISA, hydrostatic_pressure, sutherland_viscosity


_VEC3_W = Vec3[WorldFrame]

# Rounding half-width² for the averaged-pool coverage saturation at
# den = 1 (coverage is dimensionless, so this is (1e-6)² in coverage
# units). Small on purpose: the derivative bound does NOT depend on it
# (smooth_max0's slope is ≤ 1 for any eps), it only sets the value error
# where bubbles exactly tile coverage 1 — √eps/2 = 5e-7, far below any
# wind-estimate resolution — and keeps the ≥-1 branch's mean exact to
# eps/4 = 2.5e-13.
_COVERAGE_EPS_SQ: float = 1e-12


@dataclass(frozen=True)
class FluidState:
    """Local fluid properties at a world-frame point.

    Fields are ordered `density, pressure, temperature, viscosity,
    velocity` — the four scalars first, the Vec3 velocity last — and that
    order is used in every signature and call site.

    density      — kg/m³. CasADi-MX scalar (composes with symbolic state).
    pressure     — Pa. MX scalar.
    temperature  — K. MX scalar.
    viscosity    — dynamic viscosity μ, Pa·s. MX scalar. An independent
                   property (like density): a gas baseline fills it from
                   temperature via Sutherland's law, while water sets it
                   directly. Drives the Reynolds number a foil sees.
                   Perturbation/overlay disturbances that carry no
                   viscosity pass `ca.MX(0.0)`.
    velocity     — bulk fluid velocity at the point, Vec3[WorldFrame].

    Disturbances and `FluidField.value_at_sym` return / consume this
    type. `__add__` (per-component sum) backs the additive pool;
    `scaled` (per-component scalar multiply) backs the membership
    weighting in the baseline / averaged blends.
    """
    density: ca.MX                  # scalar MX
    pressure: ca.MX                 # scalar MX
    temperature: ca.MX              # scalar MX
    viscosity: ca.MX                # scalar MX
    velocity: "Vec3"                # Vec3[WorldFrame]

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

    # The layered blend below evaluates `Disturbance.membership` on every
    # role, so `Field.add` accepts a bounded disturbance here (it refuses
    # one on the plain additive fields, where it would be dropped).
    honors_membership = True

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
            viscosity   = ca.MX(0.0),
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

        buckets: dict[str, list] = {
            "baseline": [], "averaged": [], "additive": []}
        for d in self._disturbances:
            if d.combining not in buckets:
                # Construction validates `combining`, but an instance
                # mutated afterwards would otherwise just vanish from
                # every bucket — silent physics loss.
                raise ValueError(
                    f"{type(self).__name__}: disturbance {d.name!r} has "
                    f"unknown combining mode {d.combining!r} (must be "
                    f"'additive', 'averaged', or 'baseline').")
            buckets[d.combining].append(d)
        baselines = buckets["baseline"]
        averaged = buckets["averaged"]
        additive = buckets["additive"]

        # Baseline regimes: layered alpha-composite override in insertion
        # order. base ← (1 − w)·base + w·contribution.
        base = self._zero_value()
        for d in baselines:
            w = d.membership(point, t)
            c = d.contribute_at_sym(point, t)
            base = base.scaled(1.0 - w) + c.scaled(w)

        # Averaged overlays: membership-weighted self-mean with a
        # smoothly SATURATED denominator. The naive `num / (den + ε)`
        # defeats the membership ramp: for a single bubble the weight
        # cancels out of its own mean (w·v / (w + ε) ≈ v anywhere
        # w ≳ ε — full-strength wind right up to the fringe), and the
        # derivative of w/(w+ε) at w → 0 is w′/ε — the C¹ boundary a
        # CraftWindBubble promises turns into a ~1/ε-amplified
        # near-step in every Jacobian the EKF/LQR consumes. Instead,
        # divide by `smooth_max0(den − 1) + 1`: at coverage den ≥ 1 the
        # denominator is den (exact overlap mean — two full bubbles
        # agree on their average), while at den < 1 it saturates at 1,
        # so the value decays as Σ wᵢ·vᵢ (≈ v·w at a lone fringe — the
        # true C¹ ramp) and the derivative stays bounded by the
        # membership ramp's own slope w′, ε-free. The transition is C^∞
        # (see `smooth_max0`); _COVERAGE_EPS_SQ only rounds the kink at
        # den = 1.
        avg = self._zero_value()
        if averaged:
            num = self._zero_value()
            den = ca.MX(0.0)
            for d in averaged:
                w = d.membership(point, t)
                num = num + d.contribute_at_sym(point, t).scaled(w)
                den = den + w
            den_sat = smooth_max0(den - 1.0, _COVERAGE_EPS_SQ) + 1.0
            avg = num.scaled(1.0 / den_sat)

        # Additive perturbations: membership-weighted sum on top.
        pert = self._zero_value()
        for d in additive:
            w = d.membership(point, t)
            pert = pert + d.contribute_at_sym(point, t).scaled(w)

        return base + avg + pert

    def add_flat_ocean(self,
                       *,
                       density: float = 1025.0,
                       surface_z: float = 0.0,
                       surface_pressure: float = 101325.0,
                       gravity: float = 9.80665,
                       temperature: float = 288.15,
                       viscosity: float = 1.35e-3,
                       velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
                       surface_blend: float = 0.05,
                       ) -> "FluidField":
        """Attach a flat hydrostatic ocean below `surface_z`. Returns
        self. See `FlatOcean` for what it is and is not."""
        return self.add(FlatOcean(
            density=density, surface_z=surface_z,
            surface_pressure=surface_pressure, gravity=gravity,
            temperature=temperature, viscosity=viscosity,
            velocity=velocity, surface_blend=surface_blend))

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
        # Default-fill viscosity from Sutherland(T) for air (the fluid-
        # column physics lives beside us in `fluid_props`).
        if viscosity is None:
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


class FlatOcean(Disturbance):
    """Incompressible water below a horizontal free surface, in the
    WORLD frame — a local z-up ocean for a world with no planet.

    `Ocean` (in `manta.planets`) is the same physics anchored to a
    planet's curved surface. This is the flat-earth version, for a
    local ENU world where the sea surface is the plane `z = surface_z`
    and gravity is a constant −z. That is the right model whenever the
    working area is small enough that curvature is irrelevant — which,
    for a vehicle operating over a few kilometres, it is by a wide
    margin — and it keeps world coordinates in metres near the origin
    instead of near the Earth's radius, which matters for anything
    doing geometry in the world frame (bathymetry, acoustic raycasts,
    sensor mount arithmetic).

        ρ = density                          (constant)
        P = surface_pressure + ρ·g·depth     (depth = surface_z − z)
        T = temperature                      (constant)

    The regime is bounded BELOW the surface by a `below_surface`
    membership, so it does not simply define fluid everywhere. That is
    what makes a broaching hull lose buoyancy instead of floating half
    out of the water with full lift, and it is why `surface_blend`
    exists: a hard cutoff would put a step in the buoyancy force and
    every integrator downstream would ring on it.

    Above the surface the field contributes nothing, so a world that
    wants air as well overlays an `Atmosphere` or a second
    `UniformFluid` — they layer by membership.

    A flat surface, deliberately: waves belong to the planet path,
    where the surface height is already a function of position and
    time. A local world that needs them can supply its own membership
    and a matching pressure term.

    Args:
        density          — kg/m³ (1025 seawater, 1000 fresh).
        surface_z        — world z of the mean free surface.
        surface_pressure — Pa at the surface. This is the value a depth
                           sensor is zeroed against, so it belongs to
                           the world rather than to the sensor.
        gravity          — magnitude, m/s². Must match the world's
                           GravityField or buoyancy and pressure will
                           disagree about how deep things are.
        surface_blend    — metres over which the regime fades out at the
                           surface.
    """

    field_value_shape = FluidState
    combining = "baseline"

    def __init__(self,
                 *,
                 density: float = 1025.0,
                 surface_z: float = 0.0,
                 surface_pressure: float = 101325.0,
                 gravity: float = 9.80665,
                 temperature: float = 288.15,
                 viscosity: float = 1.35e-3,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 surface_blend: float = 0.05,
                 name: str | None = None,
                 combining: str | None = None) -> None:
        if density <= 0.0:
            raise ValueError(
                f"FlatOcean: density must be > 0, got {density!r}")
        if gravity < 0.0:
            raise ValueError(
                f"FlatOcean: gravity must be >= 0, got {gravity!r}")
        if surface_blend <= 0.0:
            raise ValueError(
                f"FlatOcean: surface_blend must be > 0, got "
                f"{surface_blend!r} — a hard cutoff steps the buoyancy "
                f"force and rings every integrator downstream")
        z0 = float(surface_z)
        super().__init__(name=name, combining=combining,
                         membership=below_surface(
                             lambda point, t: point._mx[2] - z0,
                             float(surface_blend)))
        self.density = float(density)
        self.surface_z = z0
        self.surface_pressure = float(surface_pressure)
        self.gravity = float(gravity)
        self.temperature = float(temperature)
        self.viscosity = float(viscosity)
        self.velocity = tuple(float(x) for x in velocity)
        self.surface_blend = float(surface_blend)
        if len(self.velocity) != 3:
            raise ValueError(
                f"FlatOcean: velocity must be length-3, got {velocity!r}")

    def contribute_at_sym(self, point, t) -> FluidState:
        # Depth below the surface, floored at zero through the same
        # smooth max the Heightfield uses — the membership already fades
        # the regime out above the surface, but the pressure expression
        # itself must not go negative inside the blend band.
        depth = smooth_max0(self.surface_z - point._mx[2],
                            self.surface_blend ** 2)
        return FluidState(
            density     = ca.MX(self.density),
            # Same incompressible column `Ocean` uses (one law, one
            # helper): P = P_surface + ρ·g·depth.
            pressure    = hydrostatic_pressure(
                              ca.MX(self.surface_pressure),
                              self.density, self.gravity, depth),
            temperature = ca.MX(self.temperature),
            viscosity   = ca.MX(self.viscosity),
            velocity    = _VEC3_W.constant(self.velocity),
        )

    def __repr__(self) -> str:
        return (f"<FlatOcean density={self.density} "
                f"surface_z={self.surface_z} "
                f"P_surface={self.surface_pressure}>")


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
            viscosity   = ca.MX(0.0),
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
            viscosity   = ca.MX(0.0),
            velocity    = _VEC3_W.constant((0.0, 0.0, 0.0)),
        )

    def __repr__(self) -> str:
        def _s(v):
            return "fn" if callable(v) else repr(v)
        return (f"<WeatherPatch temperature={_s(self.temperature)} "
                f"pressure={_s(self.pressure)} density={_s(self.density)}>")
