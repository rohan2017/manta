"""FluidField + concrete fluid disturbances.

A FluidField returns `FluidState(density, velocity)` at a queried
world-frame point. The value is compound — density (Scalar, kg/m³)
plus bulk flow velocity (Vec3[WorldFrame], m/s) — so `FluidState`
implements `__add__` for per-field-component summation, letting the
generic `Field.value_at_sym` superposition machinery work unchanged.

Density and pressure/temperature gas modeling are deferred — v1 ships
density (incompressible-fluid surrogate) + bulk velocity. Future work:
add `pressure`, `temperature` to FluidState and derive density via the
ideal-gas law when configured for gases.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from .base import Disturbance, SuperposedField


_VEC3_ANCHOR = Vec3[WorldFrame]


@dataclass(frozen=True)
class FluidState:
    """Local fluid properties at an world-frame point.

    density   — kg/m³. CasADi-MX scalar (so it composes with symbolic
                state in tracing).
    velocity  — bulk fluid velocity at the point, Vec3[WorldFrame].

    Disturbances and `FluidField.value_at_sym` return / consume this
    type. Per-component addition is defined so the Field base class can
    sum disturbances without special-casing this field.
    """
    density: ca.MX                  # scalar MX
    velocity: "Vec3"                # Vec3[WorldFrame]

    def __add__(self, other: "FluidState") -> "FluidState":
        return FluidState(
            density  = self.density + other.density,
            velocity = self.velocity + other.velocity,
        )


class FluidField(SuperposedField):
    """Fluid density + bulk velocity over the world frame.

    The field value at a point is a `FluidState`. Concrete sources
    (uniform background, localized currents, planet-registered ocean +
    atmosphere) are added as Disturbance subclasses to one FluidField
    instance.

    Builder methods (`add_uniform`, `add_current`) return self for
    chaining. The constructor accepts `density=` for the common
    single-uniform case (no flow) — equivalent to one `add_uniform`.
    """

    # Sentinel for the type-check in Field.add — any disturbance whose
    # `field_value_shape` is FluidState may be added.
    value_shape = FluidState

    def __init__(self,
                 density: float | None = None,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
                 ) -> None:
        super().__init__()
        if density is not None:
            self.add_uniform(density, velocity)

    def _zero_value(self) -> FluidState:
        return FluidState(
            density  = ca.MX(0.0),
            velocity = _VEC3_ANCHOR.constant((0.0, 0.0, 0.0)),
        )

    def _scale(self, value: FluidState, scalar: float) -> FluidState:
        return FluidState(
            density  = scalar * value.density,
            velocity = _VEC3_ANCHOR.from_mx(scalar * value.velocity._mx),
        )

    def _project_combine(self, running: FluidState,
                          contribution: FluidState) -> FluidState:
        """Projected combination for a FluidState. Density adds linearly
        (no projection); velocity uses the Gram-Schmidt residual rule."""
        new_density = running.density + contribution.density
        r_mx = running.velocity._mx
        c_mx = contribution.velocity._mx
        denom = ca.dot(r_mx, r_mx) + 1e-12
        proj_scalar = ca.dot(c_mx, r_mx) / denom
        proj_clip   = ca.fmax(0.0, proj_scalar)
        residual    = c_mx - proj_clip * r_mx
        new_velocity = _VEC3_ANCHOR.from_mx(r_mx + residual)
        return FluidState(density=new_density, velocity=new_velocity)

    def add_uniform(self,
                    density: float,
                    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
                    ) -> "FluidField":
        """Attach a uniform density (+ optional flow). Returns self."""
        return self.add(UniformFluid(density, velocity))

    def add_current(self,
                    velocity: tuple[float, float, float]) -> "FluidField":
        """Attach a flow contribution at zero density. Returns self."""
        return self.add(CurrentFlow(velocity))


# ---------------------------------------------------------------------------
# Disturbance subclasses
# ---------------------------------------------------------------------------

class UniformFluid(Disturbance):
    """Position-independent fluid: constant density + (optional) flow.

    Args:
        density   — kg/m³. Common values: ~1.225 (air), ~1025 (seawater),
                    ~1000 (fresh water).
        velocity  — bulk flow vector in WorldFrame, m/s. Default zero.
    """

    field_value_shape = FluidState

    def __init__(self,
                 density: float,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0), *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.density  = float(density)
        self.velocity = tuple(float(x) for x in velocity)
        if self.density < 0.0:
            raise ValueError(
                f"UniformFluid: density must be >= 0, got {density!r}")
        if len(self.velocity) != 3:
            raise ValueError(
                f"UniformFluid: velocity must be length-3, got {velocity!r}")

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density  = ca.MX(self.density),
            velocity = _VEC3_ANCHOR.constant(self.velocity),
        )

    def __repr__(self) -> str:
        return (f"<UniformFluid density={self.density} "
                f"velocity={self.velocity}>")


class CurrentFlow(Disturbance):
    """Localized current — adds a velocity contribution without changing
    density.

    v1 ships the simplest non-spatial model: a constant velocity
    contribution everywhere. Future versions will accept a Gaussian
    envelope around a centroid, or a tabulated current map. For now
    `CurrentFlow((0, 1, 0))` adds 1 m/s in +y to whatever density the
    background UniformFluid provides.

    Args:
        velocity — world-frame velocity contribution, m/s.
    """

    field_value_shape = FluidState

    def __init__(self, velocity: tuple[float, float, float], *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.velocity = tuple(float(x) for x in velocity)
        if len(self.velocity) != 3:
            raise ValueError(
                f"CurrentFlow: velocity must be length-3, got {velocity!r}")

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density  = ca.MX(0.0),
            velocity = _VEC3_ANCHOR.constant(self.velocity),
        )

    def __repr__(self) -> str:
        return f"<CurrentFlow velocity={self.velocity}>"


