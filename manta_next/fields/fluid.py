"""FluidField + concrete fluid disturbances.

A FluidField returns `FluidState(density, velocity)` at a queried
anchor-frame point. The value is compound — density (Scalar, kg/m³)
plus bulk flow velocity (Vec3[AnchorFrame], m/s) — so `FluidState`
implements `__add__` for per-field-component summation, letting the
generic `Field.state_at_sym` superposition machinery work unchanged.

Density and pressure/temperature gas modeling are deferred — v1 ships
density (incompressible-fluid surrogate) + bulk velocity. Future work:
add `pressure`, `temperature` to FluidState and derive density via the
ideal-gas law when configured for gases.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from ..ir.frames import AnchorFrame
from ..ir.types import Vec3
from .base import Disturbance, Field


_VEC3_ANCHOR = Vec3[AnchorFrame]


@dataclass(frozen=True)
class FluidState:
    """Local fluid properties at an anchor-frame point.

    density   — kg/m³. CasADi-MX scalar (so it composes with symbolic
                state in tracing).
    velocity  — bulk fluid velocity at the point, Vec3[AnchorFrame].

    Disturbances and `FluidField.state_at_sym` return / consume this
    type. Per-component addition is defined so the Field base class can
    sum disturbances without special-casing this field.
    """
    density: ca.MX                  # scalar MX
    velocity: "Vec3"                # Vec3[AnchorFrame]

    def __add__(self, other: "FluidState") -> "FluidState":
        return FluidState(
            density  = self.density + other.density,
            velocity = self.velocity + other.velocity,
        )


class FluidField(Field):
    """Fluid density + bulk velocity over the anchor frame.

    The field value at a point is a `FluidState`. Concrete sources
    (uniform background, localized currents, planet-registered ocean +
    atmosphere) are added as Disturbance subclasses to one FluidField
    instance.
    """

    # Sentinel for the type-check in Field.add — any disturbance whose
    # `field_value_shape` is FluidState may be added.
    value_shape = FluidState

    def _zero_value(self) -> FluidState:
        return FluidState(
            density  = ca.MX(0.0),
            velocity = _VEC3_ANCHOR.constant((0.0, 0.0, 0.0)),
        )


# ---------------------------------------------------------------------------
# Disturbance subclasses
# ---------------------------------------------------------------------------

class UniformFluid(Disturbance):
    """Position-independent fluid: constant density + (optional) flow.

    Args:
        density   — kg/m³. Common values: ~1.225 (air), ~1025 (seawater),
                    ~1000 (fresh water).
        velocity  — bulk flow vector in AnchorFrame, m/s. Default zero.
    """

    field_value_shape = FluidState

    def __init__(self,
                 density: float,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.density  = float(density)
        self.velocity = tuple(float(x) for x in velocity)
        if len(self.velocity) != 3:
            raise ValueError(
                f"UniformFluid: velocity must be length-3, got {velocity!r}")

    def contribute_at_sym(self, point) -> FluidState:
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
        velocity — anchor-frame velocity contribution, m/s.
    """

    field_value_shape = FluidState

    def __init__(self, velocity: tuple[float, float, float]) -> None:
        self.velocity = tuple(float(x) for x in velocity)
        if len(self.velocity) != 3:
            raise ValueError(
                f"CurrentFlow: velocity must be length-3, got {velocity!r}")

    def contribute_at_sym(self, point) -> FluidState:
        return FluidState(
            density  = ca.MX(0.0),
            velocity = _VEC3_ANCHOR.constant(self.velocity),
        )

    def __repr__(self) -> str:
        return f"<CurrentFlow velocity={self.velocity}>"
