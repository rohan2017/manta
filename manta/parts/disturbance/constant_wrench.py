"""A deterministic body-frame wrench used to represent known model bias."""

from __future__ import annotations

import math
from typing import Iterable

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import Parameter
from ..base import Part


def _vector3(value: Iterable[float], *, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite values")
    return result


class ConstantWrench(Part):
    """A fixed force and torque in the part frame.

    This is deterministic dynamics, not process noise. Mount the part at the
    craft COM when the supplied torque is already about the COM.
    """

    force: tuple[float, float, float] = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame
    )
    torque: tuple[float, float, float] = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame
    )

    def __init__(self, name: str, *, force=(0.0, 0.0, 0.0),
                 torque=(0.0, 0.0, 0.0), **overrides) -> None:
        super().__init__(
            name,
            force=_vector3(force, name="force"),
            torque=_vector3(torque, name="torque"),
            **overrides,
        )

    def update(self, ctx):
        return Wrench(
            force=Vec3[PartFrame].constant(self.force),
            torque=Vec3[PartFrame].constant(self.torque),
        )


__all__ = ["ConstantWrench"]
