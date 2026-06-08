"""MagneticSource — adds a body-fixed magnetic dipole disturbance.

Models the magnetic signature of motors, permanent magnets, or
magnetized structure on a vehicle. It adds a `BodyDipoleMag` disturbance
to the world's `MagField` (which must be registered). A magnetometer on
another craft (or elsewhere on the same craft) reads the dipole field,
which swings as the source craft turns — the moment is body-fixed and
rotates with it.
"""

from __future__ import annotations

from ...fields.mag import BodyDipoleMag, MagField
from ..base import Parameter
from .base import FieldSource


class MagneticSource(FieldSource):
    """Adds a magnetic dipole disturbance that rides the carrying craft.

    Parameters:
        moment — (mx, my, mz) dipole moment in the craft BODY frame,
                 A·m². A small hobby motor magnet is ~1e-2–1e-1.
        eps    — softening length (m) at the dipole position.
    """

    disturbance_field = MagField

    moment: tuple = Parameter((0.0, 0.0, 1.0e-2))
    eps:    float = Parameter(1e-3)

    def make_disturbance(self, craft, offset_body):
        return BodyDipoleMag(
            craft, offset_body, moment_body=tuple(self.moment),
            eps=float(self.eps), name=f"{craft.name}_{self.name}")
