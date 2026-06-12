"""GravitySource — adds an inverse-square gravity disturbance.

Bolt it to a craft that stands in for a massive body (a planet, asteroid,
station, or a heavy vehicle) and other craft feel its pull. It adds a
`BodyPointMassGravity` disturbance to the world's `GravityField` (which
must be registered); the disturbance rides this craft's pose, so the
source can move.

Note this is the gravity OTHER craft feel; it does not add the source's
mass to its own carrier's dynamics — give the carrier a `Mass` for that.
A craft that carries a GravitySource also feels it (superposition), but
the eps softening keeps the self-pull at the center finite (~0).
"""

from __future__ import annotations

from ...fields.gravity import BodyPointMassGravity, GravityField
from .._declarations import Parameter
from .base import FieldSource


class GravitySource(FieldSource):
    """Adds a point-mass gravity disturbance that rides the carrying craft.

    Parameters:
        GM  — gravitational parameter G·M (m³/s²). Earth ≈ 3.986e14,
              Moon ≈ 4.903e12, a 1000-ton asteroid ≈ 6.7e-5.
        eps — softening length (m) capping the singularity at the source.
    """

    emits_field = GravityField

    GM:  float = Parameter(1.0e6)
    eps: float = Parameter(1.0)

    def make_disturbance(self, craft, offset_body):
        return BodyPointMassGravity(
            craft, offset_body, GM=float(self.GM), eps=float(self.eps),
            name=f"{craft.name}_{self.name}")
