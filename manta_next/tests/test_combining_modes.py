"""Disturbance combining modes — additive / averaged / projected.

Three-stage fold inside `Field.state_at_sym`:

  1. additive  — straight sum.
  2. averaged  — running-additive + every averaged contribution are
                 averaged together (arithmetic mean).
  3. projected — Gram-Schmidt residual; only orthogonal-or-extending
                 components add to the running sum.

These tests exercise each mode in isolation on a FluidField (the
canonical use case — wind/current contributions).
"""

import casadi as ca
import numpy as np

from manta_next.fields import (
    Disturbance, FluidField, FluidState,
)
from manta_next.ir.frames import WorldFrame
from manta_next.ir.types import Vec3


_VEC3 = Vec3[WorldFrame]


class _ConstVelocity(Disturbance):
    """Position-independent velocity contribution (test helper)."""
    field_value_shape = FluidState

    def __init__(self, vx, vy, vz, *,
                 combining: str = "additive",
                 name: str | None = None) -> None:
        super().__init__(name=name, combining=combining)
        self._v = (float(vx), float(vy), float(vz))

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density=ca.MX(0.0),
            velocity=_VEC3.constant(self._v),
        )


def _eval_velocity(ff: FluidField) -> np.ndarray:
    s = ff.state_at_sym(_VEC3.constant((0.0, 0.0, 0.0)), ca.MX(0.0))
    return np.asarray(ca.evalf(s.velocity._mx)).ravel()


# ---------------------------------------------------------------------------

def test_additive_sums():
    ff = FluidField()
    ff.add(_ConstVelocity(1.0, 0.0, 0.0))
    ff.add(_ConstVelocity(0.0, 2.0, 0.0))
    v = _eval_velocity(ff)
    np.testing.assert_allclose(v, (1.0, 2.0, 0.0), atol=1e-12)


def test_averaged_takes_mean():
    """One additive + two averaged → average of all three."""
    ff = FluidField()
    ff.add(_ConstVelocity(3.0, 0.0, 0.0, combining="additive"))
    ff.add(_ConstVelocity(0.0, 6.0, 0.0, combining="averaged"))
    ff.add(_ConstVelocity(0.0, 0.0, 9.0, combining="averaged"))
    # Mean of (3,0,0), (0,6,0), (0,0,9) = (1, 2, 3).
    v = _eval_velocity(ff)
    np.testing.assert_allclose(v, (1.0, 2.0, 3.0), atol=1e-12)


def test_averaged_alone_means_zero_with_contribs():
    """No additive: averaged mean of (zero + every averaged term)."""
    ff = FluidField()
    ff.add(_ConstVelocity(4.0, 0.0, 0.0, combining="averaged"))
    ff.add(_ConstVelocity(0.0, 4.0, 0.0, combining="averaged"))
    # Mean of (0,0,0), (4,0,0), (0,4,0) = (4/3, 4/3, 0).
    v = _eval_velocity(ff)
    np.testing.assert_allclose(v, (4.0/3.0, 4.0/3.0, 0.0), atol=1e-12)


def test_projected_orthogonal_passes_through():
    """A projected contribution orthogonal to the running sum is
    added in full."""
    ff = FluidField()
    ff.add(_ConstVelocity(5.0, 0.0, 0.0, combining="additive"))
    ff.add(_ConstVelocity(0.0, 3.0, 0.0, combining="projected"))
    v = _eval_velocity(ff)
    np.testing.assert_allclose(v, (5.0, 3.0, 0.0), atol=1e-12)


def test_projected_aligned_and_smaller_is_dropped():
    """A projected contribution aligned with the running sum and
    smaller in magnitude contributes zero (the cap on positive
    projection)."""
    ff = FluidField()
    ff.add(_ConstVelocity(5.0, 0.0, 0.0, combining="additive"))
    ff.add(_ConstVelocity(2.0, 0.0, 0.0, combining="projected"))
    v = _eval_velocity(ff)
    np.testing.assert_allclose(v, (5.0, 0.0, 0.0), atol=1e-12)


def test_projected_antialigned_added_in_full():
    """An antialigned projected contribution has negative projection
    on the running sum → clipped to 0 → entire vector is added as
    residual."""
    ff = FluidField()
    ff.add(_ConstVelocity(5.0, 0.0, 0.0, combining="additive"))
    ff.add(_ConstVelocity(-2.0, 0.0, 0.0, combining="projected"))
    v = _eval_velocity(ff)
    np.testing.assert_allclose(v, (3.0, 0.0, 0.0), atol=1e-12)
