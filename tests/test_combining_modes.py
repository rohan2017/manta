"""FluidField combining modes — baseline / averaged / additive.

`FluidField.value_at_sym` folds disturbances per their `combining` role,
per FluidState component:

  * baseline  — regime media, LAYERED by membership in insertion order
                (`base ← (1−w)·base + w·value`). A background overlaid by
                a pocket gives an alpha-composite override, not a sum.
  * averaged  — a membership-weighted self-mean among the averaged
                disturbances (overlapping estimates agree).
  * additive  — a membership-weighted sum of perturbations, on top.

`result = baseline + averaged + additive`. These tests exercise each
mode in isolation on a FluidField (the canonical use case).
"""

import casadi as ca
import numpy as np
import pytest

from manta.fields import Disturbance, FluidField, FluidState
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3


_VEC3 = Vec3[WorldFrame]


class _Const(Disturbance):
    """Position-independent fluid contribution (test helper)."""
    field_value_shape = FluidState

    def __init__(self, *, density=0.0, velocity=(0.0, 0.0, 0.0),
                 pressure=0.0, temperature=0.0,
                 combining="additive", membership=None,
                 name: str | None = None) -> None:
        super().__init__(name=name, combining=combining, membership=membership)
        self._d = float(density)
        self._v = tuple(float(x) for x in velocity)
        self._p = float(pressure)
        self._T = float(temperature)

    def contribute_at_sym(self, point, t) -> FluidState:
        return FluidState(
            density=ca.MX(self._d), pressure=ca.MX(self._p),
            temperature=ca.MX(self._T), viscosity=ca.MX(0.0),
            velocity=_VEC3.constant(self._v))


def _const_membership(w: float):
    """A constant membership callable (for exercising the weighting)."""
    return lambda point, t: ca.MX(float(w))


def _eval(ff: FluidField):
    s = ff.value_at_sym(_VEC3.constant((0.0, 0.0, 0.0)), ca.MX(0.0))
    return (float(ca.evalf(s.density)),
            np.asarray(ca.evalf(s.velocity._mx)).ravel())


# ---------------------------------------------------------------------------
# empty / vacuum
# ---------------------------------------------------------------------------

def test_empty_field_is_zero():
    rho, v = _eval(FluidField())
    assert rho == 0.0
    np.testing.assert_allclose(v, np.zeros(3), atol=1e-12)


def test_no_baseline_means_vacuum_density():
    """Only a perturbation (no regime medium) → zero density, but its
    velocity still shows up."""
    ff = FluidField()
    ff.add(_Const(velocity=(1.0, 0.0, 0.0), combining="additive"))
    rho, v = _eval(ff)
    assert rho == 0.0
    np.testing.assert_allclose(v, (1.0, 0.0, 0.0), atol=1e-12)


# ---------------------------------------------------------------------------
# additive perturbations
# ---------------------------------------------------------------------------

def test_additive_velocity_sums():
    ff = FluidField()
    ff.add(_Const(velocity=(1.0, 0.0, 0.0), combining="additive"))
    ff.add(_Const(velocity=(0.0, 2.0, 0.0), combining="additive"))
    _, v = _eval(ff)
    np.testing.assert_allclose(v, (1.0, 2.0, 0.0), atol=1e-12)


def test_additive_on_top_of_baseline():
    """A baseline medium plus an additive perturbation: density from the
    baseline, velocity = baseline + perturbation."""
    ff = FluidField()
    ff.add(_Const(density=1000.0, velocity=(0.0, 0.0, -0.2), combining="baseline"))
    ff.add(_Const(velocity=(1.0, 0.0, 0.0), combining="additive"))
    rho, v = _eval(ff)
    assert rho == 1000.0
    np.testing.assert_allclose(v, (1.0, 0.0, -0.2), atol=1e-12)


# ---------------------------------------------------------------------------
# baseline regimes — layered override, NOT summed
# ---------------------------------------------------------------------------

def test_two_global_baselines_override_not_sum():
    """Two global baselines do NOT sum (1000 + 25 ≠ 1025): the later one
    overrides (membership 1)."""
    ff = FluidField()
    ff.add(_Const(density=1000.0, combining="baseline"))
    ff.add(_Const(density=25.0, combining="baseline"))
    rho, _ = _eval(ff)
    assert rho == 25.0


def test_baseline_pocket_blends_over_background():
    """A global background (density 1) overlaid by a pocket (density 1000,
    membership 0.25) → (1−0.25)·1 + 0.25·1000 = 250.75. No dilution to a
    bare average, no sum."""
    ff = FluidField()
    ff.add(_Const(density=1.0, combining="baseline"))
    ff.add(_Const(density=1000.0, combining="baseline",
                  membership=_const_membership(0.25)))
    rho, _ = _eval(ff)
    np.testing.assert_allclose(rho, 0.75 * 1.0 + 0.25 * 1000.0, atol=1e-9)


# ---------------------------------------------------------------------------
# averaged overlays — membership-weighted self-mean
# ---------------------------------------------------------------------------

def test_averaged_takes_self_mean():
    """Two equally-weighted averaged contributions → their mean (no zero
    seed, unlike the old 'averaged' stage)."""
    ff = FluidField()
    ff.add(_Const(velocity=(4.0, 0.0, 0.0), combining="averaged"))
    ff.add(_Const(velocity=(0.0, 4.0, 0.0), combining="averaged"))
    _, v = _eval(ff)
    np.testing.assert_allclose(v, (2.0, 2.0, 0.0), atol=1e-9)


def test_averaged_weighted_by_membership():
    """Memberships weight the mean: 0.75·a + 0.25·b over (0.75+0.25).

    Full coverage (0.75 + 0.25 = 1) is exactly the kink of the saturated
    denominator `smooth_max0(den − 1) + 1`, where the C^∞ rounding costs
    0.5·sqrt(_COVERAGE_EPS_SQ) = 5e-7 of denominator — a ~5e-7 relative
    haircut on the mean. That rounding is the whole point (it is what
    keeps the fringe Jacobian bounded), so the tolerance admits it.
    """
    ff = FluidField()
    ff.add(_Const(velocity=(8.0, 0.0, 0.0), combining="averaged",
                  membership=_const_membership(0.75)))
    ff.add(_Const(velocity=(0.0, 8.0, 0.0), combining="averaged",
                  membership=_const_membership(0.25)))
    _, v = _eval(ff)
    np.testing.assert_allclose(v, (6.0, 2.0, 0.0), rtol=2e-6, atol=1e-9)


def test_averaged_and_additive_compose():
    """result = baseline + averaged + additive, per component."""
    ff = FluidField()
    ff.add(_Const(density=1.2, combining="baseline"))
    ff.add(_Const(velocity=(0.0, 6.0, 0.0), combining="averaged"))
    ff.add(_Const(velocity=(0.0, 0.0, 9.0), combining="averaged"))
    ff.add(_Const(velocity=(3.0, 0.0, 0.0), combining="additive"))
    rho, v = _eval(ff)
    assert rho == 1.2
    # averaged mean of (0,6,0),(0,0,9) = (0,3,4.5); + additive (3,0,0).
    np.testing.assert_allclose(v, (3.0, 3.0, 4.5), atol=1e-9)


# ---------------------------------------------------------------------------
# combining validation — a typo must raise, not vanish from the physics
# ---------------------------------------------------------------------------

def test_typo_combining_kwarg_raises():
    import pytest
    with pytest.raises(ValueError, match="combining"):
        _Const(density=1000.0, combining="Baseline")


def test_typo_combining_class_attr_raises():
    """A subclass fixing `combining` as a class attribute (the documented
    way) is validated at construction too — a typo'd class attr once made
    the fluid silently vanish (density 0, no error)."""
    import pytest

    class _Regime(_Const):
        combining = "Baseline"          # typo'd capital

    with pytest.raises(ValueError, match="combining"):
        _Regime(density=1000.0, combining=None)


def test_mutated_combining_raises_at_composition():
    """Even an instance mutated AFTER construction fails loudly when the
    field composes, instead of dropping out of every bucket."""
    import pytest
    d = _Const(density=1000.0, combining="baseline", name="sea")
    ff = FluidField().add(d)
    d.combining = "Baseline"
    with pytest.raises(ValueError, match="unknown combining"):
        _eval(ff)


# ---------------------------------------------------------------------------
# the averaged fringe — the smooth boundary the mode used to defeat
# ---------------------------------------------------------------------------

def _fringe_profile(radius=10.0, width=2.0, speed=8.0, n=801, span=3.0):
    """Wind speed along +x through a single bounded `averaged` bubble.

    Returns `(xs, speeds)` sampled across the bubble's fringe. One
    disturbance means the self-mean is `v·w/den`, which is exactly where
    a naive `den + 1e-9` normalizer cancels the membership out and hands
    back full-strength wind wherever `w ≳ 1e-6`.
    """
    from manta.fields import within_sphere

    ff = FluidField()
    ff.add(_Const(velocity=(speed, 0.0, 0.0), combining="averaged",
                  membership=within_sphere((0.0, 0.0, 0.0), radius, width)))
    xs = np.linspace(radius - span * width, radius + span * width, n)
    out = []
    for x in xs:
        s = ff.value_at_sym(_VEC3.constant((float(x), 0.0, 0.0)), ca.MX(0.0))
        out.append(np.asarray(ca.evalf(s.velocity._mx)).ravel()[0])
    return xs, np.asarray(out)


def test_averaged_fringe_decays_instead_of_holding_full_strength():
    """Well outside the bubble the wind is gone, and at the nominal
    radius it is part-strength — not the full 8 m/s the cancelled
    membership used to hand back."""
    radius, width, speed = 10.0, 2.0, 8.0
    xs, v = _fringe_profile(radius, width, speed)
    inside = v[np.argmin(np.abs(xs - (radius - 3.0 * width)))]
    edge = v[np.argmin(np.abs(xs - radius))]
    outside = v[np.argmin(np.abs(xs - (radius + 3.0 * width)))]
    assert inside == pytest.approx(speed, rel=1e-5)
    assert 0.1 * speed < edge < 0.9 * speed
    assert outside == pytest.approx(0.0, abs=1e-9)


def test_averaged_fringe_derivative_is_bounded():
    """The whole point of `boundary=`: a C¹ ramp. The old `den + 1e-9`
    normalizer put a `w′/ε` spike here — of order 1e9 — which is a
    near-discontinuity for every Jacobian downstream (the EKF's F, the
    LQR's A/B). The saturated denominator caps the slope at the scale
    the blend width actually implies."""
    radius, width, speed = 10.0, 2.0, 8.0
    xs, v = _fringe_profile(radius, width, speed)
    slope = np.abs(np.diff(v) / np.diff(xs))
    # A ramp from `speed` to 0 over ~2·width cannot be steeper than a
    # small multiple of speed/width; the regularizer spike was ~1e9.
    assert slope.max() < 5.0 * speed / width


def test_averaged_fringe_is_monotone_with_no_step():
    radius, width = 10.0, 2.0
    _, v = _fringe_profile(radius, width)
    assert np.all(np.diff(v) <= 1e-9)                  # non-increasing
    assert np.abs(np.diff(v)).max() < 0.05             # no jump
