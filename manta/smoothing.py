"""Smoothed symbolic primitives shared by fields, couplings, and planets.

Differentiable replacements for the kinked primitives that appear in
contact, tether, and planet geometry. Every Jacobian the EKF/LQR
consumes runs through these expressions, so the constants here are
chosen for derivative health, not just value accuracy.
"""

from __future__ import annotations

import casadi as ca

# Softening added under a sqrt of a squared length (m²): sqrt(x² + EPS)
# has a finite derivative at x = 0 where sqrt(x²) does not. The value
# error is at most sqrt(EPS) = 1e-15 m — far below float noise for any
# meter-scale geometry — while the Jacobian stays regular when two
# points coincide (tether endpoints touching, a field query at a
# planet's center).
NORM_EPS_SQ: float = 1e-30


def soft_norm(v_mx: ca.MX) -> ca.MX:
    """``|v|`` with a regular derivative at v = 0 (value error ≤ 1e-15 m)."""
    return ca.sqrt(ca.dot(v_mx, v_mx) + NORM_EPS_SQ)


def smooth_max0(x_mx: ca.MX, eps_sq: float) -> ca.MX:
    """``max(0, x)`` blended smoothly over ``±sqrt(eps_sq)`` around x = 0.

    Identity: 0.5·(x + sqrt(x² + ε)) equals max(0, x) to within
    sqrt(ε)/2 at x = 0 and converges exponentially away from it, with a
    C^∞ transition — the derivative ramps from 0 to 1 instead of
    stepping. Pick ``eps_sq`` so sqrt(eps_sq) is the physical length
    over which the kink may be rounded.
    """
    return 0.5 * (x_mx + ca.sqrt(x_mx * x_mx + eps_sq))


def smoothstep(s_mx: ca.MX) -> ca.MX:
    """C¹ Hermite step on a clamped argument: 0 at s ≤ −1, 1 at s ≥ 1,
    rising monotonically through 0.5 at s = 0.

    Returns ``0.5 + 0.75·s − 0.25·s³`` after clamping ``s`` to [−1, 1].
    The derivative ``0.75·(1 − s²)`` vanishes at both ends, so the ramp
    has **compact support** — outside [−1, 1] the value is flat at 0/1
    with zero slope, and nothing leaks past the band (the property a
    1000:1 water/air density switch relies on). The flip side
    (``1 − smoothstep``) gives a falling step.
    """
    s = ca.fmax(-1.0, ca.fmin(1.0, s_mx))
    return 0.5 + 0.75 * s - 0.25 * s ** 3


def hermite_blend(x_mx: ca.MX, width: float) -> ca.MX:
    """Smooth 0→1 ramp in ``x``, rising over the half-width ``width``
    about x = 0: 0 at ``x ≤ −width``, 1 at ``x ≥ +width`` (via
    :func:`smoothstep`). With ``width <= 0`` it degrades to the hard step
    ``x > 0 ? 1 : 0`` — a sharp interface with no blend band.
    """
    if width <= 0.0:
        return ca.if_else(x_mx > 0.0, ca.MX(1.0), ca.MX(0.0))
    return smoothstep(x_mx / width)
