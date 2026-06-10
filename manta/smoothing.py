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
