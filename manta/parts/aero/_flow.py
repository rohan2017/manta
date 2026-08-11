"""Shared aero plumbing.

`signed_powers` is the house polynomial convention of the damping family
(DragSurface, RotationalDrag, FossenDamping): the element-wise ladder
``x^(k) = x·|x|^(k-1)`` — direction-preserving, so k=2 is the correct
quadratic drag ``v·|v|`` rather than ``v²``, which would push a backwards
craft forwards. ``|x|`` carries a 1e-30 bleed so the derivative stays
regular at ``x_i = 0``, where the EKF's F and the LQR's A/B linearize.

The three parts wrote this ladder three times; the convention is a
physics contract between them (a part that got the sign or the softening
wrong would be subtly, silently wrong at low speed), so it lives once.
"""

from __future__ import annotations

import casadi as ca

# One polynomial-order cap for the aero tensor families. Beyond quartic
# the sign-preserving powers stop corresponding to any fluid-dynamic
# regime worth fitting.
MAX_ORDER = 4


def signed_powers(x_mx, order: int) -> list:
    """``[x^(1), …, x^(order)]`` with ``x^(k) = x·|x|^(k-1)`` element-wise
    (k=1 is x itself). Returns an empty list for ``order <= 0``."""
    if order <= 0:
        return []
    powers = [x_mx]
    abs_x = ca.fabs(x_mx) + 1e-30
    for _ in range(1, order):
        powers.append(powers[-1] * abs_x)
    return powers
