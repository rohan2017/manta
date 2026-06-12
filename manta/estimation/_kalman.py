"""Joseph-form Kalman measurement update — the shared symbolic kernel.

Both symbolic Kalman recursions fold measurements through this one
helper: `EKF`'s per-sensor update kernels (R baked numeric) and
`NoiseFit`'s innovation-NLL filter step (R an MX expression in the
fitted σ) — CasADi MX subsumes constants, so one parametrization covers
both. The numeric (numpy) covariance recursion in
`manta.estimation.observability.sigma_horizon` mirrors this form; this
helper is its reference implementation.
"""

from __future__ import annotations

import casadi as ca

from ..ir._linalg import spd_solve
from ..ir.state_spec import StateSpec


def joseph_update(x: ca.MX, P: ca.MX, h: ca.MX, H: ca.MX, R: ca.MX,
                  z: ca.MX, spec: StateSpec
                  ) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    """One manifold-correct Joseph-form measurement update:

        ν  = z − h
        S  = H P Hᵀ + R
        K  = P Hᵀ S⁻¹                          (ldl solve — S is SPD)
        x⁺ = x ⊞ K ν                           (manifold boxplus)
        P⁺ = (I−KH) P (I−KH)ᵀ + K R Kᵀ         (Joseph form),
             re-symmetrized as ½(P⁺ + P⁺ᵀ)

    Returns `(x_new, P_new, nu, S)` — ν and S so likelihood-style
    callers (NoiseFit) can build their innovation NLL without
    recomputing the solve's ingredients.
    """
    nu = z - h
    S = H @ P @ H.T + R
    K = spd_solve(S, (P @ H.T).T).T            # P Hᵀ S⁻¹ (S SPD)
    x_new = spec.boxplus_sym(x, K @ nu)
    IKH = ca.MX.eye(P.size1()) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new, nu, S
