"""Joseph-form Kalman math — the shared kernels (symbolic + numeric).

Every Kalman recursion in manta folds through the helpers here, so the
covariance update, the `L Σ Lᵀ` noise assembly, the re-symmetrization, and
the singular-R guard each live in exactly one place:

  * `EKF`            — per-sensor symbolic update kernels (R baked numeric).
  * `NoiseFit`       — innovation-NLL filter step (R an MX expression in σ).
  * `sigma_horizon`  — the numeric covariance recursion over a horizon.
  * the numpy filter — the runtime caller-supplied `h(x)` update.

CasADi MX and numpy arrays share the `+` / `@` / `.T` surface these use, so
`symmetrize` and `lin_cov` are backend-blind. `joseph_update` is the
symbolic (manifold-boxplus) kernel; `joseph_update_np` is its numeric twin.
"""

from __future__ import annotations

from typing import Callable

import casadi as ca
import numpy as np

from ..ir._linalg import spd_solve
from ..ir.state_spec import StateSpec


def symmetrize(M):
    """½(M + Mᵀ) — strip the asymmetric drift a covariance update leaves.

    Backend-blind: works on a CasADi MX or a numpy array."""
    return 0.5 * (M + M.T)


def lin_cov(L, Sigma, dim: int):
    """`L Σ Lᵀ` — a linearized channel covariance, or a `dim×dim` zero when
    the model declares no noise on this map (`L is None`).

    Σ may be numeric (baked, `ca.DM`) or a symbolic σ expression; `L` is the
    process noise Jacobian ∂δ'/∂noise (→ Q) or a sensor's ∂h/∂noise (→ R)."""
    if L is None:
        return ca.MX.zeros(dim, dim)
    return L @ Sigma @ L.T


def joseph_update(x: ca.MX, P: ca.MX, h: ca.MX, H: ca.MX, R: ca.MX,
                  z: ca.MX, spec: StateSpec
                  ) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    """One manifold-correct Joseph-form measurement update (symbolic):

        ν  = z − h
        S  = H P Hᵀ + R
        K  = P Hᵀ S⁻¹                          (ldl solve — S is SPD)
        x⁺ = x ⊞ K ν                           (manifold boxplus)
        P⁺ = (I−KH) P (I−KH)ᵀ + K R Kᵀ         (Joseph form), re-symmetrized

    Returns `(x_new, P_new, nu, S)` — ν and S so likelihood-style callers
    (NoiseFit) can build their innovation NLL without recomputing the
    solve's ingredients.
    """
    nu = z - h
    S = H @ P @ H.T + R
    K = spd_solve(S, (P @ H.T).T).T            # P Hᵀ S⁻¹ (S SPD)
    x_new = spec.boxplus_sym(x, K @ nu)
    IKH = ca.MX.eye(P.size1()) - K @ H
    P_new = symmetrize(IKH @ P @ IKH.T + K @ R @ K.T)
    return x_new, P_new, nu, S


def joseph_update_np(P: np.ndarray, H: np.ndarray, R: np.ndarray, *,
                     x: np.ndarray | None = None,
                     z: np.ndarray | None = None,
                     h: np.ndarray | None = None,
                     boxplus: Callable | None = None
                     ) -> tuple:
    """Numeric twin of `joseph_update`.

    Always folds the covariance (`S = HPHᵀ+R`, `K = PHᵀS⁻¹`, Joseph `P⁺`,
    re-symmetrized). It also updates the state when `x`/`z`/`h`/`boxplus`
    are supplied; a covariance-only horizon (`sigma_horizon`) passes just
    `P, H, R`. Returns `(x_new, P_new, nu, S)` with `x_new`/`nu` left `None`
    in the covariance-only case.
    """
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T            # P Hᵀ S⁻¹ (S SPD)
    IKH = np.eye(P.shape[0]) - K @ H
    P_new = symmetrize(IKH @ P @ IKH.T + K @ R @ K.T)
    x_new = nu = None
    if x is not None:
        nu = z - h
        x_new = boxplus(x, K @ nu)
    return x_new, P_new, nu, S


def require_active_R(R: ca.MX, R_fn: ca.Function, syms: ca.MX, *,
                     x0: np.ndarray, u_defaults: np.ndarray, spec: StateSpec,
                     full: str, who: str, extra: tuple = (),
                     diag: bool = False) -> None:
    """Refuse a sensor whose baked R is (numerically) zero.

    A σ=0 sensor bakes R = 0: the first update collapses its covariance
    block to exactly zero, the second meets a singular S and NaNs silently.
    R can be state-dependent (L_h varying with x), so probe a perturbed
    point too when R depends on `syms` — a sampled check, not a proof; an R
    that vanishes only away from both probes still slips through.

    `R_fn(x, u, t, *extra)` evaluates R numerically; `extra` carries any
    trailing fixed args (NoiseFit's starting σ). `diag=True` checks only the
    diagonal (a diagonal R with structurally-zero off-diagonals).
    """
    probes = [(x0, u_defaults, 0.0)]
    if ca.depends_on(R, syms):
        x1 = spec.boxplus_num(np.asarray(x0, dtype=float).reshape(-1),
                              1e-3 * np.ones(spec.tangent_dim))
        probes.append((x1, u_defaults + 1e-3, 1.0))

    def zero_at(p) -> bool:
        M = np.asarray(ca.DM(R_fn(*p, *extra)))
        vals = np.diag(M) if diag else M
        return not np.any(np.abs(vals) > 0.0)

    if all(zero_at(p) for p in probes):
        raise ValueError(
            f"{who}: sensor {full!r} has no active noise channel — its baked "
            f"R is zero and the first update is singular (the next fold "
            f"NaNs). Declare a nonzero noise σ on the sensor, or exclude it "
            f"via sensors=[...].")
