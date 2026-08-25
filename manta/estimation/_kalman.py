"""Joseph-form Kalman math — the shared kernels (symbolic + numeric).

Every Kalman recursion in manta folds through the helpers here, so the
covariance update, the `L Σ Lᵀ` noise assembly, the re-symmetrization, and
the singular-R guard each live in exactly one place:

  * `EKF`            — per-sensor symbolic update kernels (R baked numeric).
  * `UKF`            — the unscented twin: sigma-point predict/update built
                      from the same `lin_cov` Q/R + manifold boxplus, no
                      Jacobians (`unscented_weights` / `ut_predict` /
                      `ut_update`).
  * `NoiseFit`       — innovation-NLL filter step (R an MX expression in σ).
  * `sigma_horizon`  — the numeric covariance recursion over a horizon.
  * the numpy filter — the runtime caller-supplied `h(x)` update.

CasADi MX and numpy arrays share the `+` / `@` / `.T` surface these use, so
`symmetrize` and `lin_cov` are backend-blind. `joseph_update` is the
symbolic (manifold-boxplus) kernel; `joseph_update_np` is its numeric twin.
The unscented kernels (`ut_predict` / `ut_update`) are symbolic-only — they
propagate sigma points through the nonlinear `f`/`h` rather than a linearized
F/H, but bake into the same per-entry `ca.Function`s the EKF does.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import casadi as ca
import numpy as np

from ..ir._linalg import chol_lower, spd_solve
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


def zero_R_message(full: str, who: str) -> str:
    """The one shared zero-R refusal text — `require_active_R` raises it at
    filter construction, `sigma_horizon` raises it when a state-dependent R
    vanishes mid-recursion. One string so the two refusals can never say
    different things about the same failure."""
    return (
        f"{who}: sensor {full!r} has no active noise channel — its baked "
        f"R is zero and the first update is singular (the next fold "
        f"NaNs). Declare a nonzero noise σ on the sensor, or exclude it "
        f"via sensors=[...].")


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
        raise ValueError(zero_R_message(full, who))


# ---------------------------------------------------------------------------
# Unscented transform — the UKF's sigma-point kernels (symbolic).
#
# The error-state UKF carries the same `x` (manifold) / `P` (tangent
# covariance) pair as the EKF and the same additive `Q`/`R = L Σ Lᵀ` noise.
# It replaces the EKF's `F P Fᵀ` / `H P Hᵀ` linear pushes with a deterministic
# sample of `f`/`h`: 2n+1 sigma points spread on the tangent space by the
# scaled unscented weights, retracted onto the manifold (`x ⊞ δ`), pushed
# through the *nonlinear* map, then re-summarized. No Jacobians.
# ---------------------------------------------------------------------------


def unscented_weights(n: int, alpha: float, beta: float, kappa: float):
    """Scaled unscented weights for an `n`-dim tangent state (Wan & Van der
    Merwe). With `lam = α²(n+κ) − n` (the scaling parameter, internal),
    returns `(w_m, w_c, gamma)`:

      * `w_m`  (2n+1 means)                 — `w_m[0] = lam/(n+lam)`, the
        rest `1/(2(n+lam))`,
      * `w_c`  (2n+1 covariances)           — `w_c[0]` adds `1−α²+β`,
      * `gamma = √(n+lam)`                  — the sigma-point spread.

    `w_m` sums to 1; `w_c` need not (only its weighting of mean-centered
    deviations matters).

    Raises for `n + lam <= 0` (imaginary spread). Beware the textbook
    "small α" (e.g. 1e-3) at robotics-sized `n`: it drives `w_c[0]` toward
    `−n/α²` (≈ −10⁶ at n=13) — a huge negative central weight that makes
    the summed covariances indefinite under roundoff — while shrinking the
    spread to `α√(n+κ)`·σ, at which point the "nonlinear sampling" is a
    noisy finite difference. `α = 1` keeps every covariance weight
    positive; the UKF warns when a chosen tuning does not."""
    lam = alpha ** 2 * (n + kappa) - n
    c = n + lam
    if c <= 0.0:
        raise ValueError(
            f"unscented_weights: n + lam = {c:.3g} <= 0 for n={n}, "
            f"alpha={alpha}, kappa={kappa} — the sigma spread √(n+λ) is "
            f"imaginary. Use alpha in (0, 1] with kappa >= 0 (alpha=1, "
            f"kappa=0 is the robust default).")
    w0 = 1.0 / (2.0 * c)
    w_m = [lam / c] + [w0] * (2 * n)
    w_c = [lam / c + (1.0 - alpha ** 2 + beta)] + [w0] * (2 * n)
    return w_m, w_c, math.sqrt(c)


def sigma_deltas(P: ca.MX, gamma: float, n: int, *,
                 jitter: float = 0.0) -> list[ca.MX]:
    """The `2n+1` tangent-space sigma offsets `δ_0…δ_{2n}` of covariance `P`:
    `δ_0 = 0`, then `±γ` times each column of the lower-Cholesky factor
    (`P = L Lᵀ`). `chol_lower` keeps these scalar-unrolled so the whole UKF
    C-codegens and SX-expands exactly like the EKF.

    `P` here is an *iterated* covariance — many folds of roundoff away from
    the SPD-by-construction matrices `chol_lower` was written for — so
    `jitter > 0` (units: variance) adds `jitter·I` before factoring and
    floors the pivots at `jitter`, turning a marginally-indefinite `P` into
    a regularized factor instead of an all-NaN one."""
    if jitter > 0.0:
        P = P + jitter * ca.MX.eye(n)
    L = chol_lower(P, n, floor=jitter)
    cols = [ca.vertcat(*[L[r][c] if c <= r else ca.MX.zeros(1, 1)
                         for r in range(n)])
            for c in range(n)]
    return ([ca.MX.zeros(n, 1)]
            + [gamma * col for col in cols]
            + [-gamma * col for col in cols])


def manifold_mean(points: list[ca.MX], w_m: list[float], spec: StateSpec,
                  iters: int) -> ca.MX:
    """Weighted Fréchet-style mean of manifold `points` (ambient MX), found
    by `iters` retraction steps from the central point:

        μ ← μ ⊞ Σ wᵢ (xᵢ ⊟ μ)

    A plain ambient average is wrong on SO(3); this converges the tangent
    residual to zero. `iters=1` suffices when the spread is small (the
    scaled-UT default); raise it for a wide spread on a curved manifold."""
    mu = points[0]
    for _ in range(max(1, iters)):
        e = w_m[0] * spec.boxminus_sym(points[0], mu)
        for i in range(1, len(points)):
            e = e + w_m[i] * spec.boxminus_sym(points[i], mu)
        mu = spec.boxplus_sym(mu, e)
    return mu


def ut_predict(propagated: list[ca.MX], Q: ca.MX,
               w_m: list[float], w_c: list[float], spec: StateSpec,
               mean_iters: int) -> tuple[ca.MX, ca.MX]:
    """Unscented predict from already-propagated sigma points:

        x⁺ = mean({f(x ⊞ δᵢ)})                 (manifold mean)
        P⁺ = Σ w_cᵢ (yᵢ ⊟ x⁺)(yᵢ ⊟ x⁺)ᵀ + Q   (re-symmetrized)

    `propagated[i] = f(x ⊞ δᵢ)` are the ambient next-state sigma points —
    unlike `ut_update`, the prior tangent offsets are not needed (the
    predicted covariance is re-summarized around the manifold mean)."""
    x_pred = manifold_mean(propagated, w_m, spec, mean_iters)
    d = [spec.boxminus_sym(y, x_pred) for y in propagated]
    P_pred = w_c[0] * (d[0] @ d[0].T)
    for i in range(1, len(d)):
        P_pred = P_pred + w_c[i] * (d[i] @ d[i].T)
    return x_pred, symmetrize(P_pred + Q)


def ut_update(x: ca.MX, P: ca.MX, deltas: list[ca.MX],
              measured: list[ca.MX], R: ca.MX, z: ca.MX,
              w_m: list[float], w_c: list[float], spec: StateSpec
              ) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
    """One unscented measurement update (measurement space is Euclidean, so
    its mean is a plain weighted sum):

        ẑ  = Σ w_mᵢ zᵢ
        S  = Σ w_cᵢ (zᵢ − ẑ)(zᵢ − ẑ)ᵀ + R          (innovation cov)
        C  = Σ w_cᵢ δᵢ (zᵢ − ẑ)ᵀ                    (cross cov)
        K  = C S⁻¹                                   (ldl solve — S is SPD)
        x⁺ = x ⊞ K (z − ẑ)                           (manifold boxplus)
        P⁺ = Σ w_cᵢ (δᵢ − K dzᵢ)(δᵢ − K dzᵢ)ᵀ + K R Kᵀ

    The covariance uses the sigma-point Joseph form — the updated tangent
    deviations `δᵢ − K dzᵢ` re-summarized, plus the gain-shaped measurement
    noise. It is algebraically identical to the textbook `P − K S Kᵀ`
    (since `Σ w_c δδᵀ = P` exactly and `C = K S`), but as a weighted sum of
    outer products it stays PSD under roundoff whenever every `w_c ≥ 0` —
    the subtractive form does not, and a UKF covariance that goes
    indefinite NaNs at the next sigma factorization.

    `deltas` are the prior tangent sigma offsets (so `xᵢ ⊟ x = δᵢ` exactly);
    `measured[i] = h(x ⊞ δᵢ)`. Returns `(x_new, P_new, nu, S)` to match
    `joseph_update`'s shape."""
    z_pred = w_m[0] * measured[0]
    for i in range(1, len(measured)):
        z_pred = z_pred + w_m[i] * measured[i]
    dz = [zi - z_pred for zi in measured]
    S = w_c[0] * (dz[0] @ dz[0].T)
    C = w_c[0] * (deltas[0] @ dz[0].T)
    for i in range(1, len(dz)):
        S = S + w_c[i] * (dz[i] @ dz[i].T)
        C = C + w_c[i] * (deltas[i] @ dz[i].T)
    S = S + R
    K = spd_solve(S, C.T).T                     # C S⁻¹ (S SPD)
    nu = z - z_pred
    x_new = spec.boxplus_sym(x, K @ nu)
    d_upd = [deltas[i] - K @ dz[i] for i in range(len(dz))]
    P_new = w_c[0] * (d_upd[0] @ d_upd[0].T)
    for i in range(1, len(d_upd)):
        P_new = P_new + w_c[i] * (d_upd[i] @ d_upd[i].T)
    P_new = symmetrize(P_new + K @ R @ K.T)
    return x_new, P_new, nu, S
