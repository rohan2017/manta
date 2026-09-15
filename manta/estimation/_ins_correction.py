"""Experimental correction kernels for isolated INS research.

These kernels are not selected by any production INS default. The predicted
Gaussian prior is held fixed during one measurement's iterations. They do not
restore higher moments discarded when a physical prior is Gaussianized in a
finite error chart. Callers must retain every active nuisance coordinate;
Schmidt nuisance means/cross-covariances require a separate implementation.
"""

from __future__ import annotations

import casadi as ca

from ..ir._linalg import spd_solve
from ._kalman import _reset_jacobian, symmetrize


def damped_iterated_update(x, P, R, z, spec, evaluate, *, iterations=3,
                           line_search_steps=4):
    """Bounded Gauss–Newton MAP correction in the original prior error chart.

    ``evaluate(delta)`` returns h(x ⊞ delta) and its Jacobian with respect to
    that *original* delta, not the tangent at the shifted state. R is frozen at
    the predicted state. Backtracking retains the lowest exact MAP objective
    among the bounded candidates and the previous iterate. It is not a global
    optimizer, and the Laplace covariance is not an exact posterior covariance.

    Work in dual coordinates delta = P a: the prior cost is aᵀPa, so an exact
    zero-variance direction needs neither an inverse of P nor a variance floor.
    This is equivalent to the Gaussian prior's pseudoinverse on its support.
    Innovation/NIS diagnostics describe the original prior linearization.
    """
    if iterations < 1 or line_search_steps < 1:
        raise ValueError("iteration and line-search bounds must be positive")
    n = P.size1()
    zero = ca.MX.zeros(n)
    h0, H0 = evaluate(zero)
    nu = z-h0
    S0 = H0@P@H0.T+R

    def cost(dual):
        h, _ = evaluate(P@dual)
        residual = z-h
        return ca.dot(dual, P@dual)+ca.dot(residual, spd_solve(R, residual))

    dual = zero
    for _ in range(iterations):
        delta = P@dual
        h, H = evaluate(delta)
        proposal = H.T@spd_solve(H@P@H.T+R, z-h+H@delta)
        best, best_cost = dual, cost(dual)
        for step in range(line_search_steps):
            candidate = dual+(proposal-dual)*(0.5**step)
            candidate_cost = cost(candidate)
            improved = candidate_cost < best_cost
            best = ca.if_else(improved, candidate, best)
            best_cost = ca.if_else(improved, candidate_cost, best_cost)
        dual = best
    delta = P@dual
    _, H = evaluate(delta)
    K = spd_solve(H@P@H.T+R, H@P).T
    residual_map = ca.MX.eye(n)-K@H
    posterior = symmetrize(residual_map@P@residual_map.T+K@R@K.T)
    reset = _reset_jacobian(spec, delta, x)
    return (spec.boxplus_sym(x, delta), symmetrize(reset@posterior@reset.T),
            nu, S0)


def conditional_statistics(mean, covariance, projection, evaluate):
    """Conditional positive-weight cubature, analytic in remaining directions.

    The nonlinear subspace is ``projection @ error``. For fixed projected
    error the measurement must be affine in the remaining error; this is a
    caller-owned structural contract, not an automatic observability claim.
    Cross-covariance with *all* state coordinates is retained. Use k projected
    coordinates for 2k+1 evaluations instead of 2n+1 full-state evaluations.

    The center has mean weight zero and covariance weight two (alpha=1,
    beta=2, kappa=0). Conditional covariance uses mean weights, so linear
    directions are not counted again by the center's covariance weight.
    """
    from ._ins_moments import psd_root

    n, k = covariance.size1(), projection.size1()
    if k < 1 or k > n or projection.size2() != n:
        raise ValueError("conditional projection must have between 1 and n rows")
    projected = projection@covariance@projection.T
    root = psd_root(projected)
    cross = covariance@projection.T
    conditional = ca.MX.zeros(n, k)
    for j in range(k):
        residual = cross[:, j]-sum((conditional[:, i]*root[j, i] for i in range(j)), ca.MX.zeros(n))
        conditional[:, j] = ca.if_else(root[j, j] > 0, residual/root[j, j], 0)
    remaining = symmetrize(covariance-conditional@conditional.T)
    deltas = [ca.MX.zeros(n)] + [sign*k**.5*conditional[:, j]
                                for sign in (1., -1.) for j in range(k)]
    wm = [0.] + [1/(2*k)]*(2*k)
    wc = [2.] + [1/(2*k)]*(2*k)
    values = [evaluate(mean+d) for d in deltas]
    h_mean = sum((w*value[0] for w, value in zip(wm, values)), ca.MX.zeros(values[0][0].size1()))
    measurement_dim = h_mean.size1()
    cross_h = ca.MX.zeros(n, measurement_dim)
    covariance_h = ca.MX.zeros(measurement_dim, measurement_dim)
    for d, (h, H), a, b in zip(deltas, values, wm, wc):
        innovation = h-h_mean
        cross_h += b*d@innovation.T+a*remaining@H.T
        covariance_h += b*innovation@innovation.T+a*H@remaining@H.T
    return h_mean, cross_h, symmetrize(covariance_h), (deltas, values, wm, wc, remaining)


def _projection(n, axes, projection):
    if (axes is None) == (projection is None):
        raise ValueError("supply either conditional axes or a linear projection")
    if axes is not None:
        if len(set(axes)) != len(axes) or any(i < 0 or i >= n for i in axes):
            raise ValueError("conditional axes must be unique valid state coordinates")
        return ca.DM.eye(n)[list(axes), :]
    if projection.size2() != n or not 1 <= projection.size1() <= n:
        raise ValueError("invalid conditional linear projection")
    return projection


def conditional_update(x, P, R, z, spec, evaluate, *, axes=None, projection=None):
    """Conditional statistical update; full joint covariance and exact reset."""
    n = P.size1()
    projection = _projection(n, axes, projection)
    h, cross, variance, details = conditional_statistics(ca.MX.zeros(n), P, projection, evaluate)
    nu, S = z-h, variance+R
    K = spd_solve(S, cross.T).T
    delta = K@nu
    deltas, values, wm, wc, remaining = details
    posterior = K@R@K.T
    for d, (hi, Hi), a, b in zip(deltas, values, wm, wc):
        residual = d-K@(hi-h)
        transition = ca.MX.eye(n)-K@Hi
        posterior += b*residual@residual.T+a*transition@remaining@transition.T
    reset = _reset_jacobian(spec, delta, x)
    return spec.boxplus_sym(x, delta), symmetrize(reset@posterior@reset.T), nu, S


def posterior_linearized_update(x, P, R, z, spec, evaluate, *, axes=None, projection=None,
                                iterations=3, damping=.5):
    """Damped iterated posterior statistical linearization in a fixed chart.

    A standard-normal latent prior supports exactly singular physical P.
    Only the linearization distribution changes; the conditioning prior stays
    N(0,I), preventing repeated counting of one observation. The first fold is
    the conditional statistical update. Later Gaussian proposals are damped
    by moment matching a mixture with the previous proposal. This bounded
    heuristic is not guaranteed to find the best Gaussian or a global optimum.
    """
    from ._ins_moments import psd_root

    if iterations < 1 or not 0 < damping <= 1:
        raise ValueError("invalid posterior iteration settings")
    n = P.size1()
    projection = _projection(n, axes, projection)
    root = psd_root(P)
    projection = projection@root
    mean, covariance = ca.MX.zeros(n), ca.MX.eye(n)

    def latent_evaluate(value):
        h, H = evaluate(root@value)
        return h, H@root

    nu0 = S0 = None
    for iteration in range(iterations):
        h, cross, variance, details = conditional_statistics(mean, covariance, projection, latent_evaluate)
        if iteration == 0:
            nu0, S0 = z-h, variance+R
        A = spd_solve(covariance, cross).T
        b = h-A@mean
        deltas, values, wm, wc, remaining = details
        residual_cov = ca.MX.zeros(z.numel(), z.numel())
        for d, (hi, Hi), a, c in zip(deltas, values, wm, wc):
            residual = hi-h-A@d
            residual_H = Hi-A
            residual_cov += c*residual@residual.T+a*residual_H@remaining@residual_H.T
        observation_cov = symmetrize(residual_cov+R)
        K = spd_solve(A@A.T+observation_cov, A).T
        proposal_mean = K@(z-b)
        transition = ca.MX.eye(n)-K@A
        proposal_cov = symmetrize(transition@transition.T+K@observation_cov@K.T)
        alpha = 1. if iteration == 0 else damping
        difference = proposal_mean-mean
        covariance = symmetrize((1-alpha)*covariance+alpha*proposal_cov
                                 +alpha*(1-alpha)*difference@difference.T)
        mean += alpha*difference
    delta = root@mean
    reset = _reset_jacobian(spec, delta, x)
    return (spec.boxplus_sym(x, delta), symmetrize(reset@root@covariance@root.T@reset.T),
            nu0, S0)
