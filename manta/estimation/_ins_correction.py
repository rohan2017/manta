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
