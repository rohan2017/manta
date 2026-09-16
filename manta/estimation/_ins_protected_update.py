"""Research correction with explicitly selected active error coordinates."""

import casadi as ca

from ..ir._linalg import spd_solve
from ._kalman import _reset_jacobian, symmetrize


def protected_update(x, P, h, H, R, z, spec, *, active):
    """Zero inactive gain rows, retaining Joseph covariance and every cross term.

    Inactive coordinates remain uncertain and enter S. This is a consider
    update, not an update that treats navigation as known. A finite chart must
    additionally preserve the intended physical states under active retraction.
    """
    n = spec.tangent_dim
    nu, S = z-h, symmetrize(H @ P @ H.T+R)
    mask = ca.DM.zeros(n, 1)
    for axis in active:
        if not 0 <= axis < n:
            raise ValueError("active coordinate outside the state")
        mask[axis] = 1
    gain = ca.diag(mask) @ spd_solve(S, (P @ H.T).T).T
    delta = gain @ nu
    mean = spec.boxplus_sym(x, delta)
    A = ca.MX.eye(n)-gain @ H
    posterior = A @ P @ A.T+gain @ R @ gain.T
    reset = _reset_jacobian(spec, delta, x)
    return mean, symmetrize(reset @ posterior @ reset.T), nu, S
