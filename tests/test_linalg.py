"""Unrolled SPD linear algebra (ir/_linalg): numerics + shape validation."""

import casadi as ca
import numpy as np
import pytest

from manta.ir._linalg import chol_lower, spd_logdet, spd_solve


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def test_spd_solve_matches_numpy():
    A = _spd(4)
    B = np.random.default_rng(1).standard_normal((4, 2))
    X = np.asarray(ca.evalf(spd_solve(ca.MX(ca.DM(A)), ca.MX(ca.DM(B)))))
    np.testing.assert_allclose(X, np.linalg.solve(A, B), atol=1e-10)


def test_spd_logdet_matches_numpy():
    A = _spd(5)
    val = float(ca.evalf(spd_logdet(ca.MX(ca.DM(A)))))
    assert val == pytest.approx(np.linalg.slogdet(A)[1], abs=1e-10)


def test_default_n_rejects_nonsquare():
    A = ca.MX.sym("A", 3, 2)
    with pytest.raises(ValueError, match="square"):
        chol_lower(A)
    with pytest.raises(ValueError, match="square"):
        spd_logdet(A)
