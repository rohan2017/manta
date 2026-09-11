from __future__ import annotations

import numpy as np
import pytest

from manta.fit import NnlsConvergenceError, nnls_from_normal_equations


def test_nnls_from_normal_equations_satisfies_boundary_solution() -> None:
    design = np.asarray(((1.0, 1.0), (1.0, -1.0), (1.0, 0.0)))
    target = np.asarray((1.0, -2.0, -0.5))

    solution = nnls_from_normal_equations(design.T @ design, design.T @ target)

    np.testing.assert_allclose(solution, (0.0, 1.5), atol=1e-8)
    gradient = design.T @ design @ solution - design.T @ target
    assert gradient[0] >= -1e-8


def test_nnls_refuses_unconverged_iteration_budget() -> None:
    gram = np.asarray(((2.0, 1.0), (1.0, 2.0)))
    correlation = np.asarray((-1.0, 1.0))

    with pytest.raises(NnlsConvergenceError) as caught:
        nnls_from_normal_equations(gram, correlation, max_iterations=1)

    assert caught.value.iterations == 1
    assert caught.value.active_set == (1,)
