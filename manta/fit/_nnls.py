"""Small deterministic non-negative least squares from normal equations."""

from __future__ import annotations

import numpy as np


class NnlsConvergenceError(RuntimeError):
    """The active-set NNLS iteration could not establish a KKT solution."""

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        active_set: tuple[int, ...],
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.active_set = active_set


def nnls_from_normal_equations(
    gram: np.ndarray,
    correlation: np.ndarray,
    *,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Solve a small NNLS problem from ``A.T @ A`` and ``A.T @ b``.

    The deterministic active-set method is intended for problems with tens of
    variables where retaining the full design matrix is undesirable. It
    returns only after checking primal and dual feasibility. Cycling, an
    exhausted iteration budget, or a failed passive-set solve raises
    :class:`NnlsConvergenceError`; no unconverged iterate is presented as a
    solution.
    """
    matrix = np.asarray(gram, dtype=float)
    vector = np.asarray(correlation, dtype=float).reshape(-1)
    count = len(vector)
    if matrix.shape != (count, count):
        raise ValueError("NNLS Gram matrix has incompatible shape")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("NNLS normal equations must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ValueError("NNLS Gram matrix must be symmetric")
    if max_iterations is None:
        max_iterations = max(20, 10 * count)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if count == 0:
        return np.zeros(0)

    diagonal_scale = max(float(np.max(np.diag(matrix))), np.finfo(float).tiny)
    working = matrix + np.eye(count) * (1e-9 * diagonal_scale)
    active = np.ones(count, dtype=bool)
    seen: set[tuple[int, ...]] = set()
    scale = max(float(np.max(np.abs(vector))), np.finfo(float).tiny)
    tolerance = 1e-8 * scale

    for iteration in range(1, max_iterations + 1):
        key = tuple(int(index) for index in np.flatnonzero(active))
        if key in seen:
            raise NnlsConvergenceError(
                "NNLS active set cycled before satisfying KKT conditions",
                iterations=iteration - 1,
                active_set=key,
            )
        seen.add(key)
        indices = np.flatnonzero(active)
        solution = np.zeros(count)
        if len(indices):
            try:
                solution[indices] = np.linalg.solve(
                    working[np.ix_(indices, indices)], vector[indices]
                )
            except np.linalg.LinAlgError as exc:
                raise NnlsConvergenceError(
                    "NNLS passive-set solve failed",
                    iterations=iteration,
                    active_set=key,
                ) from exc
        negative = indices[solution[indices] < -tolerance]
        if len(negative):
            active[int(negative[np.argmin(solution[negative])])] = False
            continue
        solution = np.maximum(solution, 0.0)
        gradient = working @ solution - vector
        inactive = np.flatnonzero(~active)
        violating = inactive[gradient[inactive] < -tolerance]
        if not len(violating):
            return solution
        active[int(violating[np.argmin(gradient[violating])])] = True

    raise NnlsConvergenceError(
        f"NNLS did not converge in {max_iterations} iterations",
        iterations=max_iterations,
        active_set=tuple(int(index) for index in np.flatnonzero(active)),
    )
