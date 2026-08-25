"""Small-matrix SPD linear algebra, unrolled to scalar MX ops.

CasADi's Linsol-node solves are the wrong tool for manta's small, fixed
SPD systems (the 3×3 inertia solve, an EKF innovation S):

  * the default (pivoting) plugins don't SX-expand → no JAX lowering;
  * the "symbolicqr" plugin expands but refuses C code generation → no
    C++ deploy.

A hand-unrolled Cholesky produces plain scalar arithmetic nodes that
evaluate, differentiate, C-codegen, AND SX-expand. Only for matrices
that are SPD *by construction* (a guarded inertia tensor, S = HPHᵀ + R
with R > 0); a genuinely-singular system (the joint-space mass matrix
can be) must stay on a runtime-pivoting Linsol.
"""

from __future__ import annotations

import casadi as ca


def _square_dim(A: ca.MX, *, op: str) -> int:
    """`A.shape[0]`, validated square — defaulting `n` from a non-square
    `A` would silently factor a truncated block."""
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"{op}: A must be square, got shape {tuple(A.shape)}")
    return A.shape[0]


def chol_lower(A: ca.MX, n: int | None = None, *,
               floor: float = 0.0) -> list[list[ca.MX]]:
    """Lower-triangular Cholesky factor of an n×n SPD `A`, as a list-of-
    lists of scalar MX (row-major, j ≤ i). `n` defaults to `A.shape[0]`.

    `floor > 0` clamps each diagonal pivot to at least `floor` before the
    sqrt (units: those of `A`'s diagonal). For a comfortably-PD `A` the
    clamp never engages and the factor is exact; for an `A` that roundoff
    has pushed indefinite (a covariance after many folds) it degrades to a
    regularized factor instead of poisoning the graph with NaN — a negative
    pivot would otherwise `sqrt` to NaN and every later column divides by
    it. Callers whose `A` is SPD *by construction* (inertia solve,
    S = HPHᵀ + R) keep the exact default; callers factoring an *iterated*
    covariance (the UKF's sigma spread) should pass their jitter."""
    if n is None:
        n = _square_dim(A, op="chol_lower")
    L = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = A[i, j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            if i == j:
                L[i][j] = ca.sqrt(ca.fmax(acc, floor) if floor > 0.0
                                  else acc)
            else:
                L[i][j] = acc / L[j][j]
    return L


def _chol_column_solve(L: list, b_col, n: int) -> ca.MX:
    """Solve L Lᵀ x = b for one column via forward/back substitution."""
    y = [None] * n
    for i in range(n):
        acc = b_col[i]
        for k in range(i):
            acc = acc - L[i][k] * y[k]
        y[i] = acc / L[i][i]
    x = [None] * n
    for i in reversed(range(n)):
        acc = y[i]
        for k in range(i + 1, n):
            acc = acc - L[k][i] * x[k]
        x[i] = acc / L[i][i]
    return ca.vertcat(*x)


def spd_solve(A: ca.MX, B: ca.MX) -> ca.MX:
    """`A⁻¹ B` for a small SPD `A` (n×n) and any-width `B` (n×m),
    unrolled Cholesky — see module docstring for why not `ca.solve`."""
    n = _square_dim(A, op="spd_solve")
    L = chol_lower(A, n)
    cols = [_chol_column_solve(L, B[:, j], n) for j in range(B.shape[1])]
    return ca.horzcat(*cols)


def spd_logdet(A: ca.MX, n: int | None = None) -> ca.MX:
    """`log det A` for a small SPD `A`: 2·Σ log diag(chol(A)). `n` defaults
    to `A.shape[0]`."""
    if n is None:
        n = _square_dim(A, op="spd_logdet")
    L = chol_lower(A, n)
    return 2.0 * sum(ca.log(L[i][i]) for i in range(n))
