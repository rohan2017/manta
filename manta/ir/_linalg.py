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


def chol_lower(A: ca.MX, n: int) -> list:
    """Lower-triangular Cholesky factor of an n×n SPD `A`, as a list-of-
    lists of scalar MX (row-major, j ≤ i)."""
    L = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = A[i, j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            L[i][j] = ca.sqrt(acc) if i == j else acc / L[j][j]
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
    n = A.shape[0]
    L = chol_lower(A, n)
    cols = [_chol_column_solve(L, B[:, j], n) for j in range(B.shape[1])]
    return ca.horzcat(*cols)


def spd_logdet(A: ca.MX, n: int) -> ca.MX:
    """`log det A` for a small SPD `A`: 2·Σ log diag(chol(A))."""
    L = chol_lower(A, n)
    return 2.0 * sum(ca.log(L[i][i]) for i in range(n))
