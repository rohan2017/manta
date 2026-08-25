"""Persistent native OSQP workspace over CasADi's bundled OSQP library.

CasADi remains Manta's model compiler, but the RTI hot path should not rebuild
``DM`` sparse matrices or pass through the generic conic ABI every tick.  This
module compiles a tiny, cached C bridge against the OSQP headers and shared
library distributed with CasADi.  Matrix sparsity and the factorization
workspace are created once; subsequent calls update numeric values in place.
"""

from __future__ import annotations

import contextlib
import ctypes
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import numpy.typing as npt

from ..codegen.numpy._compile import CompilationError, build_native_library

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_SOURCE = r"""
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "osqp.h"

#if defined(_WIN32)
#define MANTA_EXPORT __declspec(dllexport)
#else
#define MANTA_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    OSQPWorkspace *work;
    OSQPData data;
    OSQPSettings settings;
    csc P;
    csc A;
    c_int *Pp;
    c_int *Pi;
    c_int *Ap;
    c_int *Ai;
    c_float *Px;
    c_float *Ax;
    c_float *q;
    c_float *l;
    c_float *u;
} MantaOSQP;

static double manta_seconds(const struct timespec *start,
                            const struct timespec *finish) {
    return (double)(finish->tv_sec - start->tv_sec)
        + 1e-9 * (double)(finish->tv_nsec - start->tv_nsec);
}

static void manta_free(MantaOSQP *solver) {
    if (!solver) return;
    if (solver->work) osqp_cleanup(solver->work);
    free(solver->Pp); free(solver->Pi); free(solver->Ap); free(solver->Ai);
    free(solver->Px); free(solver->Ax); free(solver->q);
    free(solver->l); free(solver->u);
    free(solver);
}

MANTA_EXPORT void *manta_osqp_create(
    c_int n, c_int m,
    c_int p_nnz, const c_int *Pp, const c_int *Pi,
    c_int a_nnz, const c_int *Ap, const c_int *Ai,
    c_float eps_abs, c_float eps_rel, c_int max_iter,
    c_int check_termination, c_float rho, c_float alpha,
    c_int adaptive_rho, c_int adaptive_rho_interval, c_int scaling) {
    MantaOSQP *s = (MantaOSQP *)calloc(1, sizeof(MantaOSQP));
    if (!s) return NULL;
    s->Pp = (c_int *)malloc((n + 1) * sizeof(c_int));
    s->Pi = (c_int *)malloc(p_nnz * sizeof(c_int));
    s->Ap = (c_int *)malloc((n + 1) * sizeof(c_int));
    s->Ai = (c_int *)malloc(a_nnz * sizeof(c_int));
    s->Px = (c_float *)calloc(p_nnz, sizeof(c_float));
    s->Ax = (c_float *)calloc(a_nnz, sizeof(c_float));
    s->q = (c_float *)calloc(n, sizeof(c_float));
    s->l = (c_float *)malloc(m * sizeof(c_float));
    s->u = (c_float *)malloc(m * sizeof(c_float));
    if (!s->Pp || !s->Pi || !s->Ap || !s->Ai || !s->Px || !s->Ax ||
        !s->q || !s->l || !s->u) {
        manta_free(s); return NULL;
    }
    memcpy(s->Pp, Pp, (n + 1) * sizeof(c_int));
    memcpy(s->Pi, Pi, p_nnz * sizeof(c_int));
    memcpy(s->Ap, Ap, (n + 1) * sizeof(c_int));
    memcpy(s->Ai, Ai, a_nnz * sizeof(c_int));
    for (c_int j = 0; j < n; ++j) {
        for (c_int p = s->Pp[j]; p < s->Pp[j + 1]; ++p) {
            if (s->Pi[p] == j) s->Px[p] = 1e-9;
        }
    }
    for (c_int j = 0; j < n; ++j) {
        for (c_int p = s->Ap[j]; p < s->Ap[j + 1]; ++p) {
            if (s->Ai[p] == m - n + j) s->Ax[p] = 1.0;
        }
    }
    for (c_int i = 0; i < m; ++i) {
        s->l[i] = -OSQP_INFTY; s->u[i] = OSQP_INFTY;
    }
    s->P.nzmax = p_nnz; s->P.m = n; s->P.n = n;
    s->P.p = s->Pp; s->P.i = s->Pi; s->P.x = s->Px; s->P.nz = -1;
    s->A.nzmax = a_nnz; s->A.m = m; s->A.n = n;
    s->A.p = s->Ap; s->A.i = s->Ai; s->A.x = s->Ax; s->A.nz = -1;
    s->data.n = n; s->data.m = m; s->data.P = &s->P; s->data.A = &s->A;
    s->data.q = s->q; s->data.l = s->l; s->data.u = s->u;
    osqp_set_default_settings(&s->settings);
    s->settings.verbose = 0;
    s->settings.eps_abs = eps_abs; s->settings.eps_rel = eps_rel;
    s->settings.max_iter = max_iter;
    s->settings.check_termination = check_termination;
    s->settings.rho = rho; s->settings.alpha = alpha;
    s->settings.adaptive_rho = adaptive_rho;
    s->settings.adaptive_rho_interval = adaptive_rho_interval;
    s->settings.scaling = scaling;
    s->settings.polish = 0; s->settings.warm_start = 1;
    if (osqp_setup(&s->work, &s->data, &s->settings) != 0) {
        manta_free(s); return NULL;
    }
    return s;
}

MANTA_EXPORT void manta_osqp_destroy(void *handle) {
    manta_free((MantaOSQP *)handle);
}

MANTA_EXPORT c_int manta_osqp_solve(
    void *handle, const c_float *Px, const c_float *Ax,
    const c_float *q, const c_float *l, const c_float *u,
    const c_float *x0, const c_float *y0,
    double *update_seconds, double *solve_seconds) {
    MantaOSQP *s = (MantaOSQP *)handle;
    if (!s || !s->work) return -100;
    struct timespec update_start, solve_start, finish;
    clock_gettime(CLOCK_MONOTONIC, &update_start);
    c_int flag = osqp_update_P_A(s->work, Px, NULL, s->P.nzmax,
                                 Ax, NULL, s->A.nzmax);
    if (flag) return -200 - flag;
    if ((flag = osqp_update_lin_cost(s->work, q))) return -300 - flag;
    if ((flag = osqp_update_bounds(s->work, l, u))) return -400 - flag;
    if ((flag = osqp_warm_start(s->work, x0, y0))) return -500 - flag;
    clock_gettime(CLOCK_MONOTONIC, &solve_start);
    *update_seconds = manta_seconds(&update_start, &solve_start);
    if ((flag = osqp_solve(s->work))) return -600 - flag;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    *solve_seconds = manta_seconds(&solve_start, &finish);
    return s->work->info->status_val;
}

MANTA_EXPORT void manta_osqp_result(
    void *handle, c_float *x, c_float *y, c_float *cost,
    c_int *iterations, c_float *primal_residual, c_float *dual_residual,
    c_int *rho_updates, c_float *rho_estimate,
    char *status, c_int status_size) {
    MantaOSQP *s = (MantaOSQP *)handle;
    memcpy(x, s->work->solution->x, s->data.n * sizeof(c_float));
    memcpy(y, s->work->solution->y, s->data.m * sizeof(c_float));
    *cost = s->work->info->obj_val;
    *iterations = s->work->info->iter;
    *primal_residual = s->work->info->pri_res;
    *dual_residual = s->work->info->dua_res;
    *rho_updates = s->work->info->rho_updates;
    *rho_estimate = s->work->info->rho_estimate;
    if (status_size > 0) {
        strncpy(status, s->work->info->status, status_size - 1);
        status[status_size - 1] = '\0';
    }
}
"""

_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_LOCK = threading.Lock()


def _library() -> ctypes.CDLL:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        casadi_dir = Path(ca.__file__).resolve().parent
        include_dir = casadi_dir / "include" / "osqp"
        osqp_library = casadi_dir / "libosqp.so"
        if not include_dir.is_dir() or not osqp_library.is_file():
            raise CompilationError(
                "CasADi installation does not contain native OSQP headers "
                "and libosqp.so")
        # The bridge is compiled against CasADi's bundled OSQP build
        # configuration; a different `osqp_configure.h` is a different ABI.
        configure = (include_dir / "osqp_configure.h").read_bytes()
        path = build_native_library(
            _SOURCE, stem="manta_osqp", what="native OSQP",
            compiler_flags=("-O3",),
            link_args=(f"-I{include_dir}", f"-L{casadi_dir}", "-losqp",
                       f"-Wl,-rpath,{casadi_dir}"),
            identity_salt=configure,
            timeout_s=30.0,
        ).path
        library = ctypes.CDLL(str(path))
        index_pointer = np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
        float_pointer = np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        library.manta_osqp_create.argtypes = [
            ctypes.c_longlong, ctypes.c_longlong,
            ctypes.c_longlong, index_pointer, index_pointer,
            ctypes.c_longlong, index_pointer, index_pointer,
            ctypes.c_double, ctypes.c_double, ctypes.c_longlong,
            ctypes.c_longlong, ctypes.c_double, ctypes.c_double,
            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ]
        library.manta_osqp_create.restype = ctypes.c_void_p
        library.manta_osqp_destroy.argtypes = [ctypes.c_void_p]
        library.manta_osqp_solve.argtypes = [
            ctypes.c_void_p, float_pointer, float_pointer, float_pointer,
            float_pointer, float_pointer, float_pointer, float_pointer,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        library.manta_osqp_solve.restype = ctypes.c_longlong
        library.manta_osqp_result.argtypes = [
            ctypes.c_void_p, float_pointer, float_pointer,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_longlong,
        ]
        _LIBRARY = library
        return library


@dataclass(frozen=True)
class OSQPResult:
    x: FloatArray
    y: FloatArray
    cost: float
    iterations: int
    update_ms: float
    iteration_ms: float
    primal_residual: float
    dual_residual: float
    rho_updates: int
    rho_estimate: float
    status: str
    success: bool


class NativeOSQP:
    """One fixed-sparsity OSQP workspace with numeric in-place updates."""

    def __init__(
        self, n: int, m: int, Pp: Any, Pi: Any, Ap: Any, Ai: Any, *,
        eps_abs: float = 2e-3, eps_rel: float = 2e-3,
        max_iter: int = 800,
        check_termination: int = 5,
        rho: float = 0.1,
        alpha: float = 1.6,
        adaptive_rho: bool = True,
        adaptive_rho_interval: int = 0,
        scaling: int = 10,
    ) -> None:
        if eps_abs < 0.0 or eps_rel < 0.0 or max_iter < 1:
            raise ValueError("invalid OSQP tolerances or iteration limit")
        if check_termination < 1:
            raise ValueError("OSQP check_termination must be positive")
        if rho <= 0.0 or not 0.0 < alpha < 2.0:
            raise ValueError("OSQP requires rho > 0 and 0 < alpha < 2")
        if adaptive_rho_interval < 0 or scaling < 0:
            raise ValueError(
                "OSQP adaptive_rho_interval and scaling must be non-negative")
        self.n, self.m = int(n), int(m)
        self.Pp = np.ascontiguousarray(Pp, dtype=np.int64)
        self.Pi = np.ascontiguousarray(Pi, dtype=np.int64)
        self.Ap = np.ascontiguousarray(Ap, dtype=np.int64)
        self.Ai = np.ascontiguousarray(Ai, dtype=np.int64)
        self._library = _library()
        self._handle = self._library.manta_osqp_create(
            self.n, self.m, len(self.Pi), self.Pp, self.Pi,
            len(self.Ai), self.Ap, self.Ai,
            float(eps_abs), float(eps_rel), int(max_iter),
            int(check_termination), float(rho), float(alpha),
            int(adaptive_rho), int(adaptive_rho_interval), int(scaling))
        if not self._handle:
            raise RuntimeError("native OSQP workspace setup failed")

    def close(self) -> None:
        if self._handle:
            self._library.manta_osqp_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # __del__ must never raise
            self.close()

    @staticmethod
    def _float(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
        result = np.ascontiguousarray(value, dtype=np.float64)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError(
                f"{name} must be finite with shape {shape}")
        return result

    def solve(
        self, Px: Any, Ax: Any, q: Any, lower: Any, upper: Any,
        x0: Any, y0: Any,
    ) -> OSQPResult:
        Px = self._float(Px, (len(self.Pi),), "OSQP P values")
        Ax = self._float(Ax, (len(self.Ai),), "OSQP A values")
        q = self._float(q, (self.n,), "OSQP linear cost")
        lower = self._float(lower, (self.m,), "OSQP lower bounds")
        upper = self._float(upper, (self.m,), "OSQP upper bounds")
        x0 = self._float(x0, (self.n,), "OSQP primal warm start")
        y0 = self._float(y0, (self.m,), "OSQP dual warm start")
        update_seconds, solve_seconds = ctypes.c_double(), ctypes.c_double()
        status_value = int(self._library.manta_osqp_solve(
            self._handle, Px, Ax, q, lower, upper, x0, y0,
            ctypes.byref(update_seconds), ctypes.byref(solve_seconds)))
        if status_value < 0:
            raise RuntimeError(
                f"native OSQP numeric update failed with {status_value}")
        x, y = np.empty(self.n), np.empty(self.m)
        cost, iterations = ctypes.c_double(), ctypes.c_longlong()
        primal_residual, dual_residual = ctypes.c_double(), ctypes.c_double()
        rho_updates, rho_estimate = ctypes.c_longlong(), ctypes.c_double()
        status = ctypes.create_string_buffer(32)
        self._library.manta_osqp_result(
            self._handle, x, y, ctypes.byref(cost), ctypes.byref(iterations),
            ctypes.byref(primal_residual), ctypes.byref(dual_residual),
            ctypes.byref(rho_updates), ctypes.byref(rho_estimate),
            status, len(status))
        return OSQPResult(
            x=x, y=y, cost=float(cost.value),
            iterations=int(iterations.value),
            update_ms=1e3*float(update_seconds.value),
            iteration_ms=1e3*float(solve_seconds.value),
            primal_residual=float(primal_residual.value),
            dual_residual=float(dual_residual.value),
            rho_updates=int(rho_updates.value),
            rho_estimate=float(rho_estimate.value),
            status=status.value.decode(errors="replace"),
            success=status_value in (1, 2),
        )


__all__ = ["NativeOSQP", "OSQPResult"]
