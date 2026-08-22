"""Persistent HPIPM optimal-control QP workspace for Manta RTI.

The bridge uses the HPIPM and BLASFEO libraries bundled with CasADi, so the
optional structured backend does not add a Python or system dependency.  It
keeps Manta's direct multiple-shooting problem intact.  The previous actuator
correction is appended to the shooting state so the slew penalty remains
stage-local, and the shared bank slack is appended to each stage input.
"""

from __future__ import annotations

import ctypes
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import numpy.typing as npt

from ..codegen.numpy._compile import CompilationError, build_native_library

FloatArray = npt.NDArray[np.float64]

_SOURCE = r"""
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hpipm_common.h"
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp.h"
#include "hpipm_d_ocp_qp_sol.h"
#include "hpipm_d_ocp_qp_ipm.h"
#include "hpipm_d_part_cond.h"

#if defined(_WIN32)
#define MANTA_EXPORT __declspec(dllexport)
#else
#define MANTA_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    int N, nx, nu, nc, ng, nxa, max_nu, cond_N;
    double effort, rate, slack_weight;

    struct d_ocp_qp_dim dim;
    struct d_ocp_qp qp;
    struct d_ocp_qp_sol sol;
    struct d_ocp_qp_ipm_arg arg;
    struct d_ocp_qp_ipm_ws ws;
    void *dim_mem, *qp_mem, *sol_mem, *arg_mem, *ws_mem;

    struct d_ocp_qp_dim cond_dim;
    struct d_ocp_qp cond_qp;
    struct d_ocp_qp_sol cond_sol;
    struct d_ocp_qp_ipm_arg cond_ipm_arg;
    struct d_ocp_qp_ipm_ws cond_ipm_ws;
    struct d_part_cond_qp_arg cond_arg;
    struct d_part_cond_qp_ws cond_ws;
    void *cond_dim_mem, *cond_qp_mem, *cond_sol_mem;
    void *cond_ipm_arg_mem, *cond_ipm_ws_mem;
    void *cond_arg_mem, *cond_ws_mem;
    int *block_size;

    int *nxv, *nuv, *nbx, *nbu, *ngv, *zero_dims;
    double **A, **B, **b, **Q, **S, **R, **q, **r;
    int **idxbx, **idxbu;
    double **lbx, **ubx, **lbu, **ubu, **C, **D, **lg, **ug;
    double **lbu_mask, **ubu_mask, **lg_mask, **ug_mask;
    double **zeros;
    double *tmp_x, *tmp_u;
} MantaHPIPM;

static double seconds_between(const struct timespec *start,
                              const struct timespec *finish) {
    return (double)(finish->tv_sec - start->tv_sec)
        + 1e-9 * (double)(finish->tv_nsec - start->tv_nsec);
}

static void *checked_calloc(size_t count, size_t size) {
    if (count == 0) count = 1;
    return calloc(count, size);
}

static void manta_hpipm_free(MantaHPIPM *s) {
    if (!s) return;
    if (s->A) for (int k = 0; k <= s->N; ++k) {
        free(s->A[k]); free(s->B[k]); free(s->b[k]);
        free(s->Q[k]); free(s->S[k]); free(s->R[k]);
        free(s->q[k]); free(s->r[k]);
        free(s->idxbx[k]); free(s->idxbu[k]);
        free(s->lbx[k]); free(s->ubx[k]);
        free(s->lbu[k]); free(s->ubu[k]);
        free(s->C[k]); free(s->D[k]); free(s->lg[k]); free(s->ug[k]);
        free(s->lbu_mask[k]); free(s->ubu_mask[k]);
        free(s->lg_mask[k]); free(s->ug_mask[k]);
        free(s->zeros[k]);
    }
    free(s->A); free(s->B); free(s->b); free(s->Q); free(s->S);
    free(s->R); free(s->q); free(s->r); free(s->idxbx); free(s->idxbu);
    free(s->lbx); free(s->ubx); free(s->lbu); free(s->ubu);
    free(s->C); free(s->D); free(s->lg); free(s->ug); free(s->zeros);
    free(s->lbu_mask); free(s->ubu_mask);
    free(s->lg_mask); free(s->ug_mask);
    free(s->tmp_x); free(s->tmp_u);
    free(s->nxv); free(s->nuv); free(s->nbx); free(s->nbu);
    free(s->ngv); free(s->zero_dims); free(s->block_size);
    free(s->dim_mem); free(s->qp_mem); free(s->sol_mem);
    free(s->arg_mem); free(s->ws_mem);
    free(s->cond_dim_mem); free(s->cond_qp_mem); free(s->cond_sol_mem);
    free(s->cond_ipm_arg_mem); free(s->cond_ipm_ws_mem);
    free(s->cond_arg_mem); free(s->cond_ws_mem);
    free(s);
}

static int allocate_stage_arrays(MantaHPIPM *s) {
    int count = s->N + 1;
#define PTRS(name, type) \
    s->name = (type **)checked_calloc(count, sizeof(type *)); \
    if (!s->name) return 0
    PTRS(A, double); PTRS(B, double); PTRS(b, double);
    PTRS(Q, double); PTRS(S, double); PTRS(R, double);
    PTRS(q, double); PTRS(r, double);
    PTRS(idxbx, int); PTRS(idxbu, int);
    PTRS(lbx, double); PTRS(ubx, double);
    PTRS(lbu, double); PTRS(ubu, double);
    PTRS(C, double); PTRS(D, double); PTRS(lg, double); PTRS(ug, double);
    PTRS(lbu_mask, double); PTRS(ubu_mask, double);
    PTRS(lg_mask, double); PTRS(ug_mask, double);
    PTRS(zeros, double);
#undef PTRS
    for (int k = 0; k <= s->N; ++k) {
        int nxk = s->nxv[k], nuk = s->nuv[k], ngk = s->ngv[k];
        int nxnext = k < s->N ? s->nxv[k+1] : 0;
#define ALLOC(name, count_, type) \
        s->name[k] = (type *)checked_calloc((count_), sizeof(type)); \
        if (!s->name[k]) return 0
        ALLOC(A, nxnext*nxk, double); ALLOC(B, nxnext*nuk, double);
        ALLOC(b, nxnext, double); ALLOC(Q, nxk*nxk, double);
        ALLOC(S, nuk*nxk, double); ALLOC(R, nuk*nuk, double);
        ALLOC(q, nxk, double); ALLOC(r, nuk, double);
        ALLOC(idxbx, s->nbx[k], int); ALLOC(idxbu, s->nbu[k], int);
        ALLOC(lbx, s->nbx[k], double); ALLOC(ubx, s->nbx[k], double);
        ALLOC(lbu, s->nbu[k], double); ALLOC(ubu, s->nbu[k], double);
        ALLOC(C, ngk*nxk, double); ALLOC(D, ngk*nuk, double);
        ALLOC(lg, ngk, double); ALLOC(ug, ngk, double);
        ALLOC(lbu_mask, s->nbu[k], double);
        ALLOC(ubu_mask, s->nbu[k], double);
        ALLOC(lg_mask, ngk, double); ALLOC(ug_mask, ngk, double);
        ALLOC(zeros, 1, double);
#undef ALLOC
    }
    return 1;
}

MANTA_EXPORT void *manta_hpipm_create(
    int N, int nx, int nu, int nc, int attitude_rows, int cond_N,
    double effort, double rate, double slack_weight,
    double tolerance, int max_iter) {
    if (N < 2 || nx < 1 || nu < 1 || nc < 1 ||
        (attitude_rows != 0 && attitude_rows != 3*nc) ||
        cond_N < 0 || cond_N > N) return NULL;
    MantaHPIPM *s = (MantaHPIPM *)calloc(1, sizeof(MantaHPIPM));
    if (!s) return NULL;
    s->N=N; s->nx=nx; s->nu=nu; s->nc=nc;
    s->ng=2*nc+attitude_rows; s->nxa=nx+nu;
    s->max_nu=nu+nc; s->cond_N=cond_N;
    s->effort=effort; s->rate=rate; s->slack_weight=slack_weight;
    int count=N+1;
    s->nxv=(int *)checked_calloc(count,sizeof(int));
    s->nuv=(int *)checked_calloc(count,sizeof(int));
    s->nbx=(int *)checked_calloc(count,sizeof(int));
    s->nbu=(int *)checked_calloc(count,sizeof(int));
    s->ngv=(int *)checked_calloc(count,sizeof(int));
    s->zero_dims=(int *)checked_calloc(count,sizeof(int));
    s->tmp_x=(double *)checked_calloc(s->nxa,sizeof(double));
    s->tmp_u=(double *)checked_calloc(s->max_nu,sizeof(double));
    if (!s->nxv || !s->nuv || !s->nbx || !s->nbu || !s->ngv ||
        !s->zero_dims || !s->tmp_x || !s->tmp_u) {
        manta_hpipm_free(s); return NULL;
    }
    s->nxv[0]=0; s->nuv[0]=nu; s->nbu[0]=nu;
    for (int k=1; k<N; ++k) {
        s->nxv[k]=s->nxa; s->nuv[k]=nu+nc;
        s->nbu[k]=nu+nc; s->ngv[k]=s->ng;
    }
    s->nxv[N]=s->nxa; s->nuv[N]=nc; s->nbu[N]=nc; s->ngv[N]=s->ng;
    s->dim_mem=checked_calloc(d_ocp_qp_dim_memsize(N),1);
    if (!s->dim_mem) { manta_hpipm_free(s); return NULL; }
    d_ocp_qp_dim_create(N,&s->dim,s->dim_mem);
    d_ocp_qp_dim_set_all(s->nxv,s->nuv,s->nbx,s->nbu,s->ngv,
        s->zero_dims,s->zero_dims,s->zero_dims,&s->dim);
    if (!allocate_stage_arrays(s)) { manta_hpipm_free(s); return NULL; }
    s->qp_mem=checked_calloc(d_ocp_qp_memsize(&s->dim),1);
    s->sol_mem=checked_calloc(d_ocp_qp_sol_memsize(&s->dim),1);
    s->arg_mem=checked_calloc(d_ocp_qp_ipm_arg_memsize(&s->dim),1);
    if (!s->qp_mem || !s->sol_mem || !s->arg_mem) {
        manta_hpipm_free(s); return NULL;
    }
    d_ocp_qp_create(&s->dim,&s->qp,s->qp_mem);
    d_ocp_qp_sol_create(&s->dim,&s->sol,s->sol_mem);
    d_ocp_qp_ipm_arg_create(&s->dim,&s->arg,s->arg_mem);
    d_ocp_qp_ipm_arg_set_default(BALANCE,&s->arg);
    d_ocp_qp_ipm_arg_set_tol_stat(&tolerance,&s->arg);
    d_ocp_qp_ipm_arg_set_tol_eq(&tolerance,&s->arg);
    d_ocp_qp_ipm_arg_set_tol_ineq(&tolerance,&s->arg);
    d_ocp_qp_ipm_arg_set_tol_comp(&tolerance,&s->arg);
    d_ocp_qp_ipm_arg_set_iter_max(&max_iter,&s->arg);
    int warm=1; d_ocp_qp_ipm_arg_set_warm_start(&warm,&s->arg);
    s->ws_mem=checked_calloc(d_ocp_qp_ipm_ws_memsize(&s->dim,&s->arg),1);
    if (!s->ws_mem) { manta_hpipm_free(s); return NULL; }
    d_ocp_qp_ipm_ws_create(&s->dim,&s->arg,&s->ws,s->ws_mem);

    if (cond_N > 0 && cond_N < N) {
        s->block_size=(int *)checked_calloc(cond_N+1,sizeof(int));
        s->cond_dim_mem=checked_calloc(d_ocp_qp_dim_memsize(cond_N),1);
        s->cond_arg_mem=checked_calloc(d_part_cond_qp_arg_memsize(cond_N),1);
        if (!s->block_size || !s->cond_dim_mem || !s->cond_arg_mem) {
            manta_hpipm_free(s); return NULL;
        }
        d_part_cond_qp_compute_block_size(N,cond_N,s->block_size);
        d_ocp_qp_dim_create(cond_N,&s->cond_dim,s->cond_dim_mem);
        d_part_cond_qp_compute_dim(&s->dim,s->block_size,&s->cond_dim);
        d_part_cond_qp_arg_create(cond_N,&s->cond_arg,s->cond_arg_mem);
        d_part_cond_qp_arg_set_default(&s->cond_arg);
        d_part_cond_qp_arg_set_comp_prim_sol(1,&s->cond_arg);
        d_part_cond_qp_arg_set_comp_dual_sol_eq(1,&s->cond_arg);
        d_part_cond_qp_arg_set_comp_dual_sol_ineq(1,&s->cond_arg);
        s->cond_qp_mem=checked_calloc(d_ocp_qp_memsize(&s->cond_dim),1);
        s->cond_sol_mem=checked_calloc(d_ocp_qp_sol_memsize(&s->cond_dim),1);
        s->cond_ipm_arg_mem=checked_calloc(
            d_ocp_qp_ipm_arg_memsize(&s->cond_dim),1);
        if (!s->cond_qp_mem || !s->cond_sol_mem || !s->cond_ipm_arg_mem) {
            manta_hpipm_free(s); return NULL;
        }
        d_ocp_qp_create(&s->cond_dim,&s->cond_qp,s->cond_qp_mem);
        d_ocp_qp_sol_create(&s->cond_dim,&s->cond_sol,s->cond_sol_mem);
        d_ocp_qp_ipm_arg_create(
            &s->cond_dim,&s->cond_ipm_arg,s->cond_ipm_arg_mem);
        d_ocp_qp_ipm_arg_set_default(BALANCE,&s->cond_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_stat(&tolerance,&s->cond_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_eq(&tolerance,&s->cond_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_ineq(&tolerance,&s->cond_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_comp(&tolerance,&s->cond_ipm_arg);
        d_ocp_qp_ipm_arg_set_iter_max(&max_iter,&s->cond_ipm_arg);
        warm=0; d_ocp_qp_ipm_arg_set_warm_start(&warm,&s->cond_ipm_arg);
        s->cond_ipm_ws_mem=checked_calloc(
            d_ocp_qp_ipm_ws_memsize(&s->cond_dim,&s->cond_ipm_arg),1);
        s->cond_ws_mem=checked_calloc(d_part_cond_qp_ws_memsize(
            &s->dim,s->block_size,&s->cond_dim,&s->cond_arg),1);
        if (!s->cond_ipm_ws_mem || !s->cond_ws_mem) {
            manta_hpipm_free(s); return NULL;
        }
        d_ocp_qp_ipm_ws_create(&s->cond_dim,&s->cond_ipm_arg,
            &s->cond_ipm_ws,s->cond_ipm_ws_mem);
        d_part_cond_qp_ws_create(&s->dim,s->block_size,&s->cond_dim,
            &s->cond_arg,&s->cond_ws,s->cond_ws_mem);
    }
    return s;
}

MANTA_EXPORT void manta_hpipm_destroy(void *handle) {
    manta_hpipm_free((MantaHPIPM *)handle);
}

MANTA_EXPORT int manta_hpipm_solve(
    void *handle, const double *dyn_A, const double *dyn_B,
    const double *state_Q, const double *state_q,
    const double *control_nominal, const double *previous_control,
    const double *control_lower, const double *control_upper,
    const double *general_C, const double *general_lower,
    const double *general_upper, const double *slack_scale,
    const double *warm, double *solution, double *objective,
    int *iterations, double *res_stat, double *res_eq,
    double *res_ineq, double *res_comp,
    double *update_seconds, double *solve_seconds) {
    MantaHPIPM *s=(MantaHPIPM *)handle;
    if (!s) return -100;
    struct timespec update_start, solve_start, finish;
    clock_gettime(CLOCK_MONOTONIC,&update_start);
    const double reg=1e-9, inf=1e30;
    for (int j=0; j<=s->N; ++j) {
        int nxj=s->nxv[j], nuj=s->nuv[j], ngj=s->ngv[j];
        int slack_offset=j<s->N ? s->nu : 0;
        memset(s->Q[j],0,(size_t)nxj*nxj*sizeof(double));
        memset(s->S[j],0,(size_t)nuj*nxj*sizeof(double));
        memset(s->R[j],0,(size_t)nuj*nuj*sizeof(double));
        memset(s->q[j],0,(size_t)nxj*sizeof(double));
        memset(s->r[j],0,(size_t)nuj*sizeof(double));
        memset(s->C[j],0,(size_t)ngj*nxj*sizeof(double));
        memset(s->D[j],0,(size_t)ngj*nuj*sizeof(double));
        if (j>0) {
            int k=j-1;
            for (int row=0; row<s->nx; ++row) {
                memcpy(&s->Q[j][row*nxj],
                    &state_Q[((size_t)k*s->nx+row)*s->nx],
                    (size_t)s->nx*sizeof(double));
                s->q[j][row]=state_q[(size_t)k*s->nx+row];
            }
            for (int row=0; row<s->ng; ++row) {
                memcpy(&s->C[j][row*nxj],
                    &general_C[((size_t)k*s->ng+row)*s->nx],
                    (size_t)s->nx*sizeof(double));
                s->lg[j][row]=general_lower[(size_t)k*s->ng+row];
                s->ug[j][row]=general_upper[(size_t)k*s->ng+row];
                s->lg_mask[j][row]=s->lg[j][row]>-0.5*inf ? 1.0 : 0.0;
                s->ug_mask[j][row]=s->ug[j][row]< 0.5*inf ? 1.0 : 0.0;
                if (!s->lg_mask[j][row]) s->lg[j][row]=0.0;
                if (!s->ug_mask[j][row]) s->ug[j][row]=0.0;
            }
            for (int craft=0; craft<s->nc; ++craft) {
                double scale=slack_scale[(size_t)k*s->nc+craft];
                s->D[j][craft*nuj+slack_offset+craft]=-scale;
                s->D[j][(s->nc+craft)*nuj+slack_offset+craft]=scale;
            }
            for (int craft=0; craft<s->nc; ++craft) {
                int index=slack_offset+craft;
                s->R[j][index*nuj+index]=2.0*s->slack_weight+reg;
            }
        }
        if (j<s->N) {
            for (int i=0; i<s->nu; ++i) {
                double prior=j==0 ? previous_control[i]
                    : control_nominal[(size_t)(j-1)*s->nu+i];
                double error=control_nominal[(size_t)j*s->nu+i]-prior;
                s->R[j][i*nuj+i]=2.0*(s->effort+s->rate)+reg;
                s->r[j][i]=2.0*s->effort*
                    control_nominal[(size_t)j*s->nu+i]+2.0*s->rate*error;
                if (j>0) {
                    int p=s->nx+i;
                    s->Q[j][p*nxj+p]+=2.0*s->rate;
                    s->q[j][p]-=2.0*s->rate*error;
                    s->S[j][i*nxj+p]=-2.0*s->rate;
                }
            }
        }
        for (int i=0; i<s->nbu[j]; ++i) {
            s->idxbu[j][i]=i;
            if (j<s->N && i<s->nu) {
                s->lbu[j][i]=control_lower[(size_t)j*s->nu+i];
                s->ubu[j][i]=control_upper[(size_t)j*s->nu+i];
                s->lbu_mask[j][i]=1.0; s->ubu_mask[j][i]=1.0;
            } else {
                s->lbu[j][i]=0.0; s->ubu[j][i]=0.0;
                s->lbu_mask[j][i]=1.0; s->ubu_mask[j][i]=0.0;
            }
        }
        if (j<s->N) {
            int nxnext=s->nxv[j+1];
            memset(s->A[j],0,(size_t)nxnext*nxj*sizeof(double));
            memset(s->B[j],0,(size_t)nxnext*nuj*sizeof(double));
            memset(s->b[j],0,(size_t)nxnext*sizeof(double));
            if (j>0) for (int row=0; row<s->nx; ++row)
                memcpy(&s->A[j][row*nxj],
                    &dyn_A[((size_t)j*s->nx+row)*s->nx],
                    (size_t)s->nx*sizeof(double));
            for (int row=0; row<s->nx; ++row)
                memcpy(&s->B[j][row*nuj],
                    &dyn_B[((size_t)j*s->nx+row)*s->nu],
                    (size_t)s->nu*sizeof(double));
            for (int i=0; i<s->nu; ++i)
                s->B[j][(s->nx+i)*nuj+i]=1.0;
        }
    }
    d_ocp_qp_set_all_rowmaj(s->A,s->B,s->b,s->Q,s->S,s->R,s->q,s->r,
        s->idxbx,s->lbx,s->ubx,s->idxbu,s->lbu,s->ubu,s->C,s->D,s->lg,s->ug,
        s->zeros,s->zeros,s->zeros,s->zeros,s->idxbx,s->zeros,s->zeros,&s->qp);
    for (int j=0; j<=s->N; ++j) {
        d_ocp_qp_set_lbu_mask(j,s->lbu_mask[j],&s->qp);
        d_ocp_qp_set_ubu_mask(j,s->ubu_mask[j],&s->qp);
        d_ocp_qp_set_lg_mask(j,s->lg_mask[j],&s->qp);
        d_ocp_qp_set_ug_mask(j,s->ug_mask[j],&s->qp);
    }
    for (int j=0; j<=s->N; ++j) {
        if (j>0) {
            int k=j-1, nxj=s->nxv[j];
            memset(s->tmp_x,0,(size_t)nxj*sizeof(double));
            memcpy(s->tmp_x,&warm[(size_t)k*s->nx],
                (size_t)s->nx*sizeof(double));
            memcpy(s->tmp_x+s->nx,
                &warm[(size_t)s->N*s->nx+(size_t)k*s->nu],
                (size_t)s->nu*sizeof(double));
            d_ocp_qp_sol_set_x(j,s->tmp_x,&s->sol);
        }
        int nuj=s->nuv[j];
        memset(s->tmp_u,0,(size_t)nuj*sizeof(double));
        if (j<s->N) memcpy(s->tmp_u,
            &warm[(size_t)s->N*s->nx+(size_t)j*s->nu],
            (size_t)s->nu*sizeof(double));
        if (j>0) memcpy(s->tmp_u+(j<s->N?s->nu:0),
            &warm[(size_t)s->N*(s->nx+s->nu)+(size_t)(j-1)*s->nc],
            (size_t)s->nc*sizeof(double));
        d_ocp_qp_sol_set_u(j,s->tmp_u,&s->sol);
    }
    clock_gettime(CLOCK_MONOTONIC,&solve_start);
    int status;
    if (s->cond_N>0 && s->cond_N<s->N) {
        d_part_cond_qp_cond(&s->qp,&s->cond_qp,&s->cond_arg,&s->cond_ws);
        d_ocp_qp_ipm_solve(&s->cond_qp,&s->cond_sol,
            &s->cond_ipm_arg,&s->cond_ipm_ws);
        d_ocp_qp_ipm_get_status(&s->cond_ipm_ws,&status);
        d_ocp_qp_ipm_get_iter(&s->cond_ipm_ws,iterations);
        d_ocp_qp_ipm_get_obj(&s->cond_ipm_ws,objective);
        d_ocp_qp_ipm_get_max_res_stat(&s->cond_ipm_ws,res_stat);
        d_ocp_qp_ipm_get_max_res_eq(&s->cond_ipm_ws,res_eq);
        d_ocp_qp_ipm_get_max_res_ineq(&s->cond_ipm_ws,res_ineq);
        d_ocp_qp_ipm_get_max_res_comp(&s->cond_ipm_ws,res_comp);
        d_part_cond_qp_expand_sol(&s->qp,&s->cond_qp,&s->cond_sol,&s->sol,
            &s->cond_arg,&s->cond_ws);
    } else {
        d_ocp_qp_ipm_solve(&s->qp,&s->sol,&s->arg,&s->ws);
        d_ocp_qp_ipm_get_status(&s->ws,&status);
        d_ocp_qp_ipm_get_iter(&s->ws,iterations);
        d_ocp_qp_ipm_get_obj(&s->ws,objective);
        d_ocp_qp_ipm_get_max_res_stat(&s->ws,res_stat);
        d_ocp_qp_ipm_get_max_res_eq(&s->ws,res_eq);
        d_ocp_qp_ipm_get_max_res_ineq(&s->ws,res_ineq);
        d_ocp_qp_ipm_get_max_res_comp(&s->ws,res_comp);
    }
    clock_gettime(CLOCK_MONOTONIC,&finish);
    *update_seconds=seconds_between(&update_start,&solve_start);
    *solve_seconds=seconds_between(&solve_start,&finish);
    memset(solution,0,(size_t)s->N*(s->nx+s->nu+s->nc)*sizeof(double));
    for (int j=1; j<=s->N; ++j) {
        int k=j-1;
        d_ocp_qp_sol_get_x(j,&s->sol,s->tmp_x);
        d_ocp_qp_sol_get_u(j,&s->sol,s->tmp_u);
        memcpy(&solution[(size_t)k*s->nx],s->tmp_x,
            (size_t)s->nx*sizeof(double));
        memcpy(&solution[(size_t)s->N*(s->nx+s->nu)+(size_t)k*s->nc],
            s->tmp_u+(j<s->N?s->nu:0),(size_t)s->nc*sizeof(double));
    }
    for (int j=0; j<s->N; ++j) {
        d_ocp_qp_sol_get_u(j,&s->sol,s->tmp_u);
        memcpy(&solution[(size_t)s->N*s->nx+(size_t)j*s->nu],
            s->tmp_u,(size_t)s->nu*sizeof(double));
    }
    return status;
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
        include_dir = casadi_dir / "include"
        hpipm_library = casadi_dir / "libhpipm.so"
        blasfeo_library = casadi_dir / "libblasfeo.so"
        if (not include_dir.is_dir() or not hpipm_library.is_file()
                or not blasfeo_library.is_file()):
            raise CompilationError(
                "CasADi installation does not contain HPIPM and BLASFEO")
        path = build_native_library(
            _SOURCE, stem="manta_hpipm", what="native HPIPM",
            compiler_flags=("-O3", "-march=native"),
            link_args=(f"-I{include_dir}", f"-L{casadi_dir}", "-lhpipm",
                       "-lblasfeo", f"-Wl,-rpath,{casadi_dir}"),
            timeout_s=60.0,
        ).path
        library = ctypes.CDLL(str(path))
        fp = np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        library.manta_hpipm_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_int,
        ]
        library.manta_hpipm_create.restype = ctypes.c_void_p
        library.manta_hpipm_destroy.argtypes = [ctypes.c_void_p]
        library.manta_hpipm_solve.argtypes = [
            ctypes.c_void_p, *([fp] * 13), fp,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        library.manta_hpipm_solve.restype = ctypes.c_int
        _LIBRARY = library
        return library


@dataclass(frozen=True)
class HPIPMResult:
    x: FloatArray
    cost: float
    iterations: int
    status: str
    success: bool
    update_ms: float
    iteration_ms: float
    stationarity_residual: float
    equality_residual: float
    inequality_residual: float
    complementarity_residual: float


class NativeHPIPM:
    """One fixed-dimension structured optimal-control QP workspace."""

    _STATUSES = {
        0: "solved",
        1: "maximum iterations reached",
        2: "minimum step reached",
        3: "NaN solution",
        4: "inconsistent equality constraints",
    }

    def __init__(
        self, horizon: int, nx: int, nu: int, nc: int, *,
        attitude_rows: int = 0,
        condense_to: int = 0,
        effort_weight: float,
        control_rate_weight: float,
        bank_slack_weight: float,
        tolerance: float = 2e-3,
        max_iter: int = 30,
    ) -> None:
        self.horizon, self.nx, self.nu, self.nc = (
            int(horizon), int(nx), int(nu), int(nc))
        self.attitude_rows = int(attitude_rows)
        if (self.horizon < 2 or self.nx < 1 or self.nu < 1 or self.nc < 1
                or self.attitude_rows not in (0, 3*self.nc)):
            raise ValueError("invalid HPIPM optimal-control dimensions")
        if not 0 <= int(condense_to) <= self.horizon:
            raise ValueError("HPIPM condense_to must be between 0 and horizon")
        if (not math.isfinite(float(tolerance)) or tolerance <= 0.0
                or max_iter < 1):
            raise ValueError("invalid HPIPM tolerance or iteration limit")
        weights = (effort_weight, control_rate_weight, bank_slack_weight)
        if any(not math.isfinite(float(value)) or value < 0.0
               for value in weights):
            raise ValueError("HPIPM cost weights must be finite and non-negative")
        self.ng = 2*self.nc + self.attitude_rows
        self.nvar = self.horizon*(self.nx+self.nu+self.nc)
        self._library = _library()
        self._handle = self._library.manta_hpipm_create(
            self.horizon, self.nx, self.nu, self.nc,
            self.attitude_rows, int(condense_to),
            float(effort_weight), float(control_rate_weight),
            float(bank_slack_weight), float(tolerance), int(max_iter))
        if not self._handle:
            raise RuntimeError("native HPIPM workspace setup failed")

    def close(self) -> None:
        if self._handle:
            self._library.manta_hpipm_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
        result = np.ascontiguousarray(value, dtype=np.float64)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        return result.reshape(-1)

    def solve(
        self, dynamics_A: Any, dynamics_B: Any,
        state_hessians: Any, state_gradients: Any,
        nominal_controls: Any, previous_control: Any,
        control_lower: Any, control_upper: Any,
        general_jacobians: Any, general_lower: Any, general_upper: Any,
        slack_scales: Any, warm_start: Any,
    ) -> HPIPMResult:
        n, nx, nu, nc, ng = (
            self.horizon, self.nx, self.nu, self.nc, self.ng)
        values = (
            self._array(dynamics_A, (n, nx, nx), "HPIPM dynamics A"),
            self._array(dynamics_B, (n, nx, nu), "HPIPM dynamics B"),
            self._array(state_hessians, (n, nx, nx), "HPIPM state Hessians"),
            self._array(state_gradients, (n, nx), "HPIPM state gradients"),
            self._array(nominal_controls, (n, nu), "HPIPM nominal controls"),
            self._array(previous_control, (nu,), "HPIPM previous control"),
            self._array(control_lower, (n, nu), "HPIPM control lower bounds"),
            self._array(control_upper, (n, nu), "HPIPM control upper bounds"),
            self._array(
                general_jacobians, (n, ng, nx),
                "HPIPM general constraint Jacobians"),
            self._array(general_lower, (n, ng), "HPIPM general lower bounds"),
            self._array(general_upper, (n, ng), "HPIPM general upper bounds"),
            self._array(slack_scales, (n, nc), "HPIPM slack scales"),
            self._array(warm_start, (self.nvar,), "HPIPM warm start"),
        )
        solution = np.empty(self.nvar)
        objective = ctypes.c_double()
        iterations = ctypes.c_int()
        residuals = [ctypes.c_double() for _ in range(4)]
        update_seconds, solve_seconds = ctypes.c_double(), ctypes.c_double()
        status_value = int(self._library.manta_hpipm_solve(
            self._handle, *values, solution, ctypes.byref(objective),
            ctypes.byref(iterations), *(ctypes.byref(v) for v in residuals),
            ctypes.byref(update_seconds), ctypes.byref(solve_seconds)))
        if status_value < 0:
            raise RuntimeError(
                f"native HPIPM numeric update failed with {status_value}")
        return HPIPMResult(
            x=solution, cost=float(objective.value),
            iterations=int(iterations.value),
            status=self._STATUSES.get(status_value, f"status {status_value}"),
            success=status_value == 0,
            update_ms=1e3*float(update_seconds.value),
            iteration_ms=1e3*float(solve_seconds.value),
            stationarity_residual=float(residuals[0].value),
            equality_residual=float(residuals[1].value),
            inequality_residual=float(residuals[2].value),
            complementarity_residual=float(residuals[3].value),
        )


__all__ = ["HPIPMResult", "NativeHPIPM"]
