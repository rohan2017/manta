"""MPC — receding-horizon point-to-point control over the compiled tick.

`MPC(world, u_bounds=..., ...)` is a controller-synthesis transform, a
sibling of `LQR`: it consumes the model and emits a typed `Module`
whose one entry point is the fixed-work real-time-iteration tick

    tick(x, target, weights, PT, plan) → (plan', u, J)

`sweeps` Gauss-Newton iLQR iterations over the world's own compiled
step (`Sim(world).module().functions["step"]` — the dynamics are never
written here), each = rollout → batch linearize → backward Riccati
sweep with one-shot box-limit masking → forward rollouts at a fixed α
ladder, best kept by cost. The warm plan is the Module's HELD state:
each call solves from it, returns the first control, and writes back
the once-shifted remainder — so a backend runtime is receding-horizon
MPC by construction, with no controller code of its own.

Two guarantees the tick shape carries (both measured in the manta-mpc
lab, where this transform was prototyped — see its README for the
numbers):

* **Never worse than the warm plan**: α = 0 (the plan's own rollout)
  is always among the scored candidates, so a tick cannot degrade the
  plan it was handed — a failed improvement IS open-loop continuation.
* **Fixed work**: every data-dependent branch of a line-searched
  solver became fixed work selected inside the graph (`if_else`), so
  the tick is ONE dataflow function: `Target*` backends lower it like
  any other kernel, latency is constant by construction, and the C
  emitted from the loop-structured graph is constant-size in horizon
  and ladder length.

Cost family — targets and weights are RUNTIME DATA, not baked
constants (the mirror of the baked-coefficients doctrine: model
coefficients are the vehicle's physics and recompile; targets and
weights are the CALLER's intent and change at every guidance-law
switch). A mask is a zero weight. The running terms, all maskable:

    per-axis position   w_p ∘ (p − p_ref)²      go-to; depth-hold is
                                                w_p = (0, 0, w)
    per-axis velocity   w_v ∘ (v − v_ref)²      cruise: guidance turns
                        (world frame)           "speed s on course d"
                                                into v_ref = s·d
    course/nose         w_h·(1 − nose(q)·d_ref) heading WITHOUT Euler
                                                angles: exactly
                                                quadratic in q via
                                                nose·d = qᵀM(d)q, so
                                                derivatives are exact
    per-channel effort  w_u ∘ u²

plus the terminal quadratic eᵀ·PT·e over (p − p_ref, v − v_ref),
where PT is a runtime MATRIX port (default = the built terminal).
The constructor's weight arguments set the DEFAULT objective — the
legacy point-to-point family — and `terminal="lqr"` builds the
default PT from the Riccati cost-to-go at the goal equilibrium (P
from this package's own `LQR`; requires a stabilizable rest
equilibrium — an underactuated hull gets a clear Riccati failure and
keeps explicit weights). When running weights change at a law
switch, re-solve P in milliseconds via `LQR.resolve_at` and push it
through the port. Whether an unmasked axis means "free" (w = 0) or
"hold current" (target = current, w > 0) is the CALLER's decision —
guidance semantics never leak in here.

Plan birth: the tick bootstraps from the zero plan for hulls that can
act from rest; for underactuated hulls or basin-sensitive transits,
`MPC.plan(x, goal)` runs the OFFLINE solver — adaptive line search,
μ-scheduled Levenberg regularization, optional multi-start seed sweep
and full-DDP second-order terms — and its plan seeds the runtime via
`NumpyMpc.reset_plan`. Plan birth is a mission-load event and stays
Python; the tick is the flight loop and is the compiled artifact.

Coefficients are BAKED (doctrine 2026-08-09): feedback absorbs small
model drift (a +10 % module swap flies on the nominal controller);
changes big enough to matter are a new controller and recompile.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role,
    StateField, StateLayout, StateRef,
)
from .lqr import LQR

#: Forward-pass step ladder: a line search's candidate set, evaluated
#: unconditionally (plus the implicit α = 0 — the warm plan itself).
ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01)


class MPC:
    """Receding-horizon point-to-point MPC about the world's dynamics.

    Args:
        world      — the model; the craft's compiled step supplies the
                     dynamics and the state layout.
        u_bounds   — `{input suffix: (lo, hi)}` box limits, keyed like
                     the Part Input names without the craft prefix
                     (`"prop.throttle"`). Every input must be bounded.
        horizon    — shooting nodes N; lookahead = N·dt. The dominant
                     compute lever (cost is linear in it).
        dt         — shooting interval = the controller period.
        substeps   — explicit tick evaluations folded into each node
                     (stability of drag/servo time constants, not
                     accuracy of the optimum).
        w_pos_running, w_u — running cost weights.
        terminal   — "weights" (explicit `w_pos_terminal` /
                     `w_vel_terminal`) or "lqr" (Riccati cost-to-go at
                     the goal equilibrium; raises for hulls without a
                     stabilizable rest equilibrium).
        sweeps     — Gauss-Newton iterations per tick.
        alphas     — the stepped forward-pass ladder.
        mu         — fixed Levenberg regularization on Quu.
        parallel   — None or "openmp": fan the linearization map and
                     the α rollouts across cores at codegen time.

    The emitted Module (see `.module()`): HELD state `plan` (nu × N);
    ports `x` (STATE, full manifold), `target`/`weights`/`PT` (the
    runtime objective, defaults = the constructor's), `u` (CONTROL,
    one field per Part Input), `J` (OUTPUT, the plan cost); one entry
    `tick` that writes the shifted plan and returns (u, J).
    """

    def __init__(self, world, *,
                 u_bounds: dict[str, tuple[float, float]],
                 horizon: int = 40, dt: float = 0.25, substeps: int = 5,
                 w_pos_terminal: float = 120.0,
                 w_vel_terminal: float = 0.0,
                 w_pos_running: float = 0.4,
                 w_u: float = 0.05,
                 terminal: str = "weights",
                 sweeps: int = 2,
                 alphas: tuple[float, ...] = ALPHAS,
                 mu: float = 1e-6,
                 parallel: str | None = None) -> None:
        if not world.crafts:
            raise ValueError("MPC: world has no crafts.")
        if terminal not in ("weights", "lqr"):
            raise ValueError(f"MPC: terminal must be 'weights' or "
                             f"'lqr', got {terminal!r}.")

        from ..sim import Sim
        sim_module = Sim(world).module()
        step = sim_module.functions["step"].expand()
        spec = sim_module.state.fields[0].manifold
        x_init = np.asarray(sim_module.state.fields[0].init,
                            dtype=float).reshape(-1)

        self.horizon, self.dt = int(horizon), float(dt)
        nx = step.size_in(0)[0]
        n_noise = step.size_in(2)[0]
        u_fields = sim_module.port("u").fields
        self.input_names = tuple(f.name for f in u_fields)
        nu = len(self.input_names)
        if nu == 0:
            raise ValueError("MPC: world has no Part Inputs — no "
                             "control authority.")
        lo, hi = [], []
        for full in self.input_names:
            suffix = full.split(".", 1)[1]
            if suffix not in u_bounds:
                raise KeyError(f"MPC: no bounds declared for input "
                               f"{full!r}")
            b = u_bounds[suffix]
            lo.append(float(b[0]))
            hi.append(float(b[1]))
        u_lo, u_hi = np.asarray(lo), np.asarray(hi)
        self.u_lo, self.u_hi = u_lo, u_hi
        self.nx, self.nu = nx, nu

        craft = world.crafts[0].name
        pos = spec.slot(f"{craft}.position")
        vel = spec.slot(f"{craft}.velocity")
        quat = spec.slot(f"{craft}.orientation")
        po, pd = pos.ambient_offset, pos.ambient_dim
        vo = vel.ambient_offset
        qo = quat.ambient_offset

        # ---- terminal quadratic: eᵀ·PT·e, e = x − xref(goal) ----------
        if terminal == "lqr":
            # The lab's lqr_terminal, inlined: Q/R restate the running
            # cost per step, so P is the same objective's true tail.
            # Position-invariant dynamics ⇒ one solve serves every goal;
            # Euclidean regulated slots ⇒ P scatters into ambient.
            lqr = LQR(world, x_ref={}, dt=self.dt,
                      regulate=[f"{craft}.position", f"{craft}.velocity"])
            Q = np.zeros((6, 6))
            Q[:3, :3] = self.dt * w_pos_running * np.eye(3)
            R = self.dt * w_u * np.eye(len(lqr.input_names))
            P = lqr.resolve_at(Q=Q, R=R).P
            PT = np.zeros((nx, nx))
            idx = np.r_[po:po + pd, vo:vo + 3]
            PT[np.ix_(idx, idx)] = 0.5 * (P + P.T)
        else:
            d = np.zeros(nx)
            d[po:po + pd] = w_pos_terminal
            d[vo:vo + 3] = w_vel_terminal
            PT = np.diag(d)

        # ---- shooting kernels over the compiled step -------------------
        x_s = ca.SX.sym("x", nx)
        u_s = ca.SX.sym("u", nu)
        zero_noise = ca.SX.zeros(n_noise)
        dt_sub = self.dt / int(substeps)
        x_next = x_s
        for j in range(int(substeps)):
            x_next = step(x_next, u_s, zero_noise, dt_sub, j * dt_sub)
        f = ca.Function("f", [x_s, u_s], [x_next])
        fj = ca.Function("fj", [x_s, u_s],
                         [x_next, ca.jacobian(x_next, x_s),
                          ca.jacobian(x_next, u_s)])

        tick = _build_tick(
            f, fj, nx=nx, nu=nu, N=self.horizon, dt=self.dt,
            po=po, pd=pd, vo=vo, qo=qo, u_lo=u_lo, u_hi=u_hi,
            sweeps=int(sweeps), alphas=tuple(alphas), mu=float(mu),
            parallel=parallel)

        # the DEFAULT objective — the legacy point-to-point family:
        # hold the initial position, velocity/heading masked, uniform
        # effort. target = [p_ref(3), v_ref(3), d_ref(3)]; weights =
        # [w_p(3), w_v(3), w_h, w_u(nu)].
        t0 = np.zeros(9)
        t0[0:3] = x_init[po:po + pd]
        t0[6] = 1.0                              # d_ref: unit +x
        w0 = np.zeros(7 + nu)
        w0[0:3] = float(w_pos_running)
        w0[7:] = float(w_u)
        self._t0, self._w0 = t0, w0

        # kept for the offline plan-birth solver (`plan()`)
        self._f, self._fj = f, fj
        self._fj_map = fj.map(self.horizon)
        self._fh: ca.Function | None = None      # built on first fxx use
        self._spec = spec
        self._x_init = x_init
        self._PT = PT
        self._po, self._pd = po, pd
        self._vo, self._qo = vo, qo

        self._module = Module(
            name=f"{world.name}_mpc",
            state=StateLayout((StateField(
                "plan", kind="matrix", shape=(nu, self.horizon),
                init=np.zeros((nu, self.horizon))),)),
            ports=(
                Port("x", Role.STATE, (spec.ambient_dim,),
                     manifold=spec, init=x_init),
                Port("target", Role.MATRIX, (9, 1),
                     init=t0.reshape(9, 1)),
                Port("weights", Role.MATRIX, (7 + nu, 1),
                     init=w0.reshape(7 + nu, 1)),
                Port("PT", Role.MATRIX, (nx, nx), init=PT),
                Port("u", Role.CONTROL, (nu,), fields=tuple(
                    PortField(f.name, 1, float(f.default))
                    for f in u_fields)),
                Port("J", Role.OUTPUT, (1,)),
            ),
            functions={"tick": tick},
            entry_points=(EntryPoint(
                "tick", "tick",
                (PortRef("x"), PortRef("target"), PortRef("weights"),
                 PortRef("PT"), StateRef("plan")),
                writes=("plan",), returns=("u", "J")),),
            hosting=Hosting.HELD)

    def module(self) -> Module:
        """The typed `Module` IR a backend lowers."""
        return self._module

    # ------------------------------------------------------------------
    # Plan birth — the offline solver
    # ------------------------------------------------------------------

    def objective(self, *, goal=None, velocity=None, heading=None,
                  w_position=None, w_velocity=None, w_heading=None,
                  w_effort=None, P=None):
        """Assemble (target, weights, PT) vectors over the built
        defaults — the same currency the tick's runtime ports carry.

        `goal`/`velocity` are world-frame 3-vectors; `heading` is a
        direction (2- or 3-vector, normalized here); the w_* are
        scalars (broadcast) or per-axis/per-channel vectors; `P` a
        replacement terminal matrix (e.g. `LQR.resolve_at(...).P`
        scattered — or just rescaled — when running weights change)."""
        t = self._t0.copy()
        w = self._w0.copy()
        PT = self._PT
        if goal is not None:
            t[0:3] = np.asarray(goal, dtype=float).ravel()
        if velocity is not None:
            t[3:6] = np.asarray(velocity, dtype=float).ravel()
        if heading is not None:
            d = np.zeros(3)
            h = np.asarray(heading, dtype=float).ravel()
            d[:h.size] = h
            n = np.linalg.norm(d)
            if n < 1e-9:
                raise ValueError("objective: heading must be a "
                                 "non-zero direction")
            t[6:9] = d / n
        for lo, hi, val in ((0, 3, w_position), (3, 6, w_velocity),
                            (6, 7, w_heading), (7, 7 + self.nu,
                                                w_effort)):
            if val is not None:
                w[lo:hi] = np.asarray(val, dtype=float).ravel()
        if P is not None:
            PT = np.asarray(P, dtype=float).reshape(self.nx, self.nx)
        return t, w, PT

    def plan(self, x, goal=None, *, u_init=None,
             multi_start: bool = False,
             second_order: bool = False, max_iter: int = 300,
             tol: float = 1e-5, **objective) -> "MpcPlan":
        """Solve for a seed plan OFFLINE — mission load, goal change,
        basin doubt. Same cost, same kernels, same box limits as the
        tick, but with everything the fixed-work tick gave up:
        adaptive backtracking line search, μ-scheduled Levenberg
        regularization, and optionally

        * `multi_start` — one quarter-horizon pulse seed per input
          channel plus the jittered zero, best kept: the
          vehicle-agnostic answer to bilinear-payoff saddles that
          Gauss-Newton cannot see from any single start (a symmetric
          hull's crab basin — measured in the manta-mpc lab).
        * `second_order` — full DDP: contract the next Vx against the
          dynamics' second derivatives in the backward sweep (with
          state-space regularization and a PD gate on Quu — both
          load-bearing). Escapes those saddles from a SINGLE start at
          ~2× per-iteration cost; best used with modest terminal
          weights ("lqr"), where the value gradients keep the
          second-order model honest.

        `x` as packed vector or dict; returns an `MpcPlan` whose `.U`
        (nu × N) feeds `NumpyMpc.reset_plan`. Improvement is monotone
        from whatever `u_init` is given — hand it a cruise and it can
        only make the cruise better.

        The objective: `goal` is the position-target shorthand; the
        full family (velocity, heading, per-term weights, terminal P)
        comes through keyword arguments — see `objective()`. Same
        currency as the tick, so a plan born here is priced exactly
        as the runtime will price it.
        """
        x0 = (np.asarray(x, dtype=float).reshape(-1)
              if isinstance(x, np.ndarray)
              else self._spec.pack_any(x, base=self._x_init))
        tv, wv, PT = self.objective(goal=goal, **objective)
        if second_order and self._fh is None:
            xs = ca.SX.sym("x", self.nx)
            us = ca.SX.sym("u", self.nu)
            lam = ca.SX.sym("lam", self.nx)
            H, _ = ca.hessian(ca.dot(lam, self._f(xs, us)),
                              ca.vertcat(xs, us))
            self._fh = ca.Function("fh", [xs, us, lam], [H])

        if not multi_start:
            return self._solve(x0, tv, wv, PT, u_init, second_order,
                               max_iter, tol)
        N, nu = self.horizon, self.nu
        seeds: list = [None] if u_init is None else [u_init]
        for ch in range(nu):
            U = np.zeros((N, nu))
            U[:N // 4, ch] = self.u_hi[ch]
            seeds.append(U)
            if self.u_lo[ch] < 0.0:
                U = np.zeros((N, nu))
                U[:N // 4, ch] = self.u_lo[ch]
                seeds.append(U)
        best: MpcPlan | None = None
        for seed in seeds:
            r = self._solve(x0, tv, wv, PT, seed, second_order,
                            max_iter, tol)
            if best is None or r.cost < best.cost:
                best = r
        assert best is not None
        return best

    # ---- solver internals (ported from the manta-mpc lab's Ddp) -------

    def _rollout(self, x0, U):
        X = np.empty((self.horizon + 1, self.nx))
        X[0] = x0
        for i in range(self.horizon):
            X[i + 1] = np.asarray(self._f(X[i], U[i])).ravel()
        return X

    def _cost(self, X, U, tv, wv, PT):
        po, pd = self._po, self._pd
        vo, qo = self._vo, self._qo
        p_ref, v_ref = tv[0:3], tv[3:6]
        wp, wvel, wh = wv[0:3], wv[3:6], wv[6]
        wu = wv[7:]
        Mh = np.array(_heading_M(*tv[6:9]))
        p_err = X[:-1, po:po + pd] - p_ref
        v_err = X[:-1, vo:vo + 3] - v_ref
        Q = X[:-1, qo:qo + 4]
        head = np.einsum("ij,jk,ik->i", Q, Mh, Q)
        e = X[-1].copy()
        e[po:po + pd] -= p_ref
        e[vo:vo + 3] -= v_ref
        return float(self.dt * (np.sum((U * U) * wu)
                                + np.sum((p_err * p_err) * wp)
                                + np.sum((v_err * v_err) * wvel)
                                + wh * np.sum(1.0 - head))
                     + e @ PT @ e)

    def _solve(self, x0, tv, wv, PT, u_init, second_order, max_iter,
               tol):
        N, nu, nx = self.horizon, self.nu, self.nx
        if u_init is None:
            # a deterministic whisper of control: exactly symmetric
            # trajectories sit on the saddles this solver must leave
            rng = np.random.default_rng(0)
            span = np.maximum(self.u_hi - self.u_lo, 1e-9)
            U = 0.01 * span * rng.standard_normal((N, nu))
        else:
            u_init = np.asarray(u_init, dtype=float)
            U = (np.tile(u_init, (N, 1)) if u_init.ndim == 1
                 else u_init.copy())
            if U.shape == (nu, N):
                U = U.T.copy()
        U = np.clip(U, self.u_lo, self.u_hi)

        X = self._rollout(x0, U)
        J = self._cost(X, U, tv, wv, PT)
        trace = [J]
        mu, mu_min, mu_max = 1e-6, 1e-9, 1e8
        stall = 0
        converged = False

        for it in range(max_iter):
            _, A_flat, B_flat = self._fj_map(X[:N].T, U.T)
            A_flat, B_flat = np.asarray(A_flat), np.asarray(B_flat)
            A = np.stack([A_flat[:, i * nx:(i + 1) * nx]
                          for i in range(N)])
            B = np.stack([B_flat[:, i * nu:(i + 1) * nu]
                          for i in range(N)])

            while True:
                ok, kff, K, dV1, dV2 = self._backward(
                    A, B, X, U, tv, wv, PT, mu, second_order)
                if ok:
                    break
                mu = mu * 10.0
                if mu > mu_max:
                    return MpcPlan(U.T.copy(), J, it, False)

            accepted = False
            for alpha in ALPHAS:
                X_new, U_new = self._forward(x0, X, U, kff, K, alpha)
                J_new = self._cost(X_new, U_new, tv, wv, PT)
                expected = -(alpha * dV1 + alpha * alpha * dV2)
                if not np.isfinite(J_new):
                    continue
                if J - J_new > max(1e-4 * expected, 0.0):
                    accepted = True
                    break
            if accepted:
                rel = (J - J_new) / max(abs(J), 1e-12)
                X, U, J = X_new, U_new, J_new
                trace.append(J)
                mu = max(mu / 5.0, mu_min)
                stall = stall + 1 if rel < tol else 0
                if stall >= 3:
                    converged = True
                    break
            else:
                mu = mu * 10.0
                stall += 1
                if mu > mu_max or stall >= 8:
                    converged = trace[-1] < trace[0]
                    break

        return MpcPlan(U.T.copy(), J, len(trace) - 1, converged)

    def _backward(self, A, B, X, U, tv, wv, PT, mu, second_order):
        """Control-limited backward pass (Tassa box-QP): honest
        promised decrease under saturation. With `second_order`, the
        Vx·fxx contraction plus state-space regularization (μ on Vxx,
        not Quu) and an explicit PD gate — indefinite curvature routes
        to μ escalation, never to the line search."""
        N, nx, nu = self.horizon, self.nx, self.nu
        po, pd = self._po, self._pd
        vo, qo = self._vo, self._qo
        dt = self.dt
        p_ref, v_ref = tv[0:3], tv[3:6]
        wp, wvel, wh = wv[0:3], wv[3:6], wv[6]
        wu = wv[7:]
        Mh = np.array(_heading_M(*tv[6:9]))
        luu = np.diag(2.0 * dt * wu)
        lxx_c = np.zeros((nx, nx))
        lxx_c[po:po + pd, po:po + pd] = np.diag(2.0 * dt * wp)
        lxx_c[vo:vo + 3, vo:vo + 3] = np.diag(2.0 * dt * wvel)
        lxx_c[qo:qo + 4, qo:qo + 4] = -2.0 * dt * wh * Mh

        e = X[N].copy()
        e[po:po + pd] -= p_ref
        e[vo:vo + 3] -= v_ref
        Vx = 2.0 * (PT @ e)
        Vxx = 2.0 * PT.copy()
        kff = np.zeros((N, nu))
        K = np.zeros((N, nu, nx))
        dV1 = dV2 = 0.0
        for i in range(N - 1, -1, -1):
            Ai, Bi = A[i], B[i]
            lx = np.zeros(nx)
            lx[po:po + pd] = 2.0 * dt * wp * (X[i, po:po + pd] - p_ref)
            lx[vo:vo + 3] = 2.0 * dt * wvel * (X[i, vo:vo + 3] - v_ref)
            lx[qo:qo + 4] = -2.0 * dt * wh * (Mh @ X[i, qo:qo + 4])
            lu = 2.0 * dt * wu * U[i]
            Qx = lx + Ai.T @ Vx
            Qu = lu + Bi.T @ Vx
            VxxA = Vxx @ Ai
            Qxx = lxx_c + Ai.T @ VxxA
            Qux = Bi.T @ VxxA
            Quu = luu + Bi.T @ Vxx @ Bi
            if not second_order:
                Qux_reg = Qux
                Quu_reg = Quu + mu * np.eye(nu)
            else:
                H = np.asarray(self._fh(X[i], U[i], Vx))
                Hux = H[nx:, :nx]
                Huu = 0.5 * (H[nx:, nx:] + H[nx:, nx:].T)
                Qxx = Qxx + H[:nx, :nx]
                Qux = Qux + Hux
                Quu = Quu + Huu
                VxxR = Vxx + mu * np.eye(nx)
                Qux_reg = Bi.T @ VxxR @ Ai + Hux
                Quu_reg = luu + Bi.T @ VxxR @ Bi + Huu
                Quu_reg = 0.5 * (Quu_reg + Quu_reg.T)
                try:
                    np.linalg.cholesky(Quu_reg)
                except np.linalg.LinAlgError:
                    return False, None, None, 0.0, 0.0

            delta, free = _boxqp(Quu_reg, Qu,
                                 self.u_lo - U[i], self.u_hi - U[i])
            if delta is None:
                return False, None, None, 0.0, 0.0
            kff[i] = delta
            K[i] = 0.0
            if free.any():
                idx = np.flatnonzero(free)
                K[i][np.ix_(idx, range(nx))] = -np.linalg.solve(
                    Quu_reg[np.ix_(idx, idx)], Qux_reg[idx])
            dV1 += float(Qu @ delta)
            dV2 += 0.5 * float(delta @ Quu @ delta)
            Vx = Qx + K[i].T @ (Quu @ kff[i] + Qu) + Qux.T @ kff[i]
            Vxx = Qxx + K[i].T @ (Quu @ K[i] + Qux) + Qux.T @ K[i]
            Vxx = 0.5 * (Vxx + Vxx.T)
        return True, kff, K, dV1, dV2

    def _forward(self, x0, X_bar, U_bar, kff, K, alpha):
        N = self.horizon
        X = np.empty_like(X_bar)
        U = np.empty_like(U_bar)
        X[0] = x0
        for i in range(N):
            u = U_bar[i] + alpha * kff[i] + K[i] @ (X[i] - X_bar[i])
            U[i] = np.clip(u, self.u_lo, self.u_hi)
            X[i + 1] = np.asarray(self._f(X[i], U[i])).ravel()
        return X, U

    def __repr__(self) -> str:
        return (f"<MPC N={self.horizon} dt={self.dt} nx={self.nx} "
                f"nu={self.nu} inputs={list(self.input_names)}>")


@dataclass(frozen=True)
class MpcPlan:
    """One offline plan-birth solve, as data.

    `U` is (nu × N) — the Module's plan orientation, so it feeds
    `NumpyMpc.reset_plan(plan.U)` directly; `cost` is the same
    currency as the runtime's `.J` (compare them to judge how much
    the tick has polished or lost since birth).
    """
    U: np.ndarray
    cost: float
    iterations: int
    converged: bool


def _boxqp(H, q, lo, hi, *, tol=1e-8, max_iter=25):
    """min ½δᵀHδ + qᵀδ over the box [lo, hi] — projected Newton with
    active-set identification (Tassa's boxQP, compact). Returns
    (δ, free_mask); (None, None) when the free block loses positive
    definiteness — the caller's cue to raise μ."""
    delta = np.clip(np.zeros_like(q), lo, hi)
    free = np.ones_like(q, dtype=bool)
    value = lambda d: 0.5 * d @ H @ d + q @ d
    v = value(delta)
    for _ in range(max_iter):
        g = q + H @ delta
        clamped = (((delta <= lo + 1e-10) & (g > 0.0))
                   | ((delta >= hi - 1e-10) & (g < 0.0)))
        free = ~clamped
        if not free.any():
            return delta, free
        idx = np.flatnonzero(free)
        try:
            step_f = -np.linalg.solve(H[np.ix_(idx, idx)], g[idx])
        except np.linalg.LinAlgError:
            return None, None
        if np.linalg.norm(g[idx]) < tol:
            return delta, free
        improved = False
        for a in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
            cand = delta.copy()
            cand[idx] = delta[idx] + a * step_f
            np.clip(cand, lo, hi, out=cand)
            cv = value(cand)
            if cv < v - 1e-12:
                delta, v, improved = cand, cv, True
                break
        if not improved:
            return delta, free
    return delta, free


def _heading_M(d1, d2, d3):
    """The symmetric 4×4 M(d) with nose(q)·d = qᵀ M q for unit q —
    the heading cost's exactly-quadratic form (works for numpy scalars
    and MX/SX symbols alike)."""
    z = d1 * 0.0
    return [[d1,   z, -d3,  d2],
            [z,   d1,  d2,  d3],
            [-d3, d2, -d1,   z],
            [d2,  d3,   z, -d1]]


def _build_tick(f: ca.Function, fj: ca.Function, *, nx: int, nu: int,
                N: int, dt: float, po: int, pd: int, vo: int, qo: int,
                u_lo: np.ndarray, u_hi: np.ndarray,
                sweeps: int, alphas: tuple[float, ...],
                mu: float, parallel: str | None) -> ca.Function:
    """The loop-structured RTI tick:
    (x0, target, weights, PT, plan) → (plan', u0, J).

    Structure is the manta-mpc lab's `rti._build_tick`, with the shift
    folded in: `mapaccum` for the scans (rollouts, the backward Riccati
    sweep), `map` for the per-node linearization and all α rollouts —
    so generated C is loops over one emitted body per kernel, constant
    in N and ladder size. The backward step is one SX function whose
    small solves expand to scalar ops (Linsol-free, the same discipline
    that keeps the world tick expandable); box limits enter as a
    one-shot masked active set, honest for warm ticks where the active
    set is stable between solves.

    The objective (targets, weights, terminal PT) enters as runtime
    INPUTS: one compiled kernel serves every command family, masks are
    zero weights, and nothing here recompiles at a guidance-law
    switch. The heading term rides on nose·d = qᵀM(d)q — exactly
    quadratic in the quaternion, so lx/lxx stay closed-form (M is
    indefinite; μ and the box-QP's PD handling absorb that, same as
    any other curvature).
    """
    lo_dm, hi_dm = ca.DM(u_lo), ca.DM(u_hi)

    # ---- backward step (accumulators Vx, Vxx first: mapaccum) ---------
    A = ca.SX.sym("A", nx, nx)
    B = ca.SX.sym("B", nx, nu)
    lx = ca.SX.sym("lx", nx)
    lu = ca.SX.sym("lu", nu)
    lxx = ca.SX.sym("lxx", nx, nx)
    wu_s = ca.SX.sym("wu", nu)
    ubar = ca.SX.sym("ubar", nu)
    Vx = ca.SX.sym("Vx", nx)
    Vxx = ca.SX.sym("Vxx", nx, nx)

    Qx = lx + A.T @ Vx
    Qu = lu + B.T @ Vx
    VxxA = Vxx @ A
    Qxx = lxx + A.T @ VxxA
    Qux = B.T @ VxxA
    Quu = 2.0 * dt * ca.diag(wu_s) + B.T @ Vxx @ B
    Quu_reg = Quu + mu * ca.SX.eye(nu)

    blo = lo_dm - ubar
    bhi = hi_dm - ubar
    d_unc = -ca.solve(Quu_reg, Qu)
    d_clip = ca.fmin(ca.fmax(d_unc, blo), bhi)
    c = ca.fmax(d_unc > bhi, d_unc < blo)
    D = ca.diag(c)
    F = ca.SX.eye(nu) - D
    M = F @ Quu_reg @ F + D
    r = -F @ (Qu + Quu_reg @ (D @ d_clip)) + D @ d_clip
    delta = ca.solve(M, r)
    K = -ca.solve(M, F @ Qux)

    Vx_n = Qx + K.T @ (Quu @ delta + Qu) + Qux.T @ delta
    Vxx_n = Qxx + K.T @ (Quu @ K + Qux) + Qux.T @ K
    Vxx_n = 0.5 * (Vxx_n + Vxx_n.T)
    bstep = ca.Function("bstep",
                        [Vx, Vxx, A, B, lx, lu, lxx, wu_s, ubar],
                        [Vx_n, Vxx_n, delta, K])

    # ---- forward step with feedback (accumulator x) --------------------
    xf = ca.MX.sym("x", nx)
    ub_f = ca.MX.sym("ub", nu)
    kf_f = ca.MX.sym("kf", nu)
    Kf_f = ca.MX.sym("Kf", nu, nx)
    xb_f = ca.MX.sym("xb", nx)
    al_f = ca.MX.sym("al")
    u_ap = ub_f + al_f * kf_f + Kf_f @ (xf - xb_f)
    u_ap = ca.fmin(ca.fmax(u_ap, lo_dm), hi_dm)
    fstep = ca.Function("fstep", [xf, ub_f, kf_f, Kf_f, xb_f, al_f],
                        [f(xf, u_ap), u_ap])

    # ---- loop combinators ----------------------------------------------
    roll = f.mapaccum("roll", N)
    fj_map = fj.map(N, parallel) if parallel else fj.map(N)
    bsweep = bstep.mapaccum("bsweep", N, 2, {})
    fsweep = fstep.mapaccum("fsweep", N, 1, {})
    n_al = len(alphas)
    fsweep_all = (fsweep.map(n_al, parallel) if parallel
                  else fsweep.map(n_al))
    al_row = ca.DM(np.repeat(np.asarray(alphas, dtype=float), N)).T

    rev_x = [i * nx + j for i in range(N - 1, -1, -1) for j in range(nx)]
    rev_u = [i * nu + j for i in range(N - 1, -1, -1) for j in range(nu)]

    # ---- the tick graph --------------------------------------------------
    x0 = ca.MX.sym("x0", nx)
    target = ca.MX.sym("target", 9)
    weights = ca.MX.sym("weights", 7 + nu)
    PTm = ca.MX.sym("PT", nx, nx)
    plan = ca.MX.sym("plan", nu, N)

    p_ref, v_ref = target[0:3], target[3:6]
    wp, wv, wh = weights[0:3], weights[3:6], weights[6]
    wu = weights[7:]
    Mh = ca.vertcat(*[ca.horzcat(*row) for row in
                      _heading_M(target[6], target[7], target[8])])

    xref = ca.MX.zeros(nx)
    xref[po:po + pd] = p_ref
    xref[vo:vo + 3] = v_ref
    wpN, wvN = ca.repmat(wp, 1, N), ca.repmat(wv, 1, N)
    wuN = ca.repmat(wu, 1, N)
    lxx_c = ca.MX.zeros(nx, nx)
    for j in range(pd):
        lxx_c[po + j, po + j] = 2.0 * dt * wp[j]
    for j in range(3):
        lxx_c[vo + j, vo + j] = 2.0 * dt * wv[j]
    lxx_c[qo:qo + 4, qo:qo + 4] = -2.0 * dt * wh * Mh
    lxxN = ca.repmat(lxx_c, 1, N)

    def cost_of(Xall, U):
        P = Xall[po:po + pd, :N] - ca.repmat(p_ref, 1, N)
        V = Xall[vo:vo + 3, :N] - ca.repmat(v_ref, 1, N)
        Q = Xall[qo:qo + 4, :N]
        head = ca.sum2(ca.sum1(Q * (Mh @ Q)))       # Σ nose·d over nodes
        e = Xall[:, N] - xref
        return (dt * (ca.sum2(ca.sum1((U * U) * wuN))
                      + ca.sum2(ca.sum1((P * P) * wpN))
                      + ca.sum2(ca.sum1((V * V) * wvN))
                      + wh * (N - head))
                + e.T @ PTm @ e)

    U = plan
    X_best = U_best = J_best = None
    for _ in range(sweeps):
        Xall = ca.horzcat(x0, roll(x0, U))
        _, Aall, Ball = fj_map(Xall[:, :N], U)

        LX = ca.MX.zeros(nx, N)
        LX[po:po + pd, :] = 2.0 * dt * wpN * (
            Xall[po:po + pd, :N] - ca.repmat(p_ref, 1, N))
        LX[vo:vo + 3, :] = 2.0 * dt * wvN * (
            Xall[vo:vo + 3, :N] - ca.repmat(v_ref, 1, N))
        LX[qo:qo + 4, :] = -2.0 * dt * wh * (Mh @ Xall[qo:qo + 4, :N])
        e = Xall[:, N] - xref
        _, _, k_rev, K_rev = bsweep(2.0 * (PTm @ e), 2.0 * PTm,
                                    Aall[:, rev_x], Ball[:, rev_u],
                                    LX[:, ::-1],
                                    (2.0 * dt * wuN * U)[:, ::-1],
                                    lxxN,        # node-constant blocks:
                                    wuN,         # reversal is a no-op
                                    U[:, ::-1])
        ks = k_rev[:, ::-1]
        Ks = K_rev[:, rev_x]

        Xa_all, Ua_all = fsweep_all(x0, U, ks, Ks, Xall[:, :N], al_row)
        J_best = cost_of(Xall, U)
        X_best, U_best = Xall, U
        for a in range(n_al):
            Ua = Ua_all[:, a * N:(a + 1) * N]
            Xa = ca.horzcat(x0, Xa_all[:, a * N:(a + 1) * N])
            Ja = cost_of(Xa, Ua)
            better = Ja < J_best
            X_best = ca.if_else(better, Xa, X_best)
            U_best = ca.if_else(better, Ua, U_best)
            J_best = ca.fmin(Ja, J_best)
        U = U_best

    # The receding-horizon shift, folded into the kernel: apply column
    # 0, warm-start the next tick from the once-shifted remainder.
    plan_next = ca.horzcat(U_best[:, 1:], U_best[:, N - 1])
    return ca.Function("tick", [x0, target, weights, PTm, plan],
                       [plan_next, U_best[:, 0], J_best],
                       ["x", "target", "weights", "PT", "plan"],
                       ["plan_next", "u", "J"])
