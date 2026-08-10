"""MPC — receding-horizon point-to-point control over the compiled tick.

`MPC(world, u_bounds=..., ...)` is a controller-synthesis transform, a
sibling of `LQR`: it consumes the model and emits a typed `Module`
whose one entry point is the fixed-work real-time-iteration tick

    tick(x, goal, plan) → (plan', u, J)

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

Cost family: quadratic point-to-point with FREE attitude — running
position error + effort, terminal position/velocity. Terminal weights
are either explicit (`w_pos_terminal`/`w_vel_terminal`) or, with
`terminal="lqr"`, the Riccati cost-to-go at the goal equilibrium
(P from this package's own `LQR`, with Q/R restating the running cost
— the true infinite-horizon tail of the same objective, which is what
lets the horizon shrink). `terminal="lqr"` requires a stabilizable
rest equilibrium; an underactuated hull with none (a forward-only
prop with fins dead at rest) gets a clear Riccati failure, and keeps
explicit weights.

Plan birth: the tick bootstraps from the zero plan for hulls that can
act from rest; for underactuated hulls or basin-sensitive transits,
seed the HELD plan from an offline solve (`NumpyMpc.reset_plan`) —
plan birth is a mission-load event, the tick is the flight loop.

Coefficients are BAKED (doctrine 2026-08-09): feedback absorbs small
model drift (a +10 % module swap flies on the nominal controller);
changes big enough to matter are a new controller and recompile.
"""

from __future__ import annotations

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
    ports `x` (STATE, full manifold), `goal` (3-vector), `u` (CONTROL,
    one field per Part Input), `J` (OUTPUT, the plan cost); one entry
    `tick(x, goal)` that writes the shifted plan and returns (u, J).
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
        po, pd = pos.ambient_offset, pos.ambient_dim
        vo = vel.ambient_offset

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
            po=po, pd=pd, PT=PT, u_lo=u_lo, u_hi=u_hi,
            w_pos=float(w_pos_running), w_u=float(w_u),
            sweeps=int(sweeps), alphas=tuple(alphas), mu=float(mu),
            parallel=parallel)

        self._module = Module(
            name=f"{world.name}_mpc",
            state=StateLayout((StateField(
                "plan", kind="matrix", shape=(nu, self.horizon),
                init=np.zeros((nu, self.horizon))),)),
            ports=(
                Port("x", Role.STATE, (spec.ambient_dim,),
                     manifold=spec, init=x_init),
                Port("goal", Role.MATRIX, (3, 1),
                     init=x_init[po:po + pd].reshape(3, 1)),
                Port("u", Role.CONTROL, (nu,), fields=tuple(
                    PortField(f.name, 1, float(f.default))
                    for f in u_fields)),
                Port("J", Role.OUTPUT, (1,)),
            ),
            functions={"tick": tick},
            entry_points=(EntryPoint(
                "tick", "tick",
                (PortRef("x"), PortRef("goal"), StateRef("plan")),
                writes=("plan",), returns=("u", "J")),),
            hosting=Hosting.HELD)

    def module(self) -> Module:
        """The typed `Module` IR a backend lowers."""
        return self._module

    def __repr__(self) -> str:
        return (f"<MPC N={self.horizon} dt={self.dt} nx={self.nx} "
                f"nu={self.nu} inputs={list(self.input_names)}>")


def _build_tick(f: ca.Function, fj: ca.Function, *, nx: int, nu: int,
                N: int, dt: float, po: int, pd: int, PT: np.ndarray,
                u_lo: np.ndarray, u_hi: np.ndarray, w_pos: float,
                w_u: float, sweeps: int, alphas: tuple[float, ...],
                mu: float, parallel: str | None) -> ca.Function:
    """The loop-structured RTI tick: (x0, goal, plan) → (plan', u0, J).

    Structure is the manta-mpc lab's `rti._build_tick`, with the shift
    folded in: `mapaccum` for the scans (rollouts, the backward Riccati
    sweep), `map` for the per-node linearization and all α rollouts —
    so generated C is loops over one emitted body per kernel, constant
    in N and ladder size. The backward step is one SX function whose
    small solves expand to scalar ops (Linsol-free, the same discipline
    that keeps the world tick expandable); box limits enter as a
    one-shot masked active set, honest for warm ticks where the active
    set is stable between solves.
    """
    PT_dm = ca.DM(PT)
    lo_dm, hi_dm = ca.DM(u_lo), ca.DM(u_hi)

    # ---- backward step (accumulators Vx, Vxx first: mapaccum) ---------
    A = ca.SX.sym("A", nx, nx)
    B = ca.SX.sym("B", nx, nu)
    lx = ca.SX.sym("lx", nx)
    lu = ca.SX.sym("lu", nu)
    ubar = ca.SX.sym("ubar", nu)
    Vx = ca.SX.sym("Vx", nx)
    Vxx = ca.SX.sym("Vxx", nx, nx)
    luu = 2.0 * dt * w_u

    Qx = lx + A.T @ Vx
    Qu = lu + B.T @ Vx
    VxxA = Vxx @ A
    lxx = ca.SX.zeros(nx, nx)
    for j in range(pd):
        lxx[po + j, po + j] = 2.0 * dt * w_pos
    Qxx = lxx + A.T @ VxxA
    Qux = B.T @ VxxA
    Quu = luu * ca.SX.eye(nu) + B.T @ Vxx @ B
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
    bstep = ca.Function("bstep", [Vx, Vxx, A, B, lx, lu, ubar],
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
    goal = ca.MX.sym("goal", 3)
    plan = ca.MX.sym("plan", nu, N)

    xref = ca.MX.zeros(nx)
    xref[po:po + pd] = goal

    def cost_of(Xall, U):
        P = Xall[po:po + pd, :N] - ca.repmat(goal, 1, N)
        e = Xall[:, N] - xref
        return (dt * (w_u * ca.dot(U, U) + w_pos * ca.dot(P, P))
                + e.T @ PT_dm @ e)

    U = plan
    X_best = U_best = J_best = None
    for _ in range(sweeps):
        Xall = ca.horzcat(x0, roll(x0, U))
        _, Aall, Ball = fj_map(Xall[:, :N], U)

        LX = ca.MX.zeros(nx, N)
        LX[po:po + pd, :] = 2.0 * dt * w_pos * (
            Xall[po:po + pd, :N] - ca.repmat(goal, 1, N))
        e = Xall[:, N] - xref
        _, _, k_rev, K_rev = bsweep(2.0 * (PT_dm @ e), 2.0 * PT_dm,
                                    Aall[:, rev_x], Ball[:, rev_u],
                                    LX[:, ::-1],
                                    (2.0 * dt * w_u * U)[:, ::-1],
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
    return ca.Function("tick", [x0, goal, plan],
                       [plan_next, U_best[:, 0], J_best],
                       ["x", "goal", "plan"], ["plan_next", "u", "J"])
