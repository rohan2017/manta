"""LQR — infinite-horizon discrete linear-quadratic regulator.

`LQR(world, *, x_ref, u_ref, Q, R, dt, regulate)` is an analysis transform
over the model, a sibling of `Sim(world)` / `EKF(world)`. It:

  1. linearizes the world tick at the supplied equilibrium `(x_ref,
     u_ref)` via the shared `Linearization` seam — `A = F = ∂f/∂δx`,
     `B = ∂f/∂u`, both in the error-state tangent;
  2. solves the discrete-time algebraic Riccati equation offline (a
     dependency-free backward iteration to convergence) for the optimal
     gain `K`; and
  3. bakes a manifold-correct control law into a CasADi function

         u = u_ref − K · (x ⊟ x_ref)

Lower with `TargetNumpy(LQR(...))` to a runtime controller.

**Controllability.** A free rigid body is underactuated: the full 12-D
rigid-body state is not stabilizable for any realistic actuator set (the
uncontrolled position / attitude integrators sit on the unit circle, so
the Riccati iteration won't converge). Regulate the *controllable* subset
with `regulate=` — e.g. `regulate=["c.position", "c.velocity"]` for a
craft with 3-axis thrust — and the rest is frozen at `x_ref`.
`regulate=None` (full state) is for genuinely fully-actuated systems.
Unlike the EKF's `track` (a lower bound that is *closed* over the
dynamics), `regulate` is taken verbatim: freezing the uncontrollable
states at the operating point is the whole point — it's what makes the
reduced system stabilizable. So `Q` is sized to exactly the slots you
list (their summed tangent dim).

`Q` (tracked-tangent²) and `R` (n_inputs²) are the LQR **cost** weights —
not the EKF's process/measurement noise (same letters, different role).
The operating point is *yours*: solving for the trim `u*` that makes
`f(x*, u*) = x*` is a root-solve and stays out of the IR; for a hover
that trim is just `u_ref = m·g`.

The Riccati iteration runs once, at construction (offline) — the
deployable artifact is the feed-forward control law, no in-the-loop
solve. `K` depends on `dt` (through the discrete `A`, `B`), so build the
LQR at the rate you intend to run the controller.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role, StateLayout,
)
from ..linearization import LinearizedSystem, resolve_suffix


def _solve_dare(A, B, Q, R, *, max_iter: int = 10000, tol: float = 1e-12):
    """Discrete-time algebraic Riccati: backward iteration to a fixpoint.

    Returns `(K, P)` with `K = (R + BᵀPB)⁻¹ BᵀPA`. Converges for a
    stabilizable `(A, B)` with detectable `(A, √Q)`; raises otherwise.
    `tol` is RELATIVE — `‖ΔP‖ ≤ tol·max(1, ‖P‖)` — so huge cost scales
    don't read as "not stabilizable". Dependency-free and run once at
    construction — not in the loop.
    """
    P = np.array(Q, dtype=float)
    for _ in range(max_iter):
        BtP = B.T @ P
        K   = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_next = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        P_next = 0.5 * (P_next + P_next.T)
        if np.linalg.norm(P_next - P) <= tol * max(1.0,
                                                   np.linalg.norm(P_next)):
            P = P_next
            break
        P = P_next
    else:
        raise RuntimeError(
            "LQR: discrete Riccati iteration did not converge — the "
            "linearized (A, B) is likely not stabilizable at this operating "
            "point. Regulate a controllable subset via `regulate=`, or "
            "check Q/R.")
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P


class LQR:
    """Infinite-horizon discrete LQR about an operating point.

    Args:
        world  — the model.
        x_ref  — target state (nested `{owner: {slot: value}}` or flat
                 `{"owner.slot": value}`), merged over the world's
                 initial state for any unspecified slot.
        u_ref  — trim inputs (`{input_name: value}`), merged over each
                 Part Input's default. The equilibrium command.
        Q, R   — LQR cost weights (regulated-tangent², n_inputs²). Default
                 to identity. `R` must be positive-definite.
        dt     — the discrete step the controller will run at.
        regulate — slot full-names to regulate, taken verbatim (e.g.
                 `["c.position", "c.velocity"]`); the rest are frozen at
                 `x_ref`. `None` regulates the full state (fully-actuated
                 systems only).
        tol, max_iter — Riccati-iteration convergence: relative fixpoint
                 tolerance (`‖ΔP‖ ≤ tol·max(1, ‖P‖)`) and iteration cap.

    Attributes:
        spec (full), regulated (regulated slot names), input_names,
        K (n_u × tracked_tangent), A, B, x_ref/u_ref (vectors),
        control_fn (`u(x_full, x_ref_full)` baked `ca.Function`;
        runtimes default `x_ref` to the built reference — see
        `NumpyRegulator.retarget`).
    """

    def __init__(self, world, *,
                 x_ref: dict,
                 u_ref: dict | None = None,
                 Q=None, R=None,
                 dt: float = 0.01,
                 regulate: list[str] | None = None,
                 tol: float = 1e-12,
                 max_iter: int = 10000) -> None:
        if not world.crafts:
            raise ValueError("LQR: world has no crafts.")

        # All the linearization plumbing — tick compile, signature, the
        # VERBATIM regulated subset frozen at the operating point, and
        # B = ∂f/∂u — lives in `LinearizedSystem`. `regulate` is taken
        # verbatim (NOT closed over the dynamics like the EKF's `track`):
        # for an underactuated craft the whole point is to freeze the
        # uncontrollable states (e.g. attitude) at the operating point so
        # the reduced system is stabilizable; closing the set would pull
        # them back and the Riccati solve would diverge. The reference
        # point doubles as the freeze value.
        sys = LinearizedSystem(world, track=regulate, inputs=None,
                               track_mode="verbatim", control=True,
                               ref=x_ref)
        self.sys     = sys
        self.world   = world
        self.spec    = sys.full_spec      # full layout (the law gathers from it)
        self._spec   = sys.spec           # tracked subspec
        self.regulated = sys.tracked
        self.input_names = sys.input_names
        n_u = len(self.input_names)
        if n_u == 0:
            raise ValueError(
                "LQR: world has no Part Inputs — no control authority.")

        # --- operating point ----------------------------------------------
        u_full = dict(sys.input_defaults)
        for k, v in (u_ref or {}).items():
            full = resolve_suffix(k, self.input_names, label="input", who="LQR")
            u_full[full] = float(v)
        u_ref_vec = np.array([u_full[n] for n in self.input_names], dtype=float)
        x_ref_full = sys.pack_ref(sys.full_spec)
        self.x_ref, self.u_ref = x_ref_full, u_ref_vec

        # A = F, B = ∂f/∂u, both at the operating point (subspec ambient).
        x_ref_sub = sys.pack_ref(sys.spec)
        A = np.array(sys.F_fn(x_ref_sub, u_ref_vec, dt, 0.0))
        B = np.array(sys.B_fn(x_ref_sub, u_ref_vec, dt, 0.0))
        self.A, self.B = A, B
        n_x = sys.spec.tangent_dim

        Qm = np.eye(n_x) if Q is None else np.asarray(Q, dtype=float)
        Rm = np.eye(n_u) if R is None else np.asarray(R, dtype=float)
        if Qm.shape != (n_x, n_x):
            raise ValueError(
                f"LQR: Q must be {n_x}×{n_x} (tracked tangent dim), "
                f"got {Qm.shape}.")
        if Rm.shape != (n_u, n_u):
            raise ValueError(
                f"LQR: R must be {n_u}×{n_u} (n_inputs), got {Rm.shape}.")
        if not np.allclose(Rm, Rm.T):
            raise ValueError("LQR: R must be symmetric.")
        if np.min(np.linalg.eigvalsh(Rm)) <= 0.0:
            raise ValueError(
                "LQR: R must be positive-definite (its min eigenvalue is "
                f"{np.min(np.linalg.eigvalsh(Rm)):.3g}) — every input "
                "needs a positive cost.")

        self.K, self.P = _solve_dare(A, B, Qm, Rm, tol=tol,
                                     max_iter=max_iter)

        # --- baked control law: u = u_ref − K·(x_tracked ⊟ x_ref_tracked).
        # Takes the FULL ambient state AND the reference as arguments;
        # gathers the tracked slots from each. The gain K is the baked
        # constant — feeding a moved `x_ref` retargets the regulator
        # through the same law with NO re-solve, which is exact wherever
        # the dynamics are invariant along the moved direction (e.g.
        # translating a hover setpoint under uniform gravity). A genuinely
        # different operating point (new A/B or trim) needs a new LQR.
        full_spec, spec = sys.full_spec, sys.spec
        x_full_sym = ca.MX.sym("x", full_spec.ambient_dim, 1)
        x_ref_sym = ca.MX.sym("x_ref", full_spec.ambient_dim, 1)

        def _gather(sym):
            chunks = []
            for s in spec.slots:
                fs = full_spec.slot(s.name)
                chunks.append(
                    sym[fs.ambient_offset : fs.ambient_offset + fs.ambient_dim])
            return ca.vertcat(*chunks) if chunks else sym

        dx = spec.boxminus_sym(_gather(x_full_sym), _gather(x_ref_sym))
        u_expr = ca.DM(u_ref_vec.reshape(-1, 1)) - ca.DM(self.K) @ dx
        self.control_fn = ca.Function(
            "lqr_u", [x_full_sym, x_ref_sym], [u_expr], ["x", "x_ref"], ["u"])

        # --- the typed Module: stateless, one control(x, x_ref) -> u entry.
        # Both STATE ports carry the operating point as `init`, so a
        # backend can default x_ref to the built reference.
        self._module = Module(
            name=f"{world.name}_lqr", state=StateLayout(()),
            ports=(
                Port("x", Role.STATE, (full_spec.ambient_dim,),
                     manifold=full_spec, init=x_ref_full),
                Port("x_ref", Role.STATE, (full_spec.ambient_dim,),
                     manifold=full_spec, init=x_ref_full),
                Port("u", Role.CONTROL, (n_u,), fields=tuple(
                    PortField(n, 1, float(u_full[n]))
                    for n in self.input_names)),
            ),
            functions={"control": self.control_fn},
            entry_points=(EntryPoint("control", "control",
                                     (PortRef("x"), PortRef("x_ref")),
                                     returns=("u",)),),
            hosting=Hosting.THREADED)

    def module(self) -> Module:
        """The typed `Module` IR a backend lowers."""
        return self._module

    @property
    def closed_loop_eigs(self) -> np.ndarray:
        """Eigenvalues of the closed-loop tangent map `A − B·K` (over the
        tracked subspace). All inside the unit circle ⇒ stable."""
        return np.linalg.eigvals(self.A - self.B @ self.K)

    def __repr__(self) -> str:
        return (f"<LQR n_x={self._spec.tangent_dim} n_u={len(self.input_names)} "
                f"regulated={self.regulated} inputs={self.input_names}>")
