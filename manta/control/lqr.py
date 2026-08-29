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

**Moving the operating point.** `K` and the trim are *data*, not baked
constants: the law's `K` / `u_ff` Ports carry the built solve as their
default, and any runtime may be handed a different one. `resolve_at`
re-evaluates `A`, `B` at a new reference and re-runs the Riccati solve
(µs + ms — the symbolic linearization is already compiled), returning an
`LQRSolution` a regulator installs with `reprogram()`. That is the
general answer to a moved setpoint; a bare `retarget()` keeps the old
gain and is only exact where the dynamics are invariant along the move
(a pure translation under uniform gravity — NOT a heading change, whose
world-frame position feedback comes out rotated with it).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import casadi as ca
import numpy as np

from ..ir._names import resolve_suffix
from ..ir.module import (
    EntryPoint,
    Hosting,
    Module,
    ModuleKind,
    Port,
    PortField,
    PortRef,
    Role,
    StateLayout,
)
from ..ir.state_spec import flatten_nested
from ..linearization import LinearizedSystem


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


def _same(a, b) -> bool:
    """Do two reference-slot values agree? (Either may be absent, a
    scalar, a tuple, or an array.)"""
    if a is None:
        return False
    av = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    bv = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    return av.shape == bv.shape and bool(np.allclose(av, bv))


@dataclass(frozen=True)
class LQRSolution:
    """One Riccati solve at one operating point, as plain data.

    The affine control law is `u = u_ff − K·(x ⊟ x_ref)`; these three
    fields are the whole of it. `LQR.resolve_at` returns one and a
    runtime regulator's `reprogram()` installs it — all three together,
    because a gain is only valid about the point it was solved at.

    Everything here is a plain array, so a retarget service can hand a
    compiled regulator (numpy, wasm, C++) a new setpoint over JSON with
    no CasADi on the other side.

    Attrs:
        K      — n_u × regulated-tangent feedback gain.
        u_ff   — n_u feed-forward: the trim command at this point.
        x_ref  — full ambient reference the law regulates to.
        A, B   — the tangent linearization it was solved from.
        P      — the Riccati fixpoint.
    """
    K:     np.ndarray
    u_ff:  np.ndarray
    x_ref: np.ndarray
    A:     np.ndarray
    B:     np.ndarray
    P:     np.ndarray
    controller_id: str

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("K", "u_ff", "x_ref", "A", "B", "P"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "iuf":
                raise TypeError(f"LQRSolution.{name} must be real numeric data")
            value = np.asarray(raw, dtype=float).copy()
            if not np.all(np.isfinite(value)):
                raise ValueError(f"LQRSolution.{name} must be finite")
            value.flags.writeable = False
            arrays[name] = value
            object.__setattr__(self, name, value)
        K, u_ff, x_ref, A, B, P = (arrays[n] for n in
                                    ("K", "u_ff", "x_ref", "A", "B", "P"))
        if K.ndim != 2 or A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("LQRSolution: K and square A matrices are required")
        if B.shape != (A.shape[0], K.shape[0]) or K.shape[1] != A.shape[0]:
            raise ValueError("LQRSolution: A, B, and K dimensions are inconsistent")
        if P.shape != A.shape or u_ff.size != K.shape[0] or x_ref.ndim != 1:
            raise ValueError("LQRSolution: P, u_ff, or x_ref dimensions are inconsistent")
        if not isinstance(self.controller_id, str) or not self.controller_id:
            raise ValueError("LQRSolution.controller_id is required")

    @property
    def closed_loop_eigs(self) -> np.ndarray:
        """Eigenvalues of `A − B·K`. All inside the unit circle ⇒ stable."""
        return np.linalg.eigvals(self.A - self.B @ self.K)

    def __repr__(self) -> str:
        return (f"<LQRSolution K{self.K.shape} "
                f"rho={np.max(np.abs(self.closed_loop_eigs)):.4f}>")


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
        K (n_u × tracked_tangent), A, B, P, Q, R, dt,
        x_ref/u_ref (vectors), solution (the built solve as data),
        control_fn (`u(x_full, x_ref_full, K, u_ff)` `ca.Function`;
        runtimes default every argument but the live state to the built
        operating point — see `NumpyRegulator.retarget` / `reprogram`).
    """

    def __init__(self, world, *,
                 x_ref: dict,
                 u_ref: dict | None = None,
                 Q=None, R=None,
                 dt: float = 0.01,
                 regulate: list[str] | None = None,
                 tol: float = 1e-12,
                 max_iter: int = 10000) -> None:
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
        self.world   = sys.world
        self.model   = sys.model
        if not self.world.crafts:
            raise ValueError("LQR: world has no crafts.")
        self.spec    = sys.full_spec      # full layout (the law gathers from it)
        self._spec   = sys.spec           # tracked subspec
        self.regulated = sys.tracked
        self.input_names = sys.input_names
        n_u = len(self.input_names)
        if n_u == 0:
            raise ValueError(
                "LQR: world has no Part Inputs — no control authority.")

        # --- operating point ----------------------------------------------
        self._u_full = self._merge_u(u_ref, base=sys.input_defaults, who="LQR")
        u_ref_vec = self._u_vector(self._u_full)
        self.x_ref, self.u_ref = sys.pack_ref(sys.full_spec), u_ref_vec
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("LQR.dt must be finite and positive")
        n_x = sys.spec.tangent_dim

        identity = sha256()
        identity.update(sys._cf.serialize().encode())
        identity.update(repr((tuple(sys.tracked), tuple(sys.input_names),
                              self.dt)).encode())
        self.controller_id = identity.hexdigest()

        self.Q = self._check_Q(np.eye(n_x) if Q is None else Q)
        self.R = self._check_R(np.eye(n_u) if R is None else R)
        self._tol, self._max_iter = tol, max_iter

        # A = F, B = ∂f/∂u, both at the operating point (subspec ambient).
        sol = self._solve(sys.ref_flat, u_ref_vec, self.Q, self.R)
        self.A, self.B = sol.A, sol.B
        self.K, self.P = sol.K, sol.P

        # --- the control law: u = u_ff − K·(x_tracked ⊟ x_ref_tracked).
        # Takes the FULL ambient state and the reference, gathering the
        # tracked slots from each — plus the gain and the feed-forward,
        # which are runtime DATA rather than baked constants. Every
        # argument defaults to the built solve, so a caller that ignores
        # them flies exactly the law this construction solved; handing
        # over a fresh `resolve_at` triple is what makes a genuinely new
        # operating point (new A/B or trim) reachable without rebuilding
        # anything. Moving `x_ref` alone keeps the old gain — exact only
        # where the dynamics are invariant along the move.
        full_spec, spec = sys.full_spec, sys.spec
        x_full_sym = ca.MX.sym("x", full_spec.ambient_dim, 1)
        x_ref_sym = ca.MX.sym("x_ref", full_spec.ambient_dim, 1)
        K_sym = ca.MX.sym("K", n_u, n_x)
        u_ff_sym = ca.MX.sym("u_ff", n_u, 1)

        def _gather(sym):
            chunks = []
            for s in spec.slots:
                fs = full_spec.slot(s.name)
                chunks.append(
                    sym[fs.ambient_offset : fs.ambient_offset + fs.ambient_dim])
            return ca.vertcat(*chunks) if chunks else sym

        dx = spec.boxminus_sym(_gather(x_full_sym), _gather(x_ref_sym))
        u_expr = u_ff_sym - K_sym @ dx
        self.control_fn = ca.Function(
            "lqr_u", [x_full_sym, x_ref_sym, K_sym, u_ff_sym], [u_expr],
            ["x", "x_ref", "K", "u_ff"], ["u"])

        # --- the typed Module: stateless, one
        # control(x, x_ref, K, u_ff) -> u entry. Every port but the live
        # `x` carries the built operating point as `init`, so a backend
        # defaults the reference, the gain, and the trim to this solve.
        self._module = Module(
            name=f"{world.name}_lqr", state=StateLayout(()),
            ports=(
                Port("x", Role.STATE, (full_spec.ambient_dim,),
                     spec=full_spec, init=self.x_ref),
                Port("x_ref", Role.STATE, (full_spec.ambient_dim,),
                     spec=full_spec, init=self.x_ref),
                Port("K", Role.MATRIX, (n_u, n_x), init=self.K),
                Port("u_ff", Role.MATRIX, (n_u, 1),
                     init=u_ref_vec.reshape(-1, 1)),
                Port("u", Role.CONTROL, (n_u,), fields=tuple(
                    PortField(n, 1, float(self._u_full[n]))
                    for n in self.input_names)),
            ),
            functions={"control": self.control_fn},
            entry_points=(EntryPoint("control", "control",
                                     (PortRef("x"), PortRef("x_ref"),
                                      PortRef("K"), PortRef("u_ff")),
                                     returns=("u",)),),
            kind=ModuleKind.REGULATOR,
            hosting=Hosting.THREADED,
            metadata={
                **sys.model.transform_metadata({
                    "transform": "lqr",
                    "discretization": sys.discretization,
                    "tracked": tuple(sys.tracked),
                    "inputs": tuple(sys.input_names),
                    "dt": self.dt,
                }),
                "controller_id": self.controller_id,
            })

    def module(self) -> Module:
        """The typed `Module` IR a backend lowers."""
        return self._module

    # ------------------------------------------------------------------
    # Re-solving at a new operating point
    # ------------------------------------------------------------------

    def resolve_at(self, *, x_ref: dict | None = None,
                   u_ref: dict | None = None,
                   Q=None, R=None) -> LQRSolution:
        """Re-solve the gain about a NEW operating point.

        Evaluates `A`, `B` at the moved reference and re-runs the Riccati
        iteration — the symbolic linearization is already compiled, so
        this is a matrix evaluation plus a small dense DARE (µs + ms on a
        ~12-dim tangent), not a rebuild. Returns an `LQRSolution`;
        install it on a live regulator with `reprogram()`, or ship it as
        data. `self` is untouched.

        This is the correct way to move a setpoint whenever the dynamics
        are *not* invariant along the move — most importantly a heading
        change, where `retarget()` alone leaves the world-frame position
        feedback rotated with the reference (⊥ at 90°, positive feedback
        at 180°).

        Args:
            x_ref — reference overrides (nested or flat), merged over the
                    built reference. Every named slot must be one this
                    LQR **regulates**: the complement is frozen at the
                    built point and baked into `A`/`B` as a constant, so
                    no re-evaluation can honour a move there.
            u_ref — trim overrides, merged over the built trim. The
                    equilibrium command at the new point (attitude-
                    dependent in general — solving for it is a root-solve
                    and stays yours).
            Q, R  — cost overrides; default to the built weights.

        Raises:
            ValueError — a named slot is unregulated (frozen) or unknown,
                    or the moved point is not stabilizable.
        """
        sys = self.sys
        ref_flat = dict(sys.ref_flat)
        if x_ref is not None:
            moved = flatten_nested(x_ref)
            self._check_movable(moved)
            ref_flat.update(moved)
        u_vec = self._u_vector(
            self._merge_u(u_ref, base=self._u_full, who="LQR.resolve_at"))
        return self._solve(ref_flat,
                           u_vec,
                           self.Q if Q is None else self._check_Q(Q),
                           self.R if R is None else self._check_R(R))

    def _solve(self, ref_flat: dict, u_vec: np.ndarray, Qm, Rm) -> LQRSolution:
        """Linearize at `(ref_flat, u_vec)` and solve the DARE there."""
        sys = self.sys
        x_sub = sys.spec.pack_projected(ref_flat)
        A = np.array(sys.F_fn(x_sub, u_vec, self.dt, 0.0))
        B = np.array(sys.B_fn(x_sub, u_vec, self.dt, 0.0))
        K, P = _solve_dare(A, B, Qm, Rm, tol=self._tol,
                           max_iter=self._max_iter)
        return LQRSolution(K=K, u_ff=u_vec,
                           x_ref=sys.full_spec.pack_projected(ref_flat),
                           A=A, B=B, P=P,
                           controller_id=self.controller_id)

    def _check_movable(self, moved: dict) -> None:
        """Reject a reference move this LQR cannot honour.

        The unregulated complement is FROZEN at the built operating point
        and baked into `F`/`B` as a `ca.DM` constant (see
        `linearization.engine._tick_outputs`), so re-evaluating them at a
        new reference would silently ignore the move. An unknown slot is
        refused for the same reason `pack_any` would swallow it: a typo'd
        retarget must not look like a no-op."""
        known = {s.name for s in self.spec.slots}
        regulated = set(self.regulated)
        unknown = sorted(n for n in moved if n not in known)
        if unknown:
            raise ValueError(
                f"LQR.resolve_at: unknown state slot(s) {unknown}; this "
                f"world has {sorted(known)}.")
        # Compare against `sys.frozen` — the values actually baked into
        # F/B — not the reference dict they were derived from.
        frozen = sorted(
            n for n, v in moved.items()
            if n not in regulated and not _same(self.sys.frozen.get(n), v))
        if frozen:
            raise ValueError(
                f"LQR.resolve_at: slot(s) {frozen} are not regulated by "
                f"this LQR (it regulates {self.regulated}) — they are "
                f"frozen at the built operating point and baked into A/B, "
                f"so the re-solve cannot honour a move there. Build a new "
                f"LQR at that point, or add them to `regulate=`.")

    # ---- operating-point plumbing (shared by __init__ and resolve_at) ---

    def _merge_u(self, u_ref: dict | None, *, base, who: str) -> dict:
        """Trim overrides (full or suffix input names) merged over `base`."""
        u_full = dict(base)
        for k, v in (u_ref or {}).items():
            full = resolve_suffix(k, self.input_names, label="input", who=who)
            u_full[full] = float(v)
        return u_full

    def _u_vector(self, u_full: dict) -> np.ndarray:
        return np.array([u_full[n] for n in self.input_names], dtype=float)

    def _check_Q(self, Q) -> np.ndarray:
        n_x = self._spec.tangent_dim
        Qm = np.asarray(Q, dtype=float)
        if Qm.shape != (n_x, n_x):
            raise ValueError(
                f"LQR: Q must be {n_x}×{n_x} (tracked tangent dim), "
                f"got {Qm.shape}.")
        return Qm

    def _check_R(self, R) -> np.ndarray:
        n_u = len(self.input_names)
        Rm = np.asarray(R, dtype=float)
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
        return Rm

    @property
    def solution(self) -> LQRSolution:
        """The built solve as data — what every Port defaults to, and the
        identity element for `reprogram()`."""
        return LQRSolution(K=self.K, u_ff=self.u_ref, x_ref=self.x_ref,
                           A=self.A, B=self.B, P=self.P,
                           controller_id=self.controller_id)

    @property
    def closed_loop_eigs(self) -> np.ndarray:
        """Eigenvalues of the closed-loop tangent map `A − B·K` (over the
        tracked subspace). All inside the unit circle ⇒ stable."""
        return np.linalg.eigvals(self.A - self.B @ self.K)

    def __repr__(self) -> str:
        return (f"<LQR n_x={self._spec.tangent_dim} n_u={len(self.input_names)} "
                f"regulated={self.regulated} inputs={self.input_names}>")
