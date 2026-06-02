"""LQR — infinite-horizon discrete linear-quadratic regulator.

`LQR(world, *, x_ref, u_ref, Q, R, dt, track)` is an analysis transform
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
with `track=` — e.g. `track=["c.position", "c.velocity"]` for a craft
with 3-axis thrust — and the rest is frozen at `x_ref`. `track=None`
(full state) is for genuinely fully-actuated systems.

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

from typing import Any

import casadi as ca
import numpy as np

from ..linearization import Linearization
from ..sim import Sim
from ..estimation.state_spec import StateSpec
from ..tick_signature import walk_tick_signature


def _solve_dare(A, B, Q, R, *, max_iter: int = 10000, tol: float = 1e-12):
    """Discrete-time algebraic Riccati: backward iteration to a fixpoint.

    Returns `(K, P)` with `K = (R + BᵀPB)⁻¹ BᵀPA`. Converges for a
    stabilizable `(A, B)` with detectable `(A, √Q)`; raises otherwise.
    Dependency-free and run once at construction — not in the loop.
    """
    P = np.array(Q, dtype=float)
    for _ in range(max_iter):
        BtP = B.T @ P
        K   = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_next = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        P_next = 0.5 * (P_next + P_next.T)
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    else:
        raise RuntimeError(
            "LQR: discrete Riccati iteration did not converge — the "
            "linearized (A, B) is likely not stabilizable at this operating "
            "point. Regulate a controllable subset via `track=`, or check "
            "Q/R.")
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P


def _flatten_nested(nested: dict) -> dict:
    """`{owner: {slot: v}}` → `{"owner.slot": v}`; passes flat dicts through."""
    flat: dict[str, Any] = {}
    for owner, slots in nested.items():
        if isinstance(slots, dict):
            for slot, v in slots.items():
                flat[f"{owner}.{slot}"] = v
        else:
            flat[owner] = slots
    return flat


class LQR:
    """Infinite-horizon discrete LQR about an operating point.

    Args:
        world  — the model.
        x_ref  — target state (nested `{owner: {slot: value}}` or flat
                 `{"owner.slot": value}`), merged over the world's
                 initial state for any unspecified slot.
        u_ref  — trim inputs (`{input_name: value}`), merged over each
                 Part Input's default. The equilibrium command.
        Q, R   — LQR cost weights (tracked-tangent², n_inputs²). Default
                 to identity. `R` must be positive-definite.
        dt     — the discrete step the controller will run at.
        track  — slot full-names to regulate (e.g. `["c.position",
                 "c.velocity"]`); the rest are frozen at `x_ref`. `None`
                 regulates the full state (fully-actuated systems only).

    Attributes:
        spec (full), tracked (regulated slot names), input_names,
        K (n_u × tracked_tangent), A, B, x_ref/u_ref (vectors),
        control_fn (`u(x_full)` baked `ca.Function`).
    """

    # Lowerable-block kind (see manta.codegen.block.KIND_LQR): a backend
    # lowers an LQR through its `lower_lqr` handler.
    RUNTIME_KIND = "lqr"

    def __init__(self, world, *,
                 x_ref: dict,
                 u_ref: dict | None = None,
                 Q=None, R=None,
                 dt: float = 0.01,
                 track: list[str] | None = None) -> None:
        if not world.crafts:
            raise ValueError("LQR: world has no crafts.")

        # Reuse Sim(world) for world-prep + the compiled tick.
        self._sim = Sim(world)
        self.world = world
        cf = self._sim.tick.casadi_function
        full_spec = StateSpec.from_world(world)
        self.spec = full_spec

        sig = walk_tick_signature(cf, world, full_spec)
        self.input_names = sig.input_names
        n_u = len(self.input_names)
        if n_u == 0:
            raise ValueError(
                "LQR: world has no Part Inputs — no control authority.")

        # --- operating point (flat, merged over initial state) -------------
        ref_flat = _flatten_nested(world._initial_state_dict())
        ref_flat.update(_flatten_nested(x_ref))
        x_ref_full = full_spec.pack(
            {k: v for k, v in ref_flat.items() if k in full_spec})

        u_full = dict(sig.input_defaults)
        for k, v in (u_ref or {}).items():
            key = k if k in u_full else next(
                (n for n in self.input_names if n.endswith("." + k)), k)
            if key not in u_full:
                raise KeyError(
                    f"LQR: unknown input {k!r}; available "
                    f"{sorted(self.input_names)}")
            u_full[key] = float(v)
        u_ref_vec = np.array([u_full[n] for n in self.input_names], dtype=float)
        self.x_ref, self.u_ref = x_ref_full, u_ref_vec

        # --- regulate full state, or a tracked subset (rest frozen) --------
        if track is None:
            spec = full_spec
            frozen: dict[str, Any] = {}
            self.tracked = [s.name for s in full_spec.slots]
        else:
            all_names = {s.name for s in full_spec.slots}
            unknown = set(track) - all_names
            if unknown:
                raise KeyError(
                    f"LQR: track references unknown slot(s) {sorted(unknown)}; "
                    f"available {sorted(all_names)}.")
            kept = set(track)
            spec = StateSpec.subset(full_spec, kept)
            self.tracked = [s.name for s in full_spec.slots if s.name in kept]
            frozen = {}
            for s in full_spec.slots:
                if s.name not in kept:
                    val = ref_flat.get(s.name, np.zeros(s.dim))
                    frozen[s.name] = np.atleast_1d(
                        np.asarray(val, dtype=float)).reshape(-1)
        self._spec = spec

        lin = Linearization(
            cf, spec, frozen=frozen, input_names=sig.input_names,
            noise_specs=sig.noise, outputs=[], control=True)

        # Operating point in the (sub)spec's ambient layout.
        x_ref_sub = spec.pack(
            {k: v for k, v in ref_flat.items() if k in spec})
        A = np.array(lin.F_fn(x_ref_sub, u_ref_vec, dt, 0.0))
        B = np.array(lin.B_fn(x_ref_sub, u_ref_vec, dt, 0.0))
        self.A, self.B = A, B
        n_x = spec.tangent_dim

        Qm = np.eye(n_x) if Q is None else np.asarray(Q, dtype=float)
        Rm = np.eye(n_u) if R is None else np.asarray(R, dtype=float)
        if Qm.shape != (n_x, n_x):
            raise ValueError(
                f"LQR: Q must be {n_x}×{n_x} (tracked tangent dim), "
                f"got {Qm.shape}.")
        if Rm.shape != (n_u, n_u):
            raise ValueError(
                f"LQR: R must be {n_u}×{n_u} (n_inputs), got {Rm.shape}.")

        self.K, self.P = _solve_dare(A, B, Qm, Rm)

        # --- baked control law: u = u_ref − K·(x_tracked ⊟ x_ref_tracked).
        # Takes the FULL ambient state; gathers the tracked slots from it.
        x_full_sym = ca.MX.sym("x", full_spec.ambient_dim, 1)
        chunks = []
        for s in spec.slots:
            fs = full_spec.slot(s.name)
            chunks.append(x_full_sym[fs.offset : fs.offset + fs.dim])
        x_sub_sym = ca.vertcat(*chunks) if chunks else x_full_sym
        dx = spec.boxminus_sym(x_sub_sym, ca.DM(x_ref_sub.reshape(-1, 1)))
        u_expr = ca.DM(u_ref_vec.reshape(-1, 1)) - ca.DM(self.K) @ dx
        self.control_fn = ca.Function(
            "lqr_u", [x_full_sym], [u_expr], ["x"], ["u"])

    @property
    def closed_loop_eigs(self) -> np.ndarray:
        """Eigenvalues of the closed-loop tangent map `A − B·K` (over the
        tracked subspace). All inside the unit circle ⇒ stable."""
        return np.linalg.eigvals(self.A - self.B @ self.K)

    def __repr__(self) -> str:
        return (f"<LQR n_x={self._spec.tangent_dim} n_u={len(self.input_names)} "
                f"tracked={self.tracked} inputs={self.input_names}>")
