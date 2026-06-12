"""Joint-space block dynamics — the coupled body ⊕ joint solve.

For a craft with K articulated DOFs q (revolute angles, prismatic
displacements), the rotational + joint dynamics form ONE linear system

    A(q) · [α; q̈] = [τ_com; Q_q] − bias(q, ω, q̇)

with the generalized mass matrix

    A = [ I_b(q)   J(q) ]        I_b = aggregate inertia about the COM
        [ J(q)ᵀ   M(q) ]         J   = body↔joint momentum coupling
                                 M   = joint-space mass matrix

assembled symbolically as the Hessian of the craft's rotational kinetic
energy in COM coordinates. Working about the COM decouples the LINEAR
dynamics exactly (m_total·a_com = F_ext is a theorem — no linear rows
needed), while the joint rows pick up every coupling the old rank-1
per-rotor solve approximated or dropped: nested-gimbal off-diagonal
inertia, configuration-dependent generalized inertia, prismatic lever
arms, and the mount's linear-acceleration feedback (a free-falling
pendulum no longer "swings": uniform gravity cancels exactly in the
generalized force).

Everything is DERIVED, not hand-assembled, so there are no per-joint-
type dynamics formulas to maintain:

  * T(q, ω, q̇) is built from the kinematic pass's per-part velocities
    (already linear in the joint-rate symbols) — ½Σ mᵢ|ω×ρᵢ + νᵢ|² +
    ½Σ ωᵢᵀIᵢωᵢ with ρᵢ = rᵢ − r_com, νᵢ = v_rel,ᵢ − v_com_rel.
  * p = ∂T/∂u (u = [ω; q̇]) is the generalized momentum; A = ∂p/∂u.
  * The bias follows the Boltzmann–Hamel form (ω is a body-frame
    quasi-velocity, q are true coordinates):
        ω-row:  ṗ_ω + ω×p_ω = τ_com      (gyroscopic couples emerge here)
        q-row:  ṗ_q − ∂T/∂q = Q_q        (Coriolis joint torques emerge here)
    with ṗ expanded as A·u̇ + (∂p/∂q)·q̇.
  * Generalized forces by virtual work over the per-part wrenches:
        Q += (∂vᵢ/∂u)ᵀ·Fᵢ + (∂ωᵢ/∂u)ᵀ·τᵢ
    The first three rows reproduce τ_com = Σ ρᵢ×Fᵢ + τᵢ identically; the
    joint rows are the exact motion-subspace projections (including the
    −F_tot·β̄ recoil term that cancels uniform gravity in free fall).
    Joint actuator/damping efforts are internal exchanges and enter
    ONLY their own row (their body reaction lives in A's coupling).

Joints whose generalized inertia is numerically zero (a massless rotor)
are locked (q̈ = 0) and excluded from the block — matching the legacy
behaviour. A craft with no (unlocked) joints returns None and the
caller keeps the plain 3×3 body solve, bit-identical to the flat path.

Degeneracy: A is PSD by construction but can be near-singular when the
body's inertia about a joint axis is entirely the rotor's (a "massless
stator" — e.g. a spinning top whose pole carries almost no inertia: the
body's axial rate becomes a gauge DOF). Detected numerically at the
zero pose; the fix mirrors the legacy cap — a tiny compile-time inflation
of the body block, leaving the joint momentum rows exact.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

# Below this, an inertia evaluated numerically at the zero pose (a joint's
# generalized inertia here; det of the 3×3 body inertia in world_tick) is
# treated as exactly zero — a massless rotor / point-mass craft — and the
# corresponding rotational solve is locked/skipped instead of inverted.
# Far below any physical value in SI (a 1 g, 1 mm bolt is ~1e-9 kg·m²) yet
# far above accumulated round-off from the symbolic assembly.
DEGENERATE_INERTIA_EPS = 1e-18


def _generalized_inertia(joint) -> float:
    """Numeric scalar deciding whether the DOF is dynamically live:
    rest-pose axial MOI for a revolute, subtree mass for a prismatic."""
    from ..parts.articulation.joint import PrismaticJoint
    if isinstance(joint, PrismaticJoint):
        return joint.subtree_mass
    return joint.I_axial


def build_joint_space(craft, *,
                      kin_states: dict,
                      inertia: dict,
                      state_input_nodes: dict,
                      own_wrench: dict,
                      om_mx,
                      v_com_rel_mx,
                      joint_accel_syms: dict) -> dict | None:
    """Assemble the symbolic joint-space blocks for one craft.

    Args:
        kin_states        — {part: KinematicState} from the kinematic pass.
        inertia           — symbolic_inertia_rollup() result.
        state_input_nodes — {part: {state_name: ir symbol}}.
        own_wrench        — {part: Wrench} per-part wrenches in craft
                            coords about each part's own origin (NOT
                            lifted) — coupling wrenches excluded.
        om_mx             — the body angular-velocity input symbol (3,1).
        v_com_rel_mx      — mass-weighted body-relative COM velocity.
        joint_accel_syms  — {joint: MX} DOF-acceleration placeholders.

    Returns None when the craft has no live articulated DOF; otherwise a
    dict with (all raw MX unless noted):
        joints      — ordered list of live ArticulatedJoint parts
        locked      — joints excluded for zero generalized inertia
        u           — (3+K,1) stacked [ω; q̇] symbols
        A           — (3+K)×(3+K) generalized mass matrix (function of q)
        p           — (3+K,1) generalized momentum ∂T/∂u
        bias        — (3+K,1) Hamel bias; the solve is
                          A·[α; q̈] = [τ_com; force_q] − bias
                      with τ_com supplied by the caller (so coupling
                      wrenches injected after this assembly still reach
                      the body rows — joint rows see part wrenches only)
        force_q     — (K,1) joint generalized-force rows (virtual work
                      over part wrenches + actuator/damping)
        dTdq        — (K,1) ∂T/∂q (conservative joint bias, for the
                      momentum integrator's joint-impulse advance)
    """
    from ..parts.articulation.joint import ArticulatedJoint

    joints, locked = [], []
    for part in craft.parts:
        if not isinstance(part, ArticulatedJoint):
            continue
        (joints if _generalized_inertia(part) > DEGENERATE_INERTIA_EPS
         else locked).append(part)
    if not joints:
        return None

    com_mx = inertia["com_in_craft_mx"]

    q_syms, qdot_syms = [], []
    for j in joints:
        pos_name, rate_name = j.dof_state_names()
        q_syms.append(state_input_nodes[j][pos_name]._mx)
        qdot_syms.append(state_input_nodes[j][rate_name]._mx)
    qvec    = ca.vertcat(*q_syms)
    qdotvec = ca.vertcat(*qdot_syms)
    u       = ca.vertcat(om_mx, qdotvec)
    K       = len(joints)

    # ----- rotational kinetic energy in COM coordinates -----------------
    from ..parts._trace import is_promoted
    T = ca.MX(0.0)
    for part in craft.parts:
        if not part.contributes_inertia:
            continue
        m_attr = part.mass
        if is_promoted(m_attr):
            m = m_attr._mx          # promoted (tunable) mass — keep symbolic
        else:
            m = float(m_attr)
            if m <= 0.0:
                continue
        kin = kin_states[part]
        rho = kin.r_in_craft - com_mx
        nu  = kin.velocity_rel_body - v_com_rel_mx
        v   = ca.cross(om_mx, rho) + nu
        T   = T + 0.5 * m * ca.dot(v, v)
        moi = getattr(part, "moi", (0.0, 0.0, 0.0))
        if is_promoted(moi):
            I_loc = ca.diag(moi._mx)    # promoted (tunable) moi — symbolic
        else:
            I_loc = ca.DM(np.diag([float(moi[0]), float(moi[1]),
                                   float(moi[2])]))
        R = kin.R_craft_from_input
        w_abs = kin.omega_input
        T = T + 0.5 * ca.dot(w_abs, R @ (I_loc @ (R.T @ w_abs)))

    # ----- generalized momentum, mass matrix, Hamel bias -----------------
    p = ca.jacobian(T, u).T                      # (3+K,1)
    A = ca.jacobian(p, u)                        # symmetric, q-dependent
    pdot_geom = ca.jacobian(p, qvec) @ qdotvec   # explicit q-drift of p
    dTdq = ca.jacobian(T, qvec).T                # (K,1)
    bias = (pdot_geom
            + ca.vertcat(ca.cross(om_mx, p[0:3]), ca.MX.zeros(K, 1))
            - ca.vertcat(ca.MX.zeros(3, 1), dTdq))

    # ----- joint generalized forces by virtual work ----------------------
    # Q_q[j] = Σ Fᵢ·(∂vᵢ/∂q̇_j) + τᵢ·(∂ωᵢ/∂q̇_j) — the exact motion-
    # subspace projection of every part wrench, with the COM-recoil
    # −F_tot·β̄ built in via the −v_com_rel term (which is what cancels
    # uniform gravity in free fall). The body rows are NOT built here:
    # the caller uses τ_com from the post-coupling net wrench, keeping
    # coupling handling identical to the flat path.
    force_q = ca.MX.zeros(K, 1)
    for part, w in own_wrench.items():
        kin = kin_states[part]
        nu = kin.velocity_rel_body - v_com_rel_mx
        force_q = force_q + ca.jacobian(nu, qdotvec).T @ w.force._mx
        force_q = force_q + (ca.jacobian(kin.omega_input, qdotvec).T
                             @ w.torque._mx)
    # Joint actuator + viscous damping: internal exchanges, own row only
    # (their body reaction lives in A's coupling, not in a force term).
    for jdx, j in enumerate(joints):
        force_q[jdx] = force_q[jdx] + j.applied_dof_force()

    # ----- degeneracy guard (compile-time, numeric at zero pose) ---------
    # A is PSD; near-singularity means a gauge DOF (massless stator).
    # Inflate the BODY block by a tiny compile-time epsilon — the joint
    # momentum rows stay exact, the gauge body-axial channel is pinned.
    A_used = A
    A0 = _evaluate_at_zero_pose(A, q_syms + qdot_syms, om_mx)
    scale = float(np.trace(A0)) / (3 + K)
    if scale <= 0.0:
        raise ValueError(
            f"Craft '{craft.name}': joint-space mass matrix has "
            f"non-positive trace at the zero pose.")
    min_eig = float(np.linalg.eigvalsh(A0).min())
    if min_eig < 1e-9 * scale:
        A_used = A + ca.diagcat(1e-6 * scale * ca.DM.eye(3),
                                ca.DM.zeros(K, K))

    return {
        "joints":  joints,
        "locked":  locked,
        "u":       u,
        "A":       A_used,
        "p":       p,
        "bias":    bias,
        "force_q": force_q,
        "dTdq":    dTdq,
    }


def _evaluate_at_zero_pose(expr_mx, dof_syms, om_mx) -> np.ndarray:
    """Numerically evaluate `expr_mx` (which may depend on the joint DOF
    symbols only — the mass matrix) with every DOF/rate symbol and the
    body ω symbol set to zero."""
    syms = list(dof_syms) + [om_mx]
    zeros = [ca.MX.zeros(*s.shape) if hasattr(s, "shape") else ca.MX(0.0)
             for s in syms]
    out = ca.substitute([expr_mx], syms, zeros)[0]
    dm = ca.evalf(out)
    return np.asarray(dm.full()).reshape(expr_mx.shape)
