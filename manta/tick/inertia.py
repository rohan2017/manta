"""Symbolic inertial aggregation over a Craft's part tree.

For each Mass-bearing part `P` in the tree, compute its position in
body-frame coords and the rotation that takes its own (diagonal) MOI
tensor into body-frame coords, then sum:

    m_total              = Σ_P m_P
    com_in_craft         = Σ_P m_P · r_P_in_craft  /  m_total
    I_about_origin       = Σ_P [R_P · diag(moi_P) · R_P^T
                              + m_P · (|r_P|²·I − r_P·r_P^T)]
    I_com                = I_about_origin
                           − m_total · (|com|²·I − com·com^T)

All quantities are CasADi MX expressions in the Joint angles of the
craft. For a flat craft (no joints) the result is symbolically
constant. For a craft with a Joint, the symbolic I_com varies with the
joint angle — e.g. a long thin rotor swinging out shifts the body's COM
and rolls its MOI tensor through R · diag · R^T.

Used at compile time by the world tick (`compile_world_tick`): the
symbolic com and I_com feed straight into Newton-Euler, with
`ca.solve(I_com_mx, τ)` replacing a precomputed inverse. This rollup is
the ONLY inertial tree walk: `Craft.aggregate_inertials` (introspection
/ tests) evaluates the same expressions outside a trace — where every
attribute reads as its declared numeric value, so they fold straight to
constants — rather than maintaining a numpy twin of the geometry.

Frame convention: `r_P_in_craft` is in body (CraftFrame) coords;
`R_craft_from_P` rotates a vector expressed in the part's input/output
frame coords into body-frame coords. The pass starts at the root with
`r = 0`, `R = I`; for each child it composes through the parent's
output frame (= input rotated by the joint angle if the parent is a
Joint).

Mass-frame convention: a `Mass` is a leaf; its diagonal MOI tensor
lives in its OWN frame, which equals its INPUT frame (no further
rotation). Its input frame is the parent's output frame, so
`R_craft_from_mass = R_craft_from_parent_output`. For a Mass mounted
on a Joint, that rotation includes the joint angle — the rotor's MOI
tensor rotates with the rotor, as it should.
"""

from __future__ import annotations


import casadi as ca
import numpy as np

from ..ir._rotation import R_from_axis_angle, quat_to_rotmat


# ---------------------------------------------------------------------------
# Reading a part's inertials
# ---------------------------------------------------------------------------
# Three passes need "what mass/moi does this part contribute, and is it a
# promoted (tunable) symbol?" — the rollup below, the COM-recoil reduction
# in `world_tick`, and the kinetic-energy sum in `joint_space`. The rule is
# subtle enough to be worth having once: a promoted mass contributes even
# when its DECLARED value is zero (the optimizer may move it), while a
# fixed zero mass drops out entirely.
#
# `is_promoted` is imported lazily, as everywhere in this package: `parts`
# imports `tick`, so a module-scope import would close the cycle.

def inertial_mass(part):
    """`(m, m_declared)` for an inertial `part`, or None to skip it.

    `m` is the MX symbol when the mass is promoted and the float
    otherwise; `m_declared` is always the float, for numeric totals and
    guards. None means the part contributes no inertia — either it lacks
    the trait (a bare `mass` attribute is not inertial; TrajectoryEndpoint's
    is a gain) or its fixed mass is zero.
    """
    from ..parts._trace import is_promoted
    if not part.contributes_inertia:
        return None
    attr = part.mass
    if is_promoted(attr):
        return attr._mx, float(part.declared_value("mass"))
    m = float(attr)
    return None if m <= 0.0 else (m, m)


def inertial_moi_diag(part):
    """`diag(moi)` as a 3×3 MX — the symbol when the moi is promoted,
    the declared constants otherwise. Parts without a `moi` read zero."""
    from ..parts._trace import is_promoted
    attr = getattr(part, "moi", (0.0, 0.0, 0.0))
    if is_promoted(attr):
        return ca.diag(attr._mx)
    return ca.diag(ca.MX(np.asarray([float(attr[0]), float(attr[1]),
                                     float(attr[2])])))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def symbolic_inertia_rollup(root_part, *, param_subs=(), who: str = "") -> dict:
    """Walk `root_part`'s subtree and return symbolic aggregate inertia.

    Args:
        param_subs — `(sym_mx, declared_value)` pairs for any promoted
                     (tunable) parameters bound on the active trace; the
                     at-rest numeric snapshot substitutes these at their
                     declared values.
        who        — craft name, to name the offender in the zero-mass error.

    Returns a dict with:
        m_total            : float | MX       — total mass. A float unless a
                                                part's mass is promoted to a
                                                tunable input (then MX).
        com_in_craft_mx    : MX (3,)          — COM in body-frame coords.
        I_com_in_craft_mx  : MX (3,3)         — inertia tensor about COM, in
                                                body-frame coords.
        I_origin_in_craft_mx : MX (3,3)       — the same tensor about the
                                                craft ORIGIN (pre parallel-
                                                axis reduction).
        I_com_at_zero      : np.ndarray (3,3) — numerical snapshot at all
                                                joint angles = 0 (and all
                                                promoted parameters at their
                                                declared values). Used by
                                                callers to detect singularity
                                                at compile time without doing
                                                a full ca.substitute pass.

    Raises ValueError if total (declared) mass is zero.
    """
    from ..parts.articulation.joint import PrismaticJoint, RevoluteDOF
    from ..parts._trace import is_promoted
    from ..parts.base import CompositePart

    # Accumulators (MX). com_sum is summed as m·r vectors; I_about_origin
    # is the inertia tensor about the craft origin, in body-frame coords.
    # m_total stays a float unless a promoted (symbolic) mass joins the sum;
    # m_total_decl tracks the declared numeric total for the zero-mass guard.
    m_total            = 0.0
    m_total_decl       = 0.0
    com_sum_mx         = ca.MX.zeros(3, 1)
    I_about_origin_mx  = ca.MX.zeros(3, 3)
    eye3               = ca.MX.eye(3)

    def visit(part, r_parent_in_craft_mx, R_craft_from_parent_output_mx):
        nonlocal m_total, m_total_decl, com_sum_mx, I_about_origin_mx

        # ----- part's own origin in body-frame coords ----------------------
        # part.mount_offset lives in parent's OUTPUT frame coords. A promoted
        # (tunable) transform reads as a bound IR value — keep the symbol.
        tr_attr = part.mount_offset
        transform_mx = (tr_attr._mx if is_promoted(tr_attr) else ca.MX(
            np.asarray(tr_attr, dtype=float).reshape(3, 1)))
        r_part_in_craft_mx = (r_parent_in_craft_mx
                              + ca.mtimes(R_craft_from_parent_output_mx,
                                          transform_mx))

        # ----- part's input + output frame rotations vs body --------------
        # The mount TURNS the part's own axes inside the parent's output
        # frame (`mount_orientation`, the same static pose the kinematic
        # pass composes). Skipping it would leave a rotated bracket's
        # subtree unrotated here while the kinematics places it correctly
        # — the inertia tensor and the sensor geometry would disagree.
        q_attr = part.mount_orientation
        q_mx = (q_attr._mx if is_promoted(q_attr)
                else ca.MX(np.asarray(q_attr, dtype=float).reshape(4, 1)))
        R_craft_from_input_mx = ca.mtimes(R_craft_from_parent_output_mx,
                                          quat_to_rotmat(q_mx))

        r_out_in_craft_mx = r_part_in_craft_mx
        if isinstance(part, RevoluteDOF):
            axis_mx = ca.MX(np.asarray(part.axis, dtype=float).reshape(3, 1))
            angle_attr = part.angle
            angle_mx = (angle_attr._mx if is_promoted(angle_attr)
                        else ca.MX(float(angle_attr)))
            R_in_from_out_mx = R_from_axis_angle(axis_mx, angle_mx)
            R_craft_from_output_mx = ca.mtimes(
                R_craft_from_input_mx, R_in_from_out_mx)
        elif isinstance(part, PrismaticJoint):
            # No rotation; the output origin (children's mount) slides by
            # `displacement · axis` within the input frame — COM and I_com
            # pick up the displacement dependence here. The constructor
            # normalized `axis` (unit_axis invariant).
            axis_np = np.asarray(part.axis, dtype=float)
            disp_attr = part.displacement
            disp_mx = (disp_attr._mx if is_promoted(disp_attr)
                       else ca.MX(float(disp_attr)))
            R_craft_from_output_mx = R_craft_from_input_mx
            r_out_in_craft_mx = (
                r_part_in_craft_mx
                + ca.mtimes(R_craft_from_input_mx,
                            disp_mx * ca.DM(axis_np.reshape(3, 1))))
        else:
            R_craft_from_output_mx = R_craft_from_input_mx

        # ----- contribute this part's m, m·r, I_own_in_craft ---------------
        # A Mass is a leaf — its own frame = input frame = parent output.
        # `inertial_mass` owns the trait/promotion/zero-skip rule (above).
        contribution = inertial_mass(part)
        if contribution is not None:
            m, m_decl = contribution
            I_diag_local_mx = inertial_moi_diag(part)
            # The Mass rotates with whatever its parent rotor frame is.
            R_mass_in_craft_mx = R_craft_from_input_mx
            I_own_in_craft_mx = ca.mtimes(R_mass_in_craft_mx,
                ca.mtimes(I_diag_local_mx, R_mass_in_craft_mx.T))

            # Parallel-axis lift about craft origin.
            r = r_part_in_craft_mx                       # 3×1
            r_dot_r = ca.mtimes(r.T, r)                  # 1×1 MX
            outer_r = ca.mtimes(r, r.T)                  # 3×3
            I_about_origin_mx = (I_about_origin_mx
                                 + I_own_in_craft_mx
                                 + m * (r_dot_r * eye3 - outer_r))
            m_total      = m_total + m
            m_total_decl += m_decl
            com_sum_mx   = com_sum_mx + m * r

        # ----- recurse into children ---------------------------------------
        if isinstance(part, CompositePart):
            for child in part.children:
                visit(child, r_out_in_craft_mx, R_craft_from_output_mx)

    # Visit each child of root with parent state = root (r=0, R=I).
    from ..parts.base import CompositePart as _CompositePart
    if isinstance(root_part, _CompositePart):
        for child in root_part.children:
            visit(child, ca.MX.zeros(3, 1), ca.MX.eye(3))

    if m_total_decl <= 0.0:
        where = f"Craft {who!r}: " if who else ""
        raise ValueError(
            f"{where}total mass is zero. Add at least one Mass part to the "
            f"craft.")

    com_mx     = com_sum_mx / m_total
    com_dot    = ca.mtimes(com_mx.T, com_mx)
    outer_com  = ca.mtimes(com_mx, com_mx.T)
    I_com_mx   = I_about_origin_mx - m_total * (com_dot * eye3 - outer_com)

    # Numerical I_com at all joint angles = 0 (and promoted parameters at
    # their declared values), for compile-time singularity detection.
    I_com_at_zero = _evaluate_at_zero(I_com_mx, root_part, param_subs)

    return {
        "m_total":              m_total,
        "com_in_craft_mx":      com_mx,
        "I_com_in_craft_mx":    I_com_mx,
        "I_origin_in_craft_mx": I_about_origin_mx,
        "I_com_at_zero":        I_com_at_zero,
    }


def aggregate_inertials_at_rest(root_part) -> dict:
    """The rollup's numeric snapshot at rest: total mass, COM, and the
    inertia tensor about both the craft origin and the COM, as
    plain floats / ndarrays.

    This is `symbolic_inertia_rollup` evaluated rather than a second
    tree walk. The two used to be hand-written twins under a "must
    match exactly" comment, and drifted: the numeric one never picked
    up `mount_orientation`. Deriving it removes the invariant.

    "At rest" is the rollup's own snapshot convention — every joint DOF
    on the path at zero, every promoted parameter at its declared value.
    A massless craft returns zeros rather than raising: callers of this
    are introspection and the compile-time mass guard, both of which
    want to see the zero, not a traceback.
    """
    try:
        roll = symbolic_inertia_rollup(root_part)
    except ValueError:
        return {"m_total": 0.0, "com": np.zeros(3),
                "I_origin": np.zeros((3, 3)), "I_com": np.zeros((3, 3))}
    evaluate = lambda e: _evaluate_at_zero(e, root_part)   # noqa: E731
    m_total = roll["m_total"]
    return {
        "m_total":  (float(evaluate(m_total).reshape(-1)[0])
                     if isinstance(m_total, ca.MX) else float(m_total)),
        "com":      evaluate(roll["com_in_craft_mx"]).reshape(3),
        "I_origin": evaluate(roll["I_origin_in_craft_mx"]),
        "I_com":    evaluate(roll["I_com_in_craft_mx"]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_at_zero(expr_mx, root_part, param_subs=()) -> np.ndarray:
    """Evaluate `expr_mx` with every joint's DOF/rate MX symbol substituted
    by 0 and every promoted-parameter symbol by its declared value. Used
    for compile-time singularity checks against the at-rest inertia tensor
    without spinning up a full CasADi Function."""
    from ..parts.articulation.joint import ArticulatedJoint
    from ..parts._trace import is_promoted

    def collect_joint_syms(part, syms):
        if isinstance(part, ArticulatedJoint):
            for sname in part.dof_state_names():
                attr = getattr(part, sname)
                if is_promoted(attr):
                    syms.append(attr._mx)
        from ..parts.base import CompositePart
        if isinstance(part, CompositePart):
            for c in part.children:
                collect_joint_syms(c, syms)

    syms: list = []
    collect_joint_syms(root_part, syms)
    vals = [ca.MX(0.0)] * len(syms)
    for sym, declared in param_subs:
        syms.append(sym)
        vals.append(ca.MX(declared))
    if syms:
        expr_mx = ca.substitute([expr_mx], syms, vals)[0]
    # `ca.evalf` folds an MX with no free symbols straight to DM.
    dm = ca.evalf(expr_mx)
    return np.asarray(dm.full()).reshape(expr_mx.shape)


def added_mass_rollup(craft, kin_states, param_subs=()) -> dict | None:
    """Collect the craft's `AddedMass` parts into craft-frame tensors.

    Added mass augments the SOLVES, not the mass rollup: entrained fluid
    weighs nothing, never shifts the COM, and contributes no gravity —
    which is why `AddedMass` is not `contributes_inertia` and this walk
    is separate from `symbolic_inertia_rollup`.

    Returns None when the craft carries no added mass; otherwise:
        A_lin_mx      : MX (3,3) — added linear inertia, craft frame.
        B_rot_mx      : MX (3,3) — added rotational inertia, craft frame.
        B_rot_at_zero : np.ndarray (3,3) — numeric snapshot (promoted
                        parameters at declared values) for the
                        degenerate-inertia guards.

    Raises if an `AddedMass` rides an articulated joint: the tensors
    would become configuration-dependent, and a spinning appendage's
    added mass is not a modelled effect — mount it rigidly.
    """
    from ..parts._trace import is_promoted
    from ..parts.aero.added_mass import AddedMass
    from ..parts.articulation.joint import ArticulatedJoint
    from ..parts.base import CompositePart

    def collect(part, under_joint, out):
        if isinstance(part, AddedMass) and under_joint:
            raise ValueError(
                f"Craft '{craft.name}': AddedMass '{part.name}' is "
                f"mounted under an articulated joint. Its tensors would "
                f"become configuration-dependent; added mass models the "
                f"rigid hull — mount it rigidly.")
        if isinstance(part, AddedMass):
            out.append(part)
        if isinstance(part, CompositePart):
            for child in part.children:
                collect(child, under_joint
                        or isinstance(part, ArticulatedJoint), out)

    parts: list = []
    collect(craft.root, False, parts)
    if not parts:
        return None

    A_mx = ca.MX.zeros(3, 3)
    B_mx = ca.MX.zeros(3, 3)
    for part in parts:
        # Includes the static mount rotation — a diagonal authored in
        # the part's frame lands in craft axes here.
        R = kin_states[part].R_craft_from_input
        for attr, is_linear in ((part.translational, True),
                                (part.rotational, False)):
            if is_promoted(attr):
                diag_mx = ca.diag(attr._mx)
            else:
                diag_mx = ca.diag(ca.MX(np.asarray(
                    [float(attr[0]), float(attr[1]), float(attr[2])])))
            rotated = ca.mtimes(R, ca.mtimes(diag_mx, R.T))
            if is_linear:
                A_mx = A_mx + rotated
            else:
                B_mx = B_mx + rotated

    return {
        "A_lin_mx": A_mx,
        "B_rot_mx": B_mx,
        "B_rot_at_zero": _evaluate_at_zero(B_mx, craft.root, param_subs),
    }
