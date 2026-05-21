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
craft. For a flat craft (no joints) the result is symbolically constant
and matches the legacy numpy `_aggregate_inertials` exactly. For a
craft with a Joint, the symbolic I_com varies with the joint angle —
e.g. a long thin rotor swinging out shifts the body's COM and rolls
its MOI tensor through R · diag · R^T.

Used at compile time by `Craft.compile_tick`: the symbolic com and I_com
feed straight into Newton-Euler, with `ca.solve(I_com_mx, τ)` replacing
the precomputed numpy inverse. The legacy numpy `_aggregate_inertials`
is kept as a non-symbolic introspection helper (`Craft.aggregate_inertials`)
and reports the at-rest snapshot — fine for `m_total` queries and for
sanity-checking degenerate inertia.

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

from typing import Any

import casadi as ca
import numpy as np


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def symbolic_inertia_rollup(root_part) -> dict:
    """Walk `root_part`'s subtree and return symbolic aggregate inertia.

    Returns a dict with:
        m_total            : float            — total mass (constant; Mass.mass
                                                is a Parameter, not state).
        com_in_craft_mx    : MX (3,)          — COM in body-frame coords.
        I_com_in_craft_mx  : MX (3,3)         — inertia tensor about COM, in
                                                body-frame coords.
        I_com_at_zero      : np.ndarray (3,3) — numerical snapshot at all
                                                joint angles = 0. Used by
                                                callers to detect singularity
                                                at compile time without doing
                                                a full ca.substitute pass.

    Raises ValueError if total mass is zero.
    """
    from .parts.articulation.joint import Joint
    from .parts.base import CompositePart

    # Accumulators (MX). com_sum is summed as m·r vectors; I_about_origin
    # is the inertia tensor about the craft origin, in body-frame coords.
    m_total            = 0.0
    com_sum_mx         = ca.MX.zeros(3, 1)
    I_about_origin_mx  = ca.MX.zeros(3, 3)
    eye3               = ca.MX.eye(3)

    def visit(part, r_parent_in_craft_mx, R_craft_from_parent_output_mx):
        nonlocal m_total, com_sum_mx, I_about_origin_mx

        # ----- part's own origin in body-frame coords ----------------------
        # part.transform lives in parent's OUTPUT frame coords.
        transform_mx = ca.MX(
            np.asarray(part.transform, dtype=float).reshape(3, 1))
        r_part_in_craft_mx = (r_parent_in_craft_mx
                              + ca.mtimes(R_craft_from_parent_output_mx,
                                          transform_mx))

        # ----- part's input + output frame rotations vs body --------------
        # The MOUNT doesn't rotate, so input = parent.output.
        R_craft_from_input_mx = R_craft_from_parent_output_mx

        if isinstance(part, Joint):
            axis_mx = ca.MX(np.asarray(part.axis, dtype=float).reshape(3, 1))
            angle_attr = part.angle
            angle_mx = (angle_attr._mx if hasattr(angle_attr, "_mx")
                        else ca.MX(float(angle_attr)))
            R_in_from_out_mx = _R_from_axis_angle_mx(axis_mx, angle_mx)
            R_craft_from_output_mx = ca.mtimes(
                R_craft_from_input_mx, R_in_from_out_mx)
        else:
            R_craft_from_output_mx = R_craft_from_input_mx

        # ----- contribute this part's m, m·r, I_own_in_craft ---------------
        # A Mass is a leaf — its own frame = input frame = parent output.
        # Joint itself has no mass attribute, so the `getattr` returns None.
        mass_attr = getattr(part, "mass", None)
        if mass_attr is not None:
            m = float(mass_attr)
            if m > 0.0:
                moi_attr = getattr(part, "moi", (0.0, 0.0, 0.0))
                I_diag_local_mx = ca.diag(ca.MX(np.asarray(
                    [float(moi_attr[0]),
                     float(moi_attr[1]),
                     float(moi_attr[2])])))
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
                m_total    += m
                com_sum_mx = com_sum_mx + m * r

        # ----- recurse into children ---------------------------------------
        if isinstance(part, CompositePart):
            for child in part.children:
                visit(child, r_part_in_craft_mx, R_craft_from_output_mx)

    # Visit each child of root with parent state = root (r=0, R=I).
    from .parts.base import CompositePart as _CompositePart
    if isinstance(root_part, _CompositePart):
        for child in root_part.children:
            visit(child, ca.MX.zeros(3, 1), ca.MX.eye(3))

    if m_total <= 0.0:
        raise ValueError(
            "symbolic_inertia_rollup: total mass is zero. Add at least one "
            "Mass part to the craft.")

    com_mx     = com_sum_mx / m_total
    com_dot    = ca.mtimes(com_mx.T, com_mx)
    outer_com  = ca.mtimes(com_mx, com_mx.T)
    I_com_mx   = I_about_origin_mx - m_total * (com_dot * eye3 - outer_com)

    # Numerical I_com at all joint angles = 0, for compile-time singularity
    # detection. Substitute any MX `Joint.angle._mx` / `Joint.rate._mx`
    # symbols with 0 inside I_com_mx.
    I_com_at_zero = _evaluate_at_zero(I_com_mx, root_part)

    return {
        "m_total":           m_total,
        "com_in_craft_mx":   com_mx,
        "I_com_in_craft_mx": I_com_mx,
        "I_com_at_zero":     I_com_at_zero,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _R_from_axis_angle_mx(axis_mx, angle_mx):
    """Rodrigues' formula in MX. `axis_mx` is 3×1; normalized symbolically
    with an eps softener so a degenerate (zero) axis doesn't crash. Returns
    a 3×3 MX expressing the rotation matrix R(axis, angle)."""
    n_sq = ca.mtimes(axis_mx.T, axis_mx) + 1.0e-30
    n    = ca.sqrt(n_sq)
    u    = axis_mx / n                # 3×1 unit axis
    c    = ca.cos(angle_mx)
    s    = ca.sin(angle_mx)
    ux, uy, uz = u[0], u[1], u[2]
    zero = ca.MX(0.0)
    K = ca.vertcat(
        ca.horzcat(zero, -uz, uy),
        ca.horzcat(uz, zero, -ux),
        ca.horzcat(-uy, ux, zero),
    )
    eye3 = ca.MX.eye(3)
    return eye3 + s * K + (1.0 - c) * ca.mtimes(K, K)


def _evaluate_at_zero(expr_mx, root_part) -> np.ndarray:
    """Evaluate `expr_mx` with every Joint's angle/rate MX symbol substituted
    by 0. Used for compile-time singularity checks against the at-rest
    inertia tensor without spinning up a full CasADi Function."""
    from .parts.articulation.joint import Joint

    def collect_joint_syms(part, syms):
        if isinstance(part, Joint):
            angle_attr = part.angle
            rate_attr  = part.rate
            if hasattr(angle_attr, "_mx"):
                syms.append(angle_attr._mx)
            if hasattr(rate_attr, "_mx"):
                syms.append(rate_attr._mx)
        from .parts.base import CompositePart
        if isinstance(part, CompositePart):
            for c in part.children:
                collect_joint_syms(c, syms)

    syms: list = []
    collect_joint_syms(root_part, syms)
    if syms:
        zeros = [ca.MX(0.0)] * len(syms)
        expr_mx = ca.substitute([expr_mx], syms, zeros)[0]
    # `ca.evalf` folds an MX with no free symbols straight to DM.
    dm = ca.evalf(expr_mx)
    return np.asarray(dm.full()).reshape(expr_mx.shape)
