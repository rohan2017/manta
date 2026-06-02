"""Symbolic kinematic pass over a Craft's part tree.

Given the body's rigid-body state (position, orientation, velocity,
angular velocity), the body's symbolic acceleration / angular
acceleration, and every Joint's (angle, rate, θ̈), compute for each part
its effective kinematic state — origin position, linear velocity at
the origin, angular velocity/acceleration, the part's body-frame
position, and the rotation matrices that take its input/output frames
into body coords — all as CasADi MX expressions composed through the
joint chain. A flat craft (no nested joints) reduces to "everything is
the body's state offset by `part.transform`" with identity rotations;
deep nesting (gimbal: pan → tilt → camera) composes naturally.

Each part's absolute acceleration is the rotating-frame transport of the
body's a/α to the part's mount point PLUS the joint-induced relative
motion: the rigid lever arm (α×r + ω×(ω×r)) augmented with the Coriolis
term 2·ω_body×v_rel and the relative acceleration a_rel, where (v_rel,
a_rel) are the part's velocity/acceleration in the (rotating) body frame
— nonzero only for parts on a moving joint. This is what lets a sensor on
a spinning rotor read the right specific force; the body-relative COM
motion (the mass-weighted reduction of these) drives the moving-COM
origin recoil in `world_tick`.

Body acceleration handling: an accelerometer's reading depends on
body a/α, which is the OUTPUT of Newton-Euler — reading them inside
`update()` (the wrench-collection phase, which feeds N-E) is
circular. The framework passes MX **placeholder** symbols for a/α
(and, for the same reason, each Joint's θ̈) into this pass, lets parts
emit outputs referencing them, then substitutes the real expressions
into those outputs after the wrench sum is known. Wrenches that would
themselves depend on a/α (or θ̈) are rejected at compile time (no
implicit-equation solver).

**Frame convention**:
  * World-frame fields (`origin_in_world`, `velocity_origin`,
    `orientation_anchor_from_*`) carry the part's actual absolute pose.
  * "In CraftFrame" fields (`angular_velocity_input/output`,
    `r_in_craft`, `R_craft_from_input/output`) are expressed in the **body's
    CraftFrame coords** even for nested parts whose own frame differs
    by the joint chain. Wrench aggregation lifts each part's emitted
    wrench through `r_in_craft`; articulated parts that emit
    input-frame quantities (Joint axis, etc.) rotate them through
    `R_craft_from_input` inside their own `update()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca

from .ir._rotation import (
    R_from_axis_angle,
    quat_from_axis_angle,
    quat_mul,
    rotate_vec_by_quat,
)
from .ir.frames import WorldFrame, CraftFrame, ParentFrame, PartFrame
from .ir.types import Mat3, Quat, Vec3


# ---------------------------------------------------------------------------
# KinematicState
# ---------------------------------------------------------------------------

@dataclass
class KinematicState:
    """Kinematic state of one part — what its `TickContext` fields
    resolve to. All values are MX expressions; the Vec3/Quat/Mat3 are
    the typed manta IR wrappers.

    Two related views of the same chain are tracked:
      * World-frame quantities (`origin_in_world`, `velocity_origin`,
        `orientation_anchor_from_*`) — the part's absolute pose.
      * Body-frame quantities (`r_in_craft`, `R_craft_from_*`) — joint
        composition without the body's orientation. Inertia rollup and
        wrench lifting consume these directly so neither needs to
        unwind the body's orientation.

    Fields:
        origin_in_world              — Vec3[WorldFrame]: world-frame
                                         position of the part's mount
                                         origin.
        velocity_origin               — Vec3[WorldFrame]: world-frame
                                         linear velocity at that origin.
        angular_velocity_input        — Vec3[CraftFrame]: ω of the part's
                                         input frame, in body-frame coords.
        angular_velocity_output       — Vec3[CraftFrame]: ω of the part's
                                         output frame. Equals input for
                                         non-Joint parts.
        orientation_anchor_from_input — Quat: world-frame orientation of
                                         the part's input frame.
        orientation_anchor_from_output — Quat: same for the output frame.
        r_in_craft                    — Vec3[CraftFrame]: body-frame
                                         position of the part's origin.
                                         Equals `part.transform` for a
                                         flat craft; depends on joint
                                         angles in a nested chain.
        R_craft_from_input            — Mat3[CraftFrame, CraftFrame]:
                                         rotates a vector in the part's
                                         INPUT-frame coords into body-
                                         frame coords. Identity for
                                         parts mounted directly on the
                                         craft root.
        R_craft_from_output           — Mat3[CraftFrame, CraftFrame]:
                                         same for the OUTPUT frame. For
                                         a non-Joint part, equals
                                         `R_craft_from_input`. For a
                                         Joint, adds the axis-angle
                                         rotation.
    """

    origin_in_world:               Any   # Vec3[WorldFrame]
    velocity_origin:                Any   # Vec3[WorldFrame]
    angular_velocity_input:         Any   # Vec3[CraftFrame]
    angular_velocity_output:        Any   # Vec3[CraftFrame]
    orientation_anchor_from_input:  Any   # Quat
    orientation_anchor_from_output: Any   # Quat
    r_in_craft:                     Any   # Vec3[CraftFrame]
    R_craft_from_input:             Any   # Mat3[CraftFrame, CraftFrame]
    R_craft_from_output:            Any   # Mat3[CraftFrame, CraftFrame]
    acceleration_world:            Any   # Vec3[WorldFrame]
    # --- Body-relative (joint-induced) kinematics, body-frame coords ----
    # Motion of this part's origin / output frame relative to the body
    # frame, i.e. due ONLY to the joint angles between body and part
    # (zero for the root and for any part rigidly fixed to the body).
    # The absolute world/body acceleration above is composed from these
    # via the rotating-frame transport theorem; the moving-COM origin
    # recoil is their mass-weighted reduction.
    velocity_rel_body:              Any   # Vec3[CraftFrame]: ṙ|_body
    acceleration_rel_body:          Any   # Vec3[CraftFrame]: r̈|_body
    omega_rel_output:               Any   # Vec3[CraftFrame]: relative ω
    alpha_rel_output:               Any   # Vec3[CraftFrame]: relative α
    # --- Frame-indexed views for the part's TickContext -----------------
    # frame_views[quantity][Frame] → Vec3, for quantity in
    # {position, velocity, acceleration, angular_velocity,
    # angular_acceleration} and Frame in {World, Craft, Parent, Part}.
    # X[F] is the quantity measured RELATIVE TO frame F, in F's coords —
    # X[PartFrame] is 0, X[WorldFrame] is the absolute/inertial value,
    # X[CraftFrame] is the joint-induced motion w.r.t. the craft (0 if
    # rigidly mounted), X[ParentFrame] is motion w.r.t. the immediate
    # parent. `TickContext` wraps each in a `_FrameView`.
    frame_views:                    Any   # dict[str, dict[type, Vec3]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble_frame_views(*,
                          body_orientation,
                          # WorldFrame (absolute) quantities:
                          origin_in_world, velocity_origin, acceleration_world,
                          # CraftFrame: r_in_craft + relative-to-craft motion:
                          r_in_craft, velocity_rel_body, acceleration_rel_body,
                          # part-frame (= input frame) angular, craft coords:
                          angular_velocity_input, omega_rel_input,
                          alpha_rel_input, alpha_abs_input,
                          # ParentFrame quantities (parent-frame coords):
                          pos_rel_parent, velocity_rel_parent,
                          acceleration_rel_parent, omega_rel_parent,
                          alpha_rel_parent) -> dict:
    """Bundle the per-frame views a part's TickContext exposes. Each
    `X[F]` is the quantity relative to frame F, in F's coords (see
    `KinematicState.frame_views`). The WorldFrame angular quantities are
    the part-frame (input) ω/α rotated from craft coords into world."""
    zero_part = Vec3[PartFrame].from_mx(ca.MX.zeros(3, 1))
    ang_vel_world = body_orientation.apply(angular_velocity_input)
    ang_acc_world = body_orientation.apply(alpha_abs_input)
    # Retag world-frame angular outputs as WorldFrame (apply already does).
    return {
        "position": {
            WorldFrame: origin_in_world,
            CraftFrame: r_in_craft,
            ParentFrame: pos_rel_parent,
            PartFrame: zero_part,
        },
        "velocity": {
            WorldFrame: velocity_origin,
            CraftFrame: velocity_rel_body,
            ParentFrame: velocity_rel_parent,
            PartFrame: zero_part,
        },
        "acceleration": {
            WorldFrame: acceleration_world,
            CraftFrame: acceleration_rel_body,
            ParentFrame: acceleration_rel_parent,
            PartFrame: zero_part,
        },
        "angular_velocity": {
            WorldFrame: ang_vel_world,
            CraftFrame: omega_rel_input,
            ParentFrame: omega_rel_parent,
            PartFrame: zero_part,
        },
        "angular_acceleration": {
            WorldFrame: ang_acc_world,
            CraftFrame: alpha_rel_input,
            ParentFrame: alpha_rel_parent,
            PartFrame: zero_part,
        },
    }


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

def kinematic_pass(root_part,
                   body_position,
                   body_orientation,
                   body_velocity,
                   body_angular_velocity,
                   *,
                   body_acceleration_world,
                   body_angular_acceleration,
                   joint_angular_accels=None) -> dict:
    """Walk `root_part`'s subtree, returning `{part: KinematicState}`.

    Joint angle/rate values are read directly off each Joint instance
    via `joint.angle._mx` / `joint.rate._mx` — assumes the framework has
    already rebound the State attributes to symbolic MX inputs before
    this is called. A Joint's angular acceleration θ̈ is NOT a state
    slot, so it is supplied separately via `joint_angular_accels`
    (`{joint: MX scalar}`); like the body's a/α these are compile-time
    placeholders that the caller substitutes with the real
    state-function `(new_rate − rate)/dt` after `Joint.update()` runs.
    Joints absent from the dict (or `joint_angular_accels=None`) take
    θ̈ = 0 — fine for the velocity/position outputs, which don't use it.

    The root's state coincides with the body. Each child:
      * Position composes through `parent.orientation_output · transform`.
      * Velocity composes through standard rigid-body kinematics:
          v_child = v_parent_origin + ω_parent_output × (r_child - r_parent_origin).
      * Angular velocity of the part's input frame equals the parent's
        output frame ω; the output frame additionally gets `rate · axis`
        if the part is a Joint.
      * Body-relative kinematics (velocity_rel_body, acceleration_rel_body,
        ω/α_rel) propagate the same way with the body frame as the fixed
        base — these capture the joint-induced motion that the rigid
        lever-arm transfer alone misses, so a sensor on a moving rotor
        reads the right specific force and the system COM recoil is exact.
    """
    from .parts.base import CompositePart

    if joint_angular_accels is None:
        joint_angular_accels = {}

    # ----- root state (coincides with body) -----------------------------
    eye3_mat = Mat3[CraftFrame, CraftFrame].from_mx(ca.MX.eye(3))
    zero_r   = Vec3[CraftFrame].from_mx(ca.MX.zeros(3, 1))
    zero_vec = Vec3[CraftFrame].from_mx(ca.MX.zeros(3, 1))

    # Root sees the body's own acceleration directly (no lever arm). The
    # caller passes a / α as MX placeholder symbols that get substituted to
    # the real Newton-Euler outputs after the wrench-collection phase.
    root_state = KinematicState(
        origin_in_world=body_position,
        velocity_origin=body_velocity,
        angular_velocity_input=body_angular_velocity,
        angular_velocity_output=body_angular_velocity,
        orientation_anchor_from_input=body_orientation,
        orientation_anchor_from_output=body_orientation,
        r_in_craft=zero_r,
        R_craft_from_input=eye3_mat,
        R_craft_from_output=eye3_mat,
        acceleration_world=body_acceleration_world,
        velocity_rel_body=zero_vec,
        acceleration_rel_body=zero_vec,
        omega_rel_output=zero_vec,
        alpha_rel_output=zero_vec,
        frame_views=_assemble_frame_views(
            body_orientation=body_orientation,
            origin_in_world=body_position,
            velocity_origin=body_velocity,
            acceleration_world=body_acceleration_world,
            r_in_craft=zero_r,
            velocity_rel_body=zero_vec,
            acceleration_rel_body=zero_vec,
            angular_velocity_input=body_angular_velocity,
            omega_rel_input=zero_vec,
            alpha_rel_input=zero_vec,
            alpha_abs_input=body_angular_acceleration,
            # Root has no parent — ParentFrame views coincide with the
            # craft (zero relative motion). Tagged ParentFrame for shape.
            pos_rel_parent=Vec3[ParentFrame].from_mx(ca.MX.zeros(3, 1)),
            velocity_rel_parent=Vec3[ParentFrame].from_mx(ca.MX.zeros(3, 1)),
            acceleration_rel_parent=Vec3[ParentFrame].from_mx(ca.MX.zeros(3, 1)),
            omega_rel_parent=Vec3[ParentFrame].from_mx(ca.MX.zeros(3, 1)),
            alpha_rel_parent=Vec3[ParentFrame].from_mx(ca.MX.zeros(3, 1)),
        ),
    )

    states: dict = {root_part: root_state}

    def visit_children(parent, parent_state):
        if not isinstance(parent, CompositePart):
            return
        for child in parent.children:
            child_state = _compute_child_state(
                parent_state, parent, child, body_orientation,
                body_acceleration_world=body_acceleration_world,
                body_angular_acceleration=body_angular_acceleration,
                body_angular_velocity=body_angular_velocity,
                joint_angular_accels=joint_angular_accels)
            states[child] = child_state
            visit_children(child, child_state)

    visit_children(root_part, root_state)
    return states


def _compute_child_state(parent_state: KinematicState, parent_part, child,
                         body_orientation,
                         *,
                         body_acceleration_world,
                         body_angular_acceleration,
                         body_angular_velocity,
                         joint_angular_accels=None) -> KinematicState:
    from .parts.articulation.joint import Joint

    if joint_angular_accels is None:
        joint_angular_accels = {}

    # Child's `transform` lives in parent's OUTPUT frame coords.
    transform_mx = ca.MX(list(child.transform))

    # ----- body-frame position composition ------------------------------
    # r_child_in_craft = r_parent_in_craft + R_craft_from_parent_output · transform.
    R_craft_from_parent_out_mx = parent_state.R_craft_from_output._mx
    r_parent_in_craft_mx = parent_state.r_in_craft._mx
    # Offset from parent origin to child origin, in body-frame coords.
    offset_in_craft_mx = ca.mtimes(R_craft_from_parent_out_mx, transform_mx)
    child_r_in_craft_mx = r_parent_in_craft_mx + offset_in_craft_mx
    child_r_in_craft = Vec3[CraftFrame].from_mx(child_r_in_craft_mx)

    # Mount doesn't rotate, so child's INPUT frame = parent's OUTPUT.
    R_craft_from_child_input_mx = R_craft_from_parent_out_mx
    R_craft_from_child_input = Mat3[CraftFrame, CraftFrame].from_mx(
        R_craft_from_child_input_mx)

    # ----- position of child's origin in WorldFrame --------------------
    # Step 1: rotate transform from parent-output-frame coords to anchor
    # coords via the parent's output orientation.
    q_anchor_from_parent_output = parent_state.orientation_anchor_from_output
    offset_in_world_mx = rotate_vec_by_quat(
        q_anchor_from_parent_output._mx, transform_mx)
    offset_in_world = Vec3[WorldFrame].from_mx(offset_in_world_mx)
    child_origin_in_world = parent_state.origin_in_world + offset_in_world

    # ----- linear velocity at child's origin in WorldFrame -------------
    # v_child = v_parent_origin + ω_parent_output (in anchor) × offset_in_world
    # ω stored in body-frame coords; rotate via body_orientation.
    ω_parent_out_world = body_orientation.apply(
        parent_state.angular_velocity_output)
    child_velocity_origin = (parent_state.velocity_origin
                              + ω_parent_out_world.cross(offset_in_world))

    # ----- child input-frame ω + orientation ----------------------------
    # Child's input frame = parent's output frame.
    q_anchor_from_input_mx = q_anchor_from_parent_output._mx
    q_anchor_from_input = Quat[WorldFrame, CraftFrame].from_mx(
        q_anchor_from_input_mx)
    child_omega_input_in_craft = parent_state.angular_velocity_output

    # Body-relative angular velocity / acceleration of the INPUT frame
    # equal the parent's OUTPUT-frame relative values (the mount is rigid).
    omega_rel_input_mx = parent_state.omega_rel_output._mx
    alpha_rel_input_mx = parent_state.alpha_rel_output._mx

    # If child is a Joint, compose joint rotation onto the output frame
    # and add `rate · axis` to the output ω (and `θ̈ · axis` to α_rel).
    if isinstance(child, Joint):
        angle_mx = (child.angle._mx
                    if hasattr(child.angle, "_mx")
                    else ca.MX(float(child.angle)))
        rate_mx  = (child.rate._mx
                    if hasattr(child.rate,  "_mx")
                    else ca.MX(float(child.rate)))
        accel_mx = joint_angular_accels.get(child, ca.MX(0.0))

        # Joint axis given in input-frame local coords:
        axis_local_mx = ca.MX(list(child.axis))

        # Input → output rotation as both a quaternion (for anchor side)
        # and a rotation matrix (for body-frame side).
        q_input_from_output_mx = quat_from_axis_angle(
            axis_local_mx, angle_mx)
        R_input_from_output_mx = R_from_axis_angle(
            axis_local_mx, angle_mx)
        q_anchor_from_output_mx = quat_mul(
            q_anchor_from_input_mx, q_input_from_output_mx)
        q_anchor_from_output = Quat[WorldFrame, CraftFrame].from_mx(
            q_anchor_from_output_mx)
        R_craft_from_child_output_mx = ca.mtimes(
            R_craft_from_child_input_mx, R_input_from_output_mx)

        # Output ω = input ω + rate · axis_in_craft.
        axis_in_craft_mx = ca.mtimes(
            R_craft_from_child_input_mx, axis_local_mx)
        omega_increment_in_craft_mx = rate_mx * axis_in_craft_mx
        child_omega_output_in_craft_mx = (
            child_omega_input_in_craft._mx + omega_increment_in_craft_mx)
        child_omega_output_in_craft = Vec3[CraftFrame].from_mx(
            child_omega_output_in_craft_mx)

        # Relative ω/α gain the joint's own spin. The axis is fixed in the
        # input frame, which itself spins at ω_rel_input relative to the
        # body, hence the ω_rel_input × (θ̇·axis) term in α_rel.
        omega_rel_output_mx = omega_rel_input_mx + omega_increment_in_craft_mx
        alpha_rel_output_mx = (
            alpha_rel_input_mx
            + accel_mx * axis_in_craft_mx
            + ca.cross(omega_rel_input_mx, omega_increment_in_craft_mx))
    else:
        q_anchor_from_output = q_anchor_from_input
        R_craft_from_child_output_mx = R_craft_from_child_input_mx
        child_omega_output_in_craft = child_omega_input_in_craft
        omega_rel_output_mx = omega_rel_input_mx
        alpha_rel_output_mx = alpha_rel_input_mx

    R_craft_from_child_output = Mat3[CraftFrame, CraftFrame].from_mx(
        R_craft_from_child_output_mx)
    child_omega_rel_output = Vec3[CraftFrame].from_mx(omega_rel_output_mx)
    child_alpha_rel_output = Vec3[CraftFrame].from_mx(alpha_rel_output_mx)

    # ----- body-relative velocity / acceleration of the origin ----------
    # Rigid-chain recursion with the body frame as the fixed base, so these
    # isolate the joint-induced motion (zero when no joint lies above the
    # part). v̇|_body = v̇_parent|_body + ω_rel_parent × offset; the accel
    # adds the centripetal ω_rel×(ω_rel×offset) + tangential α_rel×offset.
    omega_rel_parent_mx = parent_state.omega_rel_output._mx
    alpha_rel_parent_mx = parent_state.alpha_rel_output._mx
    v_rel_child_mx = (parent_state.velocity_rel_body._mx
                      + ca.cross(omega_rel_parent_mx, offset_in_craft_mx))
    a_rel_child_mx = (parent_state.acceleration_rel_body._mx
                      + ca.cross(alpha_rel_parent_mx, offset_in_craft_mx)
                      + ca.cross(omega_rel_parent_mx,
                                 ca.cross(omega_rel_parent_mx,
                                          offset_in_craft_mx)))
    child_velocity_rel_body = Vec3[CraftFrame].from_mx(v_rel_child_mx)
    child_acceleration_rel_body = Vec3[CraftFrame].from_mx(a_rel_child_mx)

    # ----- acceleration at the child's mount point ---------------------
    # Body a / α come in as compile-time placeholder MX symbols (or
    # any caller-supplied expression). After the wrench-collection
    # phase the framework substitutes them with the real Newton-Euler
    # outputs, so a Part reading `ctx.acceleration_body` ends up
    # reading the current-tick value.
    #
    # Rotating-frame transport theorem: the absolute acceleration of a
    # point that also moves WITHIN the (rotating) body frame is the rigid
    # lever-arm transfer PLUS the Coriolis term 2·ω_body × v_rel and the
    # relative acceleration a_rel. For a part rigidly fixed to the body
    # (v_rel = a_rel = 0) this collapses to the original lever arm.
    lever_craft  = (body_angular_acceleration.cross(child_r_in_craft)
                    + body_angular_velocity.cross(
                        body_angular_velocity.cross(child_r_in_craft))
                    + body_angular_velocity.cross(child_velocity_rel_body)
                      * 2.0
                    + child_acceleration_rel_body)
    lever_world = body_orientation.apply(lever_craft)
    child_a_world = body_acceleration_world + lever_world

    # ----- ParentFrame views: motion relative to the immediate parent ---
    # The part's frame is its INPUT frame (= parent's OUTPUT frame). The
    # part's origin is rigidly fixed in the parent's OUTPUT frame, so the
    # only relative motion w.r.t. the parent's PartFrame comes from the
    # PARENT joint's rotation (zero if the parent is the root or a static
    # composite). All in ParentFrame (= parent input-frame) coords.
    transform_parentframe_mx = transform_mx     # parent OUTPUT coords
    if isinstance(parent_part, Joint):
        p_axis_mx = ca.MX(list(parent_part.axis))
        p_angle_mx = (parent_part.angle._mx
                      if hasattr(parent_part.angle, "_mx")
                      else ca.MX(float(parent_part.angle)))
        p_rate_mx = (parent_part.rate._mx
                     if hasattr(parent_part.rate, "_mx")
                     else ca.MX(float(parent_part.rate)))
        p_accel_mx = joint_angular_accels.get(parent_part, ca.MX(0.0))
        # transform is in the parent's OUTPUT coords; express in the
        # parent's INPUT (= ParentFrame) coords via the joint rotation.
        R_pin_from_pout = R_from_axis_angle(p_axis_mx, p_angle_mx)
        r_pf_mx = ca.mtimes(R_pin_from_pout, transform_parentframe_mx)
        omega_pf_mx = p_rate_mx * p_axis_mx
        alpha_pf_mx = p_accel_mx * p_axis_mx
        v_pf_mx = ca.cross(omega_pf_mx, r_pf_mx)
        a_pf_mx = (ca.cross(alpha_pf_mx, r_pf_mx)
                   + ca.cross(omega_pf_mx, ca.cross(omega_pf_mx, r_pf_mx)))
    else:
        r_pf_mx     = transform_parentframe_mx
        omega_pf_mx = ca.MX.zeros(3, 1)
        alpha_pf_mx = ca.MX.zeros(3, 1)
        v_pf_mx     = ca.MX.zeros(3, 1)
        a_pf_mx     = ca.MX.zeros(3, 1)

    # ----- absolute α of the part's (INPUT) frame, in craft coords ------
    # α_abs_input = α_body + α_rel_input + ω_body × ω_rel_input. Used for
    # the WorldFrame angular-acceleration view (the part frame is the
    # input frame, not the output frame).
    alpha_abs_input = (body_angular_acceleration
                       + Vec3[CraftFrame].from_mx(alpha_rel_input_mx)
                       + body_angular_velocity.cross(
                           Vec3[CraftFrame].from_mx(omega_rel_input_mx)))

    frame_views = _assemble_frame_views(
        body_orientation=body_orientation,
        origin_in_world=child_origin_in_world,
        velocity_origin=child_velocity_origin,
        acceleration_world=child_a_world,
        r_in_craft=child_r_in_craft,
        velocity_rel_body=child_velocity_rel_body,
        acceleration_rel_body=child_acceleration_rel_body,
        angular_velocity_input=child_omega_input_in_craft,
        omega_rel_input=Vec3[CraftFrame].from_mx(omega_rel_input_mx),
        alpha_rel_input=Vec3[CraftFrame].from_mx(alpha_rel_input_mx),
        alpha_abs_input=alpha_abs_input,
        pos_rel_parent=Vec3[ParentFrame].from_mx(r_pf_mx),
        velocity_rel_parent=Vec3[ParentFrame].from_mx(v_pf_mx),
        acceleration_rel_parent=Vec3[ParentFrame].from_mx(a_pf_mx),
        omega_rel_parent=Vec3[ParentFrame].from_mx(omega_pf_mx),
        alpha_rel_parent=Vec3[ParentFrame].from_mx(alpha_pf_mx),
    )

    return KinematicState(
        origin_in_world=child_origin_in_world,
        velocity_origin=child_velocity_origin,
        angular_velocity_input=child_omega_input_in_craft,
        angular_velocity_output=child_omega_output_in_craft,
        orientation_anchor_from_input=q_anchor_from_input,
        orientation_anchor_from_output=q_anchor_from_output,
        r_in_craft=child_r_in_craft,
        R_craft_from_input=R_craft_from_child_input,
        R_craft_from_output=R_craft_from_child_output,
        acceleration_world=child_a_world,
        velocity_rel_body=child_velocity_rel_body,
        acceleration_rel_body=child_acceleration_rel_body,
        omega_rel_output=child_omega_rel_output,
        alpha_rel_output=child_alpha_rel_output,
        frame_views=frame_views,
    )


