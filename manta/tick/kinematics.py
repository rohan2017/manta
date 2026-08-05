"""Symbolic kinematic pass over a Craft's part tree — raw CasADi MX.

Given the body's rigid-body state (position, orientation, velocity,
angular velocity), the body's symbolic acceleration / angular
acceleration, and every Joint's (angle, rate, θ̈), compute for each part
its effective kinematic state — origin position, linear velocity at
the origin, angular velocity/acceleration, the part's body-frame
position, and the rotations that take its input/output frames into
body coords — composed through the joint chain. A flat craft (no
nested joints) reduces to "everything is the body's state offset by
`part.mount_offset`" with identity rotations; deep nesting (gimbal:
pan → tilt → camera) composes naturally.

This pass is **frame-blind**: like the EKF/codegen symbolic layer it
works on raw `ca.MX` only. The typed frame tags (`Vec3[F]`, `Quat`,
`Mat3`) are applied exactly once at the boundary where each part's
`TickContext` is built (`world_tick._typed_views`) — the layer part
authors actually touch. Tagging inside the pass would either lie (a
part-frame quantity tagged CraftFrame) or demand per-part frame
classes; raw MX is honest about what is and isn't checked.

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

**Frame convention** (names, since the values are untagged MX):
  * `*_world` fields carry the part's actual absolute pose, world coords.
  * "in craft" fields (`omega_input/output`, `r_in_craft`,
    `R_craft_from_input/output`) are expressed in the **body's CraftFrame
    coords** even for nested parts whose own frame differs by the joint
    chain. Wrench aggregation lifts each part's emitted wrench through
    `r_in_craft`; articulated parts that emit input-frame quantities
    (Joint axis, etc.) rotate them through `R_craft_from_input` inside
    their own `update()`.
  * A part's *input* frame is its mount (= parent's output frame); a
    Joint's *output* frame additionally rotates by the joint angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..ir._rotation import (
    R_from_axis_angle,
    quat_from_axis_angle,
    quat_mul,
    quat_to_rotmat,
    rotate_vec_by_quat,
)
from ..ir.frames import WorldFrame, CraftFrame, ParentFrame, PartFrame
from ..parts._trace import is_promoted


_ZERO3 = ca.MX.zeros(3, 1)


# ---------------------------------------------------------------------------
# KinematicState
# ---------------------------------------------------------------------------

@dataclass
class KinematicState:
    """Kinematic state of one part. All values are raw `ca.MX`.

    Fields:
        origin_world        — (3,1) world-frame position of the part's
                              mount origin.
        velocity_world      — (3,1) world-frame linear velocity at that
                              origin.
        omega_input         — (3,1) ω of the part's input frame, in
                              body-frame coords.
        omega_output        — (3,1) ω of the part's output frame. Equals
                              input for non-Joint parts.
        q_world_from_input  — (4,1) world orientation of the input frame.
        q_world_from_output — (4,1) same for the output frame.
        r_in_craft          — (3,1) body-frame position of the part's
                              origin. Equals `part.mount_offset` for a flat
                              craft; depends on joint angles when nested.
        r_out_in_craft / origin_out_world / velocity_out_world /
        velocity_rel_out_body / acceleration_rel_out_body
                            — the same quantities for the part's OUTPUT
                              frame origin, which children compose from.
                              Identical (same MX objects) to the input-
                              origin fields for every part except a
                              PrismaticJoint, whose output origin sits at
                              `displacement · axis` along the slide and
                              moves at `rate · axis` within the input
                              frame (the acceleration carries the d̈
                              placeholder).
        R_craft_from_input  — (3,3) rotates input-frame coords into
                              body-frame coords. Identity for parts
                              mounted directly on the craft root.
        R_craft_from_output — (3,3) same for the output frame (adds the
                              joint's axis-angle rotation).
        acceleration_world  — (3,1) absolute acceleration of the origin
                              (carries the a/α/θ̈ placeholders).
        velocity_rel_body / acceleration_rel_body / omega_rel_output /
        alpha_rel_output    — joint-induced motion relative to the body
                              frame, body coords (zero for the root and
                              any rigidly-mounted part).
        frame_views         — `{quantity: {Frame: (3,1) MX}}` for quantity
                              in {position, velocity, acceleration,
                              angular_velocity, angular_acceleration} and
                              Frame in {World, Craft, Parent, Part}.
                              X[F] is the quantity measured RELATIVE TO
                              frame F, in F's coords — X[PartFrame] is 0,
                              X[WorldFrame] the absolute value,
                              X[CraftFrame] the joint-induced motion
                              w.r.t. the craft (0 if rigidly mounted),
                              X[ParentFrame] motion w.r.t. the immediate
                              parent. `world_tick` wraps these in typed
                              `Vec3[F]` views for the part's TickContext.
    """

    origin_world:           Any
    velocity_world:         Any
    omega_input:            Any
    omega_output:           Any
    q_world_from_input:     Any
    q_world_from_output:    Any
    r_in_craft:             Any
    R_craft_from_input:     Any
    R_craft_from_output:    Any
    acceleration_world:     Any
    velocity_rel_body:      Any
    acceleration_rel_body:  Any
    omega_rel_output:       Any
    alpha_rel_output:       Any
    # Output-frame origin (what children compose from). Equal to the
    # input-origin fields except below a PrismaticJoint.
    r_out_in_craft:            Any
    origin_out_world:          Any
    velocity_out_world:        Any
    velocity_rel_out_body:     Any
    acceleration_rel_out_body: Any
    frame_views:            Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble_frame_views(*,
                          q_body,
                          # WorldFrame (absolute) quantities:
                          origin_world, velocity_world, acceleration_world,
                          # CraftFrame: r_in_craft + relative-to-craft motion:
                          r_in_craft, velocity_rel_body, acceleration_rel_body,
                          # part-frame (= input frame) angular, craft coords:
                          omega_input, omega_rel_input,
                          alpha_rel_input, alpha_abs_input,
                          # ParentFrame quantities (parent-frame coords):
                          pos_rel_parent, velocity_rel_parent,
                          acceleration_rel_parent, omega_rel_parent,
                          alpha_rel_parent) -> dict:
    """Bundle the per-frame views a part's TickContext exposes (raw MX).
    Each `X[F]` is the quantity relative to frame F, in F's coords. The
    WorldFrame angular quantities are the part-frame (input) ω/α rotated
    from craft coords into world via the body quaternion."""
    ang_vel_world = rotate_vec_by_quat(q_body, omega_input)
    ang_acc_world = rotate_vec_by_quat(q_body, alpha_abs_input)
    return {
        "position": {
            WorldFrame: origin_world,
            CraftFrame: r_in_craft,
            ParentFrame: pos_rel_parent,
            PartFrame: _ZERO3,
        },
        "velocity": {
            WorldFrame: velocity_world,
            CraftFrame: velocity_rel_body,
            ParentFrame: velocity_rel_parent,
            PartFrame: _ZERO3,
        },
        "acceleration": {
            WorldFrame: acceleration_world,
            CraftFrame: acceleration_rel_body,
            ParentFrame: acceleration_rel_parent,
            PartFrame: _ZERO3,
        },
        "angular_velocity": {
            WorldFrame: ang_vel_world,
            CraftFrame: omega_rel_input,
            ParentFrame: omega_rel_parent,
            PartFrame: _ZERO3,
        },
        "angular_acceleration": {
            WorldFrame: ang_acc_world,
            CraftFrame: alpha_rel_input,
            ParentFrame: alpha_rel_parent,
            PartFrame: _ZERO3,
        },
    }


def _attr_mx(value) -> ca.MX:
    """A Joint angle/rate attribute: a bound `Scalar` symbol during a
    trace, a plain float outside one."""
    return value._mx if is_promoted(value) else ca.MX(float(value))


def _attr_quat(value) -> ca.MX:
    """A static mount `orientation`: a bound SO3 symbol when promoted
    (mount-misalignment sysid), else the declared wxyz constant."""
    if is_promoted(value):
        return value._mx
    return ca.MX(list(float(v) for v in value))


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
                   joint_dof_accels=None) -> dict:
    """Walk `root_part`'s subtree, returning `{part: KinematicState}`.

    All arguments are raw MX: positions/velocities (3,1), the body
    orientation quaternion (4,1) world←craft. Joint angle/rate values are
    read directly off each Joint instance via `joint.angle` / `joint.rate`
    — assumes an active trace has bound the State attributes to symbols.
    A Joint's angular acceleration θ̈ is NOT a state slot, so it is
    supplied separately via `joint_dof_accels` (`{joint: MX scalar}`);
    like the body's a/α these are compile-time placeholders that the
    caller substitutes with the real `(new_rate − rate)/dt` after
    `Joint.update()` runs. Joints absent from the dict take θ̈ = 0 — fine
    for the velocity/position outputs, which don't use it.

    The root's state coincides with the body. Each child:
      * Position composes through `parent.q_world_from_output · mount_offset`.
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
    from ..parts.base import CompositePart

    if joint_dof_accels is None:
        joint_dof_accels = {}

    eye3 = ca.MX.eye(3)

    # ----- root state (coincides with body) -----------------------------
    root_state = KinematicState(
        origin_world=body_position,
        velocity_world=body_velocity,
        omega_input=body_angular_velocity,
        omega_output=body_angular_velocity,
        q_world_from_input=body_orientation,
        q_world_from_output=body_orientation,
        r_in_craft=_ZERO3,
        R_craft_from_input=eye3,
        R_craft_from_output=eye3,
        acceleration_world=body_acceleration_world,
        velocity_rel_body=_ZERO3,
        acceleration_rel_body=_ZERO3,
        omega_rel_output=_ZERO3,
        alpha_rel_output=_ZERO3,
        r_out_in_craft=_ZERO3,
        origin_out_world=body_position,
        velocity_out_world=body_velocity,
        velocity_rel_out_body=_ZERO3,
        acceleration_rel_out_body=_ZERO3,
        frame_views=_assemble_frame_views(
            q_body=body_orientation,
            origin_world=body_position,
            velocity_world=body_velocity,
            acceleration_world=body_acceleration_world,
            r_in_craft=_ZERO3,
            velocity_rel_body=_ZERO3,
            acceleration_rel_body=_ZERO3,
            omega_input=body_angular_velocity,
            omega_rel_input=_ZERO3,
            alpha_rel_input=_ZERO3,
            alpha_abs_input=body_angular_acceleration,
            # Root has no parent — ParentFrame views coincide with the
            # craft (zero relative motion).
            pos_rel_parent=_ZERO3,
            velocity_rel_parent=_ZERO3,
            acceleration_rel_parent=_ZERO3,
            omega_rel_parent=_ZERO3,
            alpha_rel_parent=_ZERO3,
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
                joint_dof_accels=joint_dof_accels)
            states[child] = child_state
            visit_children(child, child_state)

    visit_children(root_part, root_state)
    return states


def _compute_child_state(parent_state: KinematicState, parent_part, child,
                         q_body,
                         *,
                         body_acceleration_world,
                         body_angular_acceleration,
                         body_angular_velocity,
                         joint_dof_accels) -> KinematicState:
    from ..parts.articulation.joint import PrismaticJoint, RevoluteDOF

    # Child's `mount_offset` lives in parent's OUTPUT frame coords. A promoted
    # (tunable) mount_offset reads as a trace-bound IR value — keep the symbol.
    tr_attr = child.mount_offset
    mount_offset = (tr_attr._mx if is_promoted(tr_attr)
                    else ca.MX(list(tr_attr)))

    # ----- body-frame position composition ------------------------------
    # r_child_in_craft = r_parent_out_in_craft
    #                    + R_craft_from_parent_output · mount_offset.
    # The parent's OUTPUT origin differs from its own origin only below a
    # PrismaticJoint (slid by `displacement · axis`).
    offset_in_craft = parent_state.R_craft_from_output @ mount_offset
    r_in_craft = parent_state.r_out_in_craft + offset_in_craft

    # Static mount rotation: the child's INPUT frame is the parent's
    # OUTPUT frame turned by the child's own `orientation` (part axes →
    # parent output axes). Identity for almost every part, so the fast
    # path skips the compose entirely rather than multiplying by I on
    # every part of every tick.
    if child.mounted_upright:
        R_craft_from_input = parent_state.R_craft_from_output
        q_world_from_input = parent_state.q_world_from_output
    else:
        q_mount = _attr_quat(child.mount_orientation)
        R_craft_from_input = (parent_state.R_craft_from_output
                              @ quat_to_rotmat(q_mount))
        q_world_from_input = quat_mul(
            parent_state.q_world_from_output, q_mount)

    # ----- position + velocity of child's origin, world coords ---------
    offset_world = rotate_vec_by_quat(
        parent_state.q_world_from_output, mount_offset)
    origin_world = parent_state.origin_out_world + offset_world
    # v_child = v_parent_out_origin + ω_parent_output (world) × offset_world.
    omega_parent_world = rotate_vec_by_quat(
        q_body, parent_state.omega_output)
    velocity_world = (parent_state.velocity_out_world
                      + ca.cross(omega_parent_world, offset_world))

    # ----- child input-frame ω + orientation ----------------------------
    # (q_world_from_input was composed with the mount rotation above.)
    omega_input = parent_state.omega_output
    # Body-relative ω/α of the INPUT frame equal the parent's OUTPUT-frame
    # relative values (the mount is rigid).
    omega_rel_input = parent_state.omega_rel_output
    alpha_rel_input = parent_state.alpha_rel_output

    # If child is a revolute joint, compose the joint rotation onto the
    # output frame and add `rate · axis` to the output ω (and `θ̈ · axis`
    # to α_rel).
    if isinstance(child, RevoluteDOF):
        angle = _attr_mx(child.angle)
        rate  = _attr_mx(child.rate)
        accel = joint_dof_accels.get(child, ca.MX(0.0))
        axis_local = ca.MX(list(child.axis))

        q_world_from_output = quat_mul(
            q_world_from_input, quat_from_axis_angle(axis_local, angle))
        R_craft_from_output = (
            R_craft_from_input @ R_from_axis_angle(axis_local, angle))

        # Output ω = input ω + rate · axis_in_craft.
        axis_in_craft = R_craft_from_input @ axis_local
        omega_increment = rate * axis_in_craft
        omega_output = omega_input + omega_increment

        # Relative ω/α gain the joint's own spin. The axis is fixed in the
        # input frame, which itself spins at ω_rel_input relative to the
        # body, hence the ω_rel_input × (θ̇·axis) term in α_rel.
        omega_rel_output = omega_rel_input + omega_increment
        alpha_rel_output = (alpha_rel_input
                            + accel * axis_in_craft
                            + ca.cross(omega_rel_input, omega_increment))
    else:
        q_world_from_output = q_world_from_input
        R_craft_from_output = R_craft_from_input
        omega_output = omega_input
        omega_rel_output = omega_rel_input
        alpha_rel_output = alpha_rel_input

    # ----- body-relative velocity / acceleration of the origin ----------
    # Rigid-chain recursion with the body frame as the fixed base, so these
    # isolate the joint-induced motion (zero when no joint lies above the
    # part). v̇|_body = v̇_parent_out|_body + ω_rel_parent × offset; the accel
    # adds the centripetal ω_rel×(ω_rel×offset) + tangential α_rel×offset.
    # The parent OUT fields already carry any prismatic sliding terms.
    omega_rel_parent = parent_state.omega_rel_output
    alpha_rel_parent = parent_state.alpha_rel_output
    velocity_rel_body = (parent_state.velocity_rel_out_body
                         + ca.cross(omega_rel_parent, offset_in_craft))
    acceleration_rel_body = (
        parent_state.acceleration_rel_out_body
        + ca.cross(alpha_rel_parent, offset_in_craft)
        + ca.cross(omega_rel_parent,
                   ca.cross(omega_rel_parent, offset_in_craft)))

    # ----- acceleration at the child's mount point ---------------------
    # Body a / α come in as compile-time placeholder MX symbols. After the
    # wrench-collection phase the framework substitutes them with the real
    # Newton-Euler outputs, so a Part reading ctx.acceleration ends up
    # reading the current-tick value.
    #
    # Rotating-frame transport theorem: the absolute acceleration of a
    # point that also moves WITHIN the (rotating) body frame is the rigid
    # lever-arm transfer PLUS the Coriolis term 2·ω_body × v_rel and the
    # relative acceleration a_rel. For a part rigidly fixed to the body
    # (v_rel = a_rel = 0) this collapses to the original lever arm.
    lever_craft = (ca.cross(body_angular_acceleration, r_in_craft)
                   + ca.cross(body_angular_velocity,
                              ca.cross(body_angular_velocity, r_in_craft))
                   + 2.0 * ca.cross(body_angular_velocity, velocity_rel_body)
                   + acceleration_rel_body)
    acceleration_world = (body_acceleration_world
                          + rotate_vec_by_quat(q_body, lever_craft))

    # ----- output-frame origin (what THIS child's children compose from) -
    # Identical to the input origin for everything except a prismatic
    # joint, whose output origin sits at `displacement · axis` along the
    # slide and moves at `rate · axis` within the (possibly spinning)
    # input frame. The transport terms below are the offset's lever
    # (α_rel×s + ω_rel×(ω_rel×s)) plus the sliding Coriolis
    # 2·ω_rel×(ḋ·â) and the relative slide acceleration d̈·â (placeholder).
    if isinstance(child, PrismaticJoint):
        disp   = _attr_mx(child.displacement)
        d_rate = _attr_mx(child.rate)
        d_acc  = joint_dof_accels.get(child, ca.MX(0.0))
        # `axis` is unit by the joint constructor's unit_axis invariant.
        axis_local = ca.DM(
            np.asarray(child.axis, dtype=float).reshape(3, 1))
        axis_c = R_craft_from_input @ axis_local        # craft coords
        slide_c = disp * axis_c
        axis_w = rotate_vec_by_quat(q_world_from_input, axis_local)
        omega_input_world = rotate_vec_by_quat(q_body, omega_input)

        r_out_in_craft = r_in_craft + slide_c
        origin_out_world = origin_world + disp * axis_w
        velocity_out_world = (velocity_world
                              + ca.cross(omega_input_world, disp * axis_w)
                              + d_rate * axis_w)
        velocity_rel_out_body = (velocity_rel_body
                                 + ca.cross(omega_rel_input, slide_c)
                                 + d_rate * axis_c)
        acceleration_rel_out_body = (
            acceleration_rel_body
            + ca.cross(alpha_rel_input, slide_c)
            + ca.cross(omega_rel_input,
                       ca.cross(omega_rel_input, slide_c))
            + 2.0 * ca.cross(omega_rel_input, d_rate * axis_c)
            + d_acc * axis_c)
    else:
        r_out_in_craft = r_in_craft
        origin_out_world = origin_world
        velocity_out_world = velocity_world
        velocity_rel_out_body = velocity_rel_body
        acceleration_rel_out_body = acceleration_rel_body

    # ----- ParentFrame views: motion relative to the immediate parent ---
    # The part's frame is its INPUT frame (= parent's OUTPUT frame). The
    # part's origin is rigidly fixed in the parent's OUTPUT frame, so the
    # only relative motion w.r.t. the parent's PartFrame comes from the
    # PARENT joint's rotation (zero if the parent is the root or a static
    # composite). All in ParentFrame (= parent input-frame) coords.
    if isinstance(parent_part, RevoluteDOF):
        p_axis  = ca.MX(list(parent_part.axis))
        p_angle = _attr_mx(parent_part.angle)
        p_rate  = _attr_mx(parent_part.rate)
        p_accel = joint_dof_accels.get(parent_part, ca.MX(0.0))
        # mount_offset is in the parent's OUTPUT coords; express in the
        # parent's INPUT (= ParentFrame) coords via the joint rotation.
        r_pf = R_from_axis_angle(p_axis, p_angle) @ mount_offset
        omega_pf = p_rate * p_axis
        alpha_pf = p_accel * p_axis
        v_pf = ca.cross(omega_pf, r_pf)
        a_pf = (ca.cross(alpha_pf, r_pf)
                + ca.cross(omega_pf, ca.cross(omega_pf, r_pf)))
    elif isinstance(parent_part, PrismaticJoint):
        # `axis` is unit by the joint constructor's unit_axis invariant.
        p_axis = ca.DM(
            np.asarray(parent_part.axis, dtype=float).reshape(3, 1))
        p_disp = _attr_mx(parent_part.displacement)
        p_rate = _attr_mx(parent_part.rate)
        p_accel = joint_dof_accels.get(parent_part, ca.MX(0.0))
        # Output coords = input coords (no rotation); the child rode out
        # along the slide, so it translates within the ParentFrame.
        r_pf = mount_offset + p_disp * p_axis
        v_pf = p_rate * p_axis
        a_pf = p_accel * p_axis
        omega_pf, alpha_pf = _ZERO3, _ZERO3
    else:
        r_pf, v_pf, a_pf = mount_offset, _ZERO3, _ZERO3
        omega_pf, alpha_pf = _ZERO3, _ZERO3

    # ----- absolute α of the part's (INPUT) frame, in craft coords ------
    # α_abs_input = α_body + α_rel_input + ω_body × ω_rel_input. Used for
    # the WorldFrame angular-acceleration view (the part frame is the
    # input frame, not the output frame).
    alpha_abs_input = (body_angular_acceleration
                       + alpha_rel_input
                       + ca.cross(body_angular_velocity, omega_rel_input))

    frame_views = _assemble_frame_views(
        q_body=q_body,
        origin_world=origin_world,
        velocity_world=velocity_world,
        acceleration_world=acceleration_world,
        r_in_craft=r_in_craft,
        velocity_rel_body=velocity_rel_body,
        acceleration_rel_body=acceleration_rel_body,
        omega_input=omega_input,
        omega_rel_input=omega_rel_input,
        alpha_rel_input=alpha_rel_input,
        alpha_abs_input=alpha_abs_input,
        pos_rel_parent=r_pf,
        velocity_rel_parent=v_pf,
        acceleration_rel_parent=a_pf,
        omega_rel_parent=omega_pf,
        alpha_rel_parent=alpha_pf,
    )

    return KinematicState(
        origin_world=origin_world,
        velocity_world=velocity_world,
        omega_input=omega_input,
        omega_output=omega_output,
        q_world_from_input=q_world_from_input,
        q_world_from_output=q_world_from_output,
        r_in_craft=r_in_craft,
        R_craft_from_input=R_craft_from_input,
        R_craft_from_output=R_craft_from_output,
        acceleration_world=acceleration_world,
        velocity_rel_body=velocity_rel_body,
        acceleration_rel_body=acceleration_rel_body,
        omega_rel_output=omega_rel_output,
        alpha_rel_output=alpha_rel_output,
        r_out_in_craft=r_out_in_craft,
        origin_out_world=origin_out_world,
        velocity_out_world=velocity_out_world,
        velocity_rel_out_body=velocity_rel_out_body,
        acceleration_rel_out_body=acceleration_rel_out_body,
        frame_views=frame_views,
    )
