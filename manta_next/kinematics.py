"""Symbolic kinematic pass over a Craft's part tree.

Given the body's rigid-body state (position, orientation, velocity,
angular velocity), the body's symbolic acceleration / angular
acceleration, and every Joint's (angle, rate), compute for each part
its effective kinematic state — origin position, linear velocity at
the origin, angular velocity, lever-arm-lifted acceleration, the
part's body-frame position, and the rotation matrices that take its
input/output frames into body coords — all as CasADi MX expressions
composed through the joint chain. A flat craft (no nested joints)
reduces to "everything is the body's state offset by `part.transform`"
with identity rotations; deep nesting (gimbal: pan → tilt → camera)
composes naturally.

Body acceleration handling: an accelerometer's reading depends on
body a/α, which is the OUTPUT of Newton-Euler — reading them inside
`update()` (the wrench-collection phase, which feeds N-E) is
circular. The framework passes MX **placeholder** symbols for a/α
into this pass, lets parts emit outputs referencing them, then
substitutes the real Newton-Euler expressions into those outputs
after the wrench sum is known. Wrenches that would themselves depend
on a/α are rejected at compile time (no implicit-equation solver).

**Frame convention**:
  * World-frame fields (`origin_in_world`, `velocity_origin`,
    `orientation_anchor_from_*`) carry the part's actual absolute pose.
  * "In CraftFrame" fields (`angular_velocity_input/output`,
    `velocity_body_in_craft`, `gravity_in_craft`, `r_in_craft`,
    `R_craft_from_input/output`) are expressed in the **body's
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

from .ir.frames import WorldFrame, CraftFrame
from .ir.types import Mat3, Quat, Scalar, Vec3


# ---------------------------------------------------------------------------
# KinematicState
# ---------------------------------------------------------------------------

@dataclass
class KinematicState:
    """Kinematic state of one part — what its `TickContext` fields
    resolve to. All values are MX expressions; the Vec3/Quat/Mat3 are
    the typed manta_next IR wrappers.

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
        velocity_body_in_craft        — Vec3[CraftFrame]: body-frame
                                         linear velocity at the origin
                                         (= R_body^T · velocity_origin).
        gravity_in_craft              — Vec3[CraftFrame]: gravity sampled
                                         at the part's anchor position,
                                         expressed in body-frame coords.
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
    velocity_body_in_craft:         Any   # Vec3[CraftFrame]
    gravity_in_craft:               Any   # Vec3[CraftFrame]
    orientation_anchor_from_input:  Any   # Quat
    orientation_anchor_from_output: Any   # Quat
    r_in_craft:                     Any   # Vec3[CraftFrame]
    R_craft_from_input:             Any   # Mat3[CraftFrame, CraftFrame]
    R_craft_from_output:            Any   # Mat3[CraftFrame, CraftFrame]
    acceleration_world:            Any   # Vec3[WorldFrame]
    acceleration_body:              Any   # Vec3[CraftFrame]
    angular_acceleration:           Any   # Vec3[CraftFrame]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_from_axis_angle_mx(axis_local_mx, angle_mx):
    """(w, x, y, z) MX quaternion from a length-3 axis MX in the local
    frame and a scalar angle MX. Axis is normalized symbolically with
    an eps softener; passing a zero axis gives identity (within float
    rounding)."""
    n_sq = ca.dot(axis_local_mx, axis_local_mx) + 1.0e-30
    n    = ca.sqrt(n_sq)
    axis_unit = axis_local_mx / n
    half  = 0.5 * angle_mx
    w     = ca.cos(half)
    s     = ca.sin(half)
    return ca.vertcat(w, s * axis_unit[0], s * axis_unit[1], s * axis_unit[2])


def _quat_mul_mx(qa_mx, qb_mx):
    """Hamilton product of two MX quaternions (w, x, y, z)."""
    aw, ax, ay, az = qa_mx[0], qa_mx[1], qa_mx[2], qa_mx[3]
    bw, bx, by, bz = qb_mx[0], qb_mx[1], qb_mx[2], qb_mx[3]
    return ca.vertcat(
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

def kinematic_pass(root_part,
                   body_position,
                   body_orientation,
                   body_velocity,
                   body_angular_velocity,
                   gravity_field,
                   t,
                   *,
                   body_acceleration_world,
                   body_angular_acceleration) -> dict:
    """Walk `root_part`'s subtree, returning `{part: KinematicState}`.

    Joint angle/rate values are read directly off each Joint instance
    via `joint.angle._mx` / `joint.rate._mx` — assumes the framework has
    already rebound the State attributes to symbolic MX inputs before
    this is called.

    The root's state coincides with the body. Each child:
      * Position composes through `parent.orientation_output · transform`.
      * Velocity composes through standard rigid-body kinematics:
          v_child = v_parent_origin + ω_parent_output × (r_child - r_parent_origin).
      * Angular velocity of the part's input frame equals the parent's
        output frame ω; the output frame additionally gets `rate · axis`
        if the part is a Joint.
    """
    from .parts.articulation.joint import Joint
    from .parts.base import CompositePart

    # ----- root state (coincides with body) -----------------------------
    body_v_in_body = body_orientation.conjugate().apply(body_velocity)
    g_world_body  = gravity_field.state_at_sym(body_position, t)
    g_body         = body_orientation.conjugate().apply(g_world_body)

    eye3_mat = Mat3[CraftFrame, CraftFrame].from_mx(ca.MX.eye(3))
    zero_r   = Vec3[CraftFrame].from_mx(ca.MX.zeros(3, 1))

    # Root sees the body's own acceleration directly (no lever arm).
    # Body-frame version via R^T. Caller passes a / α as MX placeholder
    # symbols that get substituted to the real Newton-Euler outputs
    # after the wrench-collection phase.
    body_acceleration_body = body_orientation.conjugate().apply(
        body_acceleration_world)

    root_state = KinematicState(
        origin_in_world=body_position,
        velocity_origin=body_velocity,
        angular_velocity_input=body_angular_velocity,
        angular_velocity_output=body_angular_velocity,
        velocity_body_in_craft=body_v_in_body,
        gravity_in_craft=g_body,
        orientation_anchor_from_input=body_orientation,
        orientation_anchor_from_output=body_orientation,
        r_in_craft=zero_r,
        R_craft_from_input=eye3_mat,
        R_craft_from_output=eye3_mat,
        acceleration_world=body_acceleration_world,
        acceleration_body=body_acceleration_body,
        angular_acceleration=body_angular_acceleration,
    )

    states: dict = {root_part: root_state}

    def visit_children(parent, parent_state):
        if not isinstance(parent, CompositePart):
            return
        for child in parent.children:
            child_state = _compute_child_state(
                parent_state, child, body_orientation, gravity_field, t,
                body_acceleration_world=body_acceleration_world,
                body_angular_acceleration=body_angular_acceleration,
                body_angular_velocity=body_angular_velocity)
            states[child] = child_state
            visit_children(child, child_state)

    visit_children(root_part, root_state)
    return states


def _compute_child_state(parent_state: KinematicState, child,
                         body_orientation, gravity_field, t,
                         *,
                         body_acceleration_world,
                         body_angular_acceleration,
                         body_angular_velocity) -> KinematicState:
    from .parts.articulation.joint import Joint

    # Child's `transform` lives in parent's OUTPUT frame coords.
    transform_mx = ca.MX(list(child.transform))

    # ----- body-frame position composition ------------------------------
    # r_child_in_craft = r_parent_in_craft + R_craft_from_parent_output · transform.
    R_craft_from_parent_out_mx = parent_state.R_craft_from_output._mx
    r_parent_in_craft_mx = parent_state.r_in_craft._mx
    child_r_in_craft_mx = (r_parent_in_craft_mx
                           + ca.mtimes(R_craft_from_parent_out_mx, transform_mx))
    child_r_in_craft = Vec3[CraftFrame].from_mx(child_r_in_craft_mx)

    # Mount doesn't rotate, so child's INPUT frame = parent's OUTPUT.
    R_craft_from_child_input_mx = R_craft_from_parent_out_mx
    R_craft_from_child_input = Mat3[CraftFrame, CraftFrame].from_mx(
        R_craft_from_child_input_mx)

    # ----- position of child's origin in WorldFrame --------------------
    # Step 1: rotate transform from parent-output-frame coords to anchor
    # coords via the parent's output orientation.
    q_anchor_from_parent_output = parent_state.orientation_anchor_from_output
    offset_in_world_mx = _rotate_vec_by_quat_mx(
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

    # If child is a Joint, compose joint rotation onto the output frame
    # and add `rate · axis` to the output ω.
    if isinstance(child, Joint):
        angle_mx = (child.angle._mx
                    if hasattr(child.angle, "_mx")
                    else ca.MX(float(child.angle)))
        rate_mx  = (child.rate._mx
                    if hasattr(child.rate,  "_mx")
                    else ca.MX(float(child.rate)))

        # Joint axis given in input-frame local coords:
        axis_local_mx = ca.MX(list(child.axis))

        # Input → output rotation as both a quaternion (for anchor side)
        # and a rotation matrix (for body-frame side).
        q_input_from_output_mx = _quat_from_axis_angle_mx(
            axis_local_mx, angle_mx)
        R_input_from_output_mx = _R_from_axis_angle_mx(
            axis_local_mx, angle_mx)
        q_anchor_from_output_mx = _quat_mul_mx(
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
    else:
        q_anchor_from_output = q_anchor_from_input
        R_craft_from_child_output_mx = R_craft_from_child_input_mx
        child_omega_output_in_craft = child_omega_input_in_craft

    R_craft_from_child_output = Mat3[CraftFrame, CraftFrame].from_mx(
        R_craft_from_child_output_mx)

    # ----- body-frame quantities ----------------------------------------
    # ctx.velocity_body: body-frame coords of the linear velocity at the
    # child's origin. This is the "DVL reading" if a DVL were mounted on
    # this part — but expressed in the body's frame, not the part's own.
    child_velocity_body_in_craft = body_orientation.conjugate().apply(
        child_velocity_origin)

    # ctx.gravity: gravity at the child's anchor position, in body-frame
    # coords. Same convention as the body's root for backward compat.
    g_world_at_child = gravity_field.state_at_sym(child_origin_in_world, t)
    g_at_child_in_craft = body_orientation.conjugate().apply(g_world_at_child)

    # ----- acceleration at the child's mount point ---------------------
    # Body a / α come in as compile-time placeholder MX symbols (or
    # any caller-supplied expression). After the wrench-collection
    # phase the framework substitutes them with the real Newton-Euler
    # outputs, so a Part reading `ctx.acceleration_body` ends up
    # reading the current-tick lever-arm-lifted value.
    lever_craft  = (body_angular_acceleration.cross(child_r_in_craft)
                    + body_angular_velocity.cross(
                        body_angular_velocity.cross(child_r_in_craft)))
    lever_world = body_orientation.apply(lever_craft)
    child_a_world = body_acceleration_world + lever_world
    child_a_body   = body_orientation.conjugate().apply(child_a_world)

    return KinematicState(
        origin_in_world=child_origin_in_world,
        velocity_origin=child_velocity_origin,
        angular_velocity_input=child_omega_input_in_craft,
        angular_velocity_output=child_omega_output_in_craft,
        velocity_body_in_craft=child_velocity_body_in_craft,
        gravity_in_craft=g_at_child_in_craft,
        orientation_anchor_from_input=q_anchor_from_input,
        orientation_anchor_from_output=q_anchor_from_output,
        r_in_craft=child_r_in_craft,
        R_craft_from_input=R_craft_from_child_input,
        R_craft_from_output=R_craft_from_child_output,
        acceleration_world=child_a_world,
        acceleration_body=child_a_body,
        angular_acceleration=body_angular_acceleration,
    )


def _R_from_axis_angle_mx(axis_local_mx, angle_mx):
    """Rodrigues' rotation matrix for an MX axis + angle. Eps-soften the
    axis norm so a zero axis (or near-zero) doesn't crash."""
    if isinstance(axis_local_mx, list):
        axis_local_mx = ca.MX(axis_local_mx)
    # Force 3×1 shape.
    if axis_local_mx.shape == (3,):
        axis_local_mx = ca.reshape(axis_local_mx, 3, 1)
    n_sq = ca.mtimes(axis_local_mx.T, axis_local_mx) + 1.0e-30
    n    = ca.sqrt(n_sq)
    u    = axis_local_mx / n
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


# ---------------------------------------------------------------------------
# Vector rotation by quaternion (acting on MX)
# ---------------------------------------------------------------------------

def _rotate_vec_by_quat_mx(q_mx, v_mx):
    """Rotate a 3-vector `v_mx` by the quaternion `q_mx = (w, x, y, z)`.

    Uses the standard q · v · q* formula expressed in MX. Result is a
    length-3 MX expressing the rotated vector in the target frame.
    """
    w = q_mx[0]
    x = q_mx[1]
    y = q_mx[2]
    z = q_mx[3]
    # rotation matrix R(q):
    #   R = [ 1-2(y²+z²),  2(xy - zw),   2(xz + yw) ]
    #       [ 2(xy + zw),  1-2(x²+z²),   2(yz - xw) ]
    #       [ 2(xz - yw),  2(yz + xw),   1-2(x²+y²) ]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    vx, vy, vz = v_mx[0], v_mx[1], v_mx[2]
    return ca.vertcat(
        r00 * vx + r01 * vy + r02 * vz,
        r10 * vx + r11 * vy + r12 * vz,
        r20 * vx + r21 * vy + r22 * vz,
    )
