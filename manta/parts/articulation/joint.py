"""Articulated joints — 1-DOF revolute and prismatic.

An `ArticulatedJoint` is a `CompositePart` with one mechanical degree of
freedom along/about an input-frame `axis`, hosting a subtree of children
that ride the moving side — any Part: `Mass` parts, nested joints
(pan–tilt gimbals), thrusters, and sensors. The symbolic kinematic and
inertia passes lift each child's position, velocity, acceleration, and
tensors through the joint chain (so a rotor's COM/inertia track the
joint angle, and a slider's COM tracks its displacement), and the
framework expresses each child's TickContext in its own moving frame and
rotates its emitted wrench back to the body — so a part needs no
awareness of being on a joint.

Two concrete DOF types:

  * `RevoluteJoint`  — rotation about `axis`; state (`angle`, `rate`),
                       commanded via `torque_cmd` (clamped to
                       `stall_torque` in saturating mode).
  * `PrismaticJoint` — translation along `axis`; state (`displacement`,
                       `rate`), commanded via `force_cmd` (clamped to
                       `stall_force` in saturating mode).

Dynamics live in `resolve()`, called by the framework's bottom-up wrench
cascade. The joint receives the total wrench from its subtree and splits
it along its motion subspace: a revolute peels the AXIAL TORQUE off to
drive the DOF (`θ̈ = (τ_axial + actuator + friction + Coriolis) /
I_axial`) and passes the rest — full force, perpendicular torque, and
the Newton-3rd reaction of the internal terms — up to the parent; a
prismatic peels the AXIAL FORCE off instead. Because the axial component
of the *external* generalized force drives the DOF, a **passive joint
moves under gravity** (a hanging bob gives `θ̈ = −(g/L)sinθ`; a vertical
slider falls at `g`). A revolute rotor's gyroscopic couple
`−ω_mount × (I_axial·rate·axis)` is applied at the body level (not
routed up the chain). `update()` itself is a no-op — children supply
the wrench, the cascade does the rest.

Approximation tier: the generalized inertia (`I_axial` / subtree mass)
uses the rest-pose subtree geometry, and the joint solve is per-joint —
the body-α mount feedback is coupled (handled in `world_tick`'s
body/rotor solve), but inter-joint inertial coupling for NESTED joints
is only approximate. A prismatic DOF additionally omits the mount's
*linear*-acceleration feedback (fixed-base tier). The planned
joint-space block solve removes both limitations.

Modes (both DOF types):
  * "passive"     — no commanded effort; the DOF responds to the axial
                    external generalized force (gravity, contact, …)
                    + friction.
  * "saturating"  — commanded effort clipped to the stall limit, applied
                    on top of the external axial term; its reaction
                    propagates to the parent.

The name `Motor` stays reserved for a future part that models real
motor dynamics (back-EMF, thermal limits, current/voltage curves) on
top of a RevoluteJoint.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ...ir.frames import CraftFrame, PartFrame
from ...ir.types import Vec3
from ..base import CompositePart, Input, Parameter, PartUpdate, State
from ...ir.wrench import Wrench
from ... import ir


_PASSIVE    = "passive"
_SATURATING = "saturating"
_MODES      = (_PASSIVE, _SATURATING)


def _offset_from(ancestor, descendant) -> np.ndarray:
    """Sum the static `transform` of every part on the path from
    `ancestor` (exclusive) down to `descendant` (inclusive). Used to
    compute the rest-pose offset of a child relative to a joint origin
    so the rotor's I_axial can apply a parallel-axis lift.

    This is the **at-zero-DOF** rest offset — it ignores any nested
    joint rotations/translations, since `I_axial` is the rotor's inertia
    about its own axis and the parallel-axis lift only depends on
    perpendicular distance, which is invariant under spins about the
    rotor's own axis when child masses are themselves axisymmetric. For
    cross-product moments-of-inertia that DO depend on inner-joint
    state, the full symbolic body-aggregate handles it; `I_axial` is
    a scalar used only inside the joint's own dynamics."""
    chain: list = []
    cur = descendant
    while cur is not None and cur is not ancestor:
        chain.append(cur)
        cur = cur.parent
    if cur is not ancestor:
        raise ValueError(
            f"_offset_from: '{descendant.name}' is not a descendant of "
            f"'{ancestor.name}'.")
    return sum((np.asarray(p.transform, dtype=float) for p in chain),
               start=np.zeros(3))


class ArticulatedJoint(CompositePart):
    """Base of the 1-DOF joint family. Holds everything DOF-type-agnostic
    (axis, mode, damping, subtree-geometry helpers, the no-op update);
    `RevoluteJoint` / `PrismaticJoint` declare their own DOF states,
    actuator input/limit, and `resolve()` projection.

    Parameters:
        axis          — input-frame unit vector along the DOF axis.
                        Default (0, 0, 1).
        mode          — "passive" or "saturating". Default "passive".
        damping       — viscous DOF friction coefficient. Default 0.
    """

    axis:          tuple = Parameter((0.0, 0.0, 1.0))
    mode:          str   = Parameter(_PASSIVE)
    damping:       float = Parameter(0.0)

    def __init__(self, name: str, **overrides) -> None:
        mode = overrides.get("mode", _PASSIVE)
        if mode not in _MODES:
            raise ValueError(
                f"{type(self).__name__} {name!r}: mode must be one of "
                f"{_MODES}, got {mode!r}")
        super().__init__(name, **overrides)

    # ----- per-DOF-type hooks ----------------------------------------------

    def dof_state_names(self) -> tuple[str, str]:
        """(position-like, rate) state names of this joint's DOF —
        ("angle", "rate") for revolute, ("displacement", "rate") for
        prismatic. The central integrator, the kinematic/inertia passes,
        and the zero-pose snapshot all enumerate DOF states through this
        hook instead of hard-coding names."""
        raise NotImplementedError

    def resolve(self, accum: Wrench, R_craft_from_input, omega_mount_c):
        """Split the accumulated subtree wrench at this joint — see the
        concrete classes. Returns `(dof_ddot, passup, gyro_torque)`."""
        raise NotImplementedError

    # ----- subtree geometry (shared) ---------------------------------------

    @property
    def I_joint_tensor(self) -> np.ndarray:
        """Full subtree inertia tensor (3×3) about the joint origin, in
        the joint's input-frame coords at rest pose.

        Walks the subtree below this joint, summing each Mass's diagonal
        MOI lifted to the joint origin via the parallel-axis theorem. The
        lift treats each child as if its frame is aligned with the joint's
        input frame at zero DOF — exact for a top-level joint, a good
        approximation for nested ones (a child's own diagonal MOI rotates
        with any inner joint above it). `I_axial` is just this tensor's
        projection onto the axis; revolute `update()` needs the full
        tensor for the Coriolis joint torque, whose off-axis components
        matter for non-axisymmetric or off-axis rotors."""
        total = np.zeros((3, 3))
        for descendant in self.walk():
            if descendant is self:
                continue
            m = float(getattr(descendant, "mass", 0.0) or 0.0)
            moi_diag = getattr(descendant, "moi", (0.0, 0.0, 0.0))
            I_own = np.diag([float(moi_diag[0]),
                             float(moi_diag[1]),
                             float(moi_diag[2])])
            r = _offset_from(self, descendant)
            # Parallel-axis lift about the joint origin (in joint input frame).
            I_lifted = I_own + m * (float(r @ r) * np.eye(3) - np.outer(r, r))
            total += I_lifted
        return total

    @property
    def I_axial(self) -> float:
        """Subtree MOI about the joint's axis: `axisᵀ · I_joint · axis`.

        For a revolute joint this is the scalar inertia driving the DOF
        dynamics and the rotor angular-momentum factor in the gyroscopic
        correction. Shares the rest-pose / symmetric-rotor approximation
        of `I_joint_tensor`; the axial projection is rotation-invariant
        for symmetric rotors."""
        axis = np.array(self.axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n <= 0.0:
            return 0.0
        axis_unit = axis / n
        return float(axis_unit @ self.I_joint_tensor @ axis_unit)

    @property
    def subtree_mass(self) -> float:
        """Total mass of the subtree riding the joint — the generalized
        inertia of a prismatic DOF."""
        total = 0.0
        for descendant in self.walk():
            if descendant is self:
                continue
            total += float(getattr(descendant, "mass", 0.0) or 0.0)
        return total

    def subtree_com_offset(self) -> np.ndarray:
        """Rest-pose subtree COM offset from the joint origin, in the
        joint's input-frame coords (mass-weighted `_offset_from` walk).
        Zero vector for a massless subtree."""
        m_total = 0.0
        weighted = np.zeros(3)
        for descendant in self.walk():
            if descendant is self:
                continue
            m = float(getattr(descendant, "mass", 0.0) or 0.0)
            if m <= 0.0:
                continue
            m_total += m
            weighted += m * _offset_from(self, descendant)
        if m_total <= 0.0:
            return np.zeros(3)
        return weighted / m_total

    # ----- update() --------------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        # The joint emits no wrench of its own and does not integrate here.
        # Its children (Mass, etc.) supply the external wrench; the
        # framework's bottom-up wrench cascade calls `self.resolve(...)` to
        # split that subtree wrench (axial component → the joint DOF, the
        # rest → the parent) and a central integrator advances the DOF.
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          new_state={})


class RevoluteJoint(ArticulatedJoint):
    """1-DOF revolute joint with an axial rotor (set of Mass children).

    Parameters:
        axis          — input-frame unit vector along the rotation axis.
                        Default (0, 0, 1).
        mode          — "passive" or "saturating". Default "passive".
        stall_torque  — saturating-mode torque clamp magnitude (N·m).
                        Ignored in passive mode. Default 1.0.
        damping       — viscous joint friction (N·m·s/rad). Default 0.

    Inputs:
        torque_cmd    — commanded torque about `axis`. Clamped to
                        ±stall_torque in saturating mode; ignored
                        entirely in passive mode.

    State:
        angle         — joint angle, rad.
        rate          — joint angular rate (rotor spin relative to body),
                        rad/s.
    """

    stall_torque: float = Parameter(1.0)
    torque_cmd:   float = Input(default=0.0)

    angle = State(init=0.0, manifold="R1")
    rate  = State(init=0.0, manifold="R1")

    def dof_state_names(self) -> tuple[str, str]:
        return ("angle", "rate")

    # ----- resolve(): the wrench split (called by the cascade) -------------

    def resolve(self, accum: Wrench, R_craft_from_input, omega_mount_c):
        """Split the accumulated subtree wrench at this joint.

        Works entirely in CraftFrame coords. `accum` is the total wrench
        from this joint's subtree, referenced about the joint origin.
        `R_craft_from_input` rotates the joint's input frame into craft
        coords; `omega_mount_c` is the mount's inertial angular velocity
        in craft coords.

        Returns `(theta_ddot_mx, passup, gyro_torque)`:
          * `theta_ddot_mx` — the joint's angular acceleration. The AXIAL
            component of `accum.torque` drives the DOF (so gravity on a
            hanging bob swings it), plus the actuator, viscous friction,
            and Coriolis terms, over the rotor's axial inertia.
          * `passup` — the wrench transmitted to the parent: the full
            force, the PERPENDICULAR torque (the constraint reaction), and
            the Newton-3rd reaction of the actuator + friction along the
            axis. The axial component of the external torque is NOT
            transmitted — a free/​frictionless axis absorbs it into the DOF.
          * `gyro_torque` — the rotor's gyroscopic correction
            `−ω_mount × (I_axial·rate·axis)` (a pure couple, CraftFrame).
            Applied at the BODY level (not routed up the joint chain),
            matching the legacy `Motor` / `rotor_angular_momentum` design.
        """
        R_mx = R_craft_from_input._mx
        axis_np = np.array(self.axis, dtype=float)
        axis_np = axis_np / np.linalg.norm(axis_np)
        axis_c = ca.mtimes(R_mx, ca.DM(axis_np.reshape(3, 1)))   # craft coords

        tau_mx   = accum.torque._mx
        rate_mx  = self.rate._mx
        I_ax     = self.I_axial

        # Axial vs perpendicular decomposition of the external torque.
        tau_axial = ca.dot(tau_mx, axis_c)
        tau_perp  = tau_mx - tau_axial * axis_c

        # Actuator torque (passive → 0; saturating → clamped command).
        if self.mode == _PASSIVE:
            tau_act = ca.MX(0.0)
        else:
            stall = float(self.stall_torque)
            tau_act = ca.fmin(ca.fmax(self.torque_cmd._mx, -stall), stall)

        # Viscous joint friction (resists rate).
        tau_damp = -float(self.damping) * rate_mx

        # Coriolis joint torque through the full rotor inertia, in craft
        # coords: τ_corio = −[ω_rotor × (I_joint·ω_rotor)]·axis. Zero for an
        # axisymmetric on-axis rotor (so a spun flywheel keeps its rate).
        omega_mount_mx = omega_mount_c._mx
        omega_rotor    = omega_mount_mx + rate_mx * axis_c
        I_joint_c      = ca.mtimes(R_mx, ca.mtimes(ca.DM(self.I_joint_tensor),
                                                   R_mx.T))
        tau_corio = -ca.dot(ca.cross(omega_rotor,
                                     ca.mtimes(I_joint_c, omega_rotor)),
                            axis_c)

        # DOF acceleration (rotor locked if it has no axial inertia).
        if I_ax > 1e-18:
            theta_ddot = (tau_axial + tau_act + tau_damp + tau_corio) / I_ax
        else:
            theta_ddot = ca.MX(0.0)

        # Rotor gyroscopic correction −ω_mount × L_rotor (a pure couple).
        # Applied at the body level, not routed up the chain.
        L_rotor  = (I_ax * rate_mx) * axis_c
        tau_gyro = -ca.cross(omega_mount_mx, L_rotor)

        # Pass-up torque: perpendicular constraint reaction + Newton-3rd of
        # the actuator, friction, AND the Coriolis joint torque along the
        # axis — each is an internal body↔rotor momentum exchange, so what
        # drives the DOF must react on the mount (the coupled body solve
        # in world_tick depends on this for axial-momentum consistency).
        # The axial external torque is absorbed by the DOF, not transmitted.
        passup_torque_mx = tau_perp \
            + (-(tau_act + tau_damp + tau_corio)) * axis_c
        passup = Wrench(
            force=accum.force,
            torque=ir.Vec3[CraftFrame].from_mx(passup_torque_mx),
        )
        gyro_torque = ir.Vec3[CraftFrame].from_mx(tau_gyro)
        return theta_ddot, passup, gyro_torque
