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

Concrete DOF types:

  * `RevoluteJoint`  — rotation about `axis`; state (`angle`, `rate`),
                       commanded via `torque_cmd` (clamped to
                       `stall_torque` in saturating mode).
  * `PrismaticJoint` — translation along `axis`; state (`displacement`,
                       `rate`), commanded via `force_cmd` (clamped to
                       `stall_force` in saturating mode).
  * `Motor` (`motor.py`) — a revolute DOF driven by a voltage-commanded
                       DC-motor electrical model (back-EMF torque-speed
                       rolloff, winding resistance, current limit,
                       gearbox) instead of a direct torque command.

The revolute *kinematics* (angle/rate state, the axis rotation the
kinematic and inertia passes compose) live in `RevoluteDOF`;
`RevoluteJoint` and `Motor` are alternative actuation models on top of
it. The framework's joint dispatch keys on `RevoluteDOF` /
`PrismaticJoint`, so a new actuation model is just another
`RevoluteDOF` subclass overriding `applied_dof_force()`.

Dynamics: the joint class supplies NO dynamics formulas of its own. The
world tick assembles the craft's full joint-space system — the
generalized mass matrix over [body ω; all joint rates], the Hamel bias,
and the virtual-work generalized forces (see `manta/tick/joint_space.py`)
— and solves body α and every q̈ together. Gyroscopic couples, Coriolis
joint torques, nested-gimbal inertia coupling, prismatic centrifugal
flinging, recoil, and the free-fall cancellation of uniform gravity all
emerge from that one solve. The only per-type dynamics surface here is
`applied_dof_force()` (actuator clamp + viscous damping — the internal
exchange that enters the joint's own generalized-force row).

Modes (both DOF types):
  * "passive"     — no commanded effort; the DOF responds to the axial
                    external generalized force (gravity, contact, …)
                    + friction.
  * "saturating"  — commanded effort clipped to the stall limit, applied
                    on top of the external axial term.

"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .._declarations import Input, Parameter, PartUpdate, State, unit_axis
from .._mounting import rest_pose_from
from .._trace import declared_attr, scalar_mx
from ..base import CompositePart

_PASSIVE    = "passive"
_SATURATING = "saturating"
_MODES      = (_PASSIVE, _SATURATING)


class ArticulatedJoint(CompositePart):
    """Base of the 1-DOF joint family. Holds everything DOF-type-agnostic
    (axis, damping, subtree-geometry snapshots, the no-op update);
    concrete DOF types declare their own states, and actuation models
    (`CommandedDOF`, `Motor`) layer their effort on top of the damping-
    only `applied_dof_force` here. A kinematic base like `RevoluteDOF`
    deliberately carries no actuation vocabulary — `mode`, commands, and
    stall limits exist only on classes where they are real.

    Parameters:
        axis          — input-frame unit vector along the DOF axis.
                        Default (0, 0, 1).
        damping       — viscous DOF friction coefficient. Default 0.
    """

    axis:          tuple = Parameter((0.0, 0.0, 1.0))
    damping:       float = Parameter(0.0)

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        # The kinematic pass and the joint-space rows both assume a unit
        # axis — normalize once here (a zero axis is a config error).
        self.axis = unit_axis(self.axis,
                              who=f"{type(self).__name__}({name!r})",
                              what="axis")

    # ----- per-DOF-type hooks ----------------------------------------------

    def dof_state_names(self) -> tuple[str, str]:
        """(position-like, rate) state names of this joint's DOF —
        ("angle", "rate") for revolute, ("displacement", "rate") for
        prismatic. The central integrator, the kinematic/inertia passes,
        and the zero-pose snapshot all enumerate DOF states through this
        hook instead of hard-coding names."""
        raise NotImplementedError

    # ----- generalized applied force ----------------------------------------

    def _dof_rate_mx(self) -> ca.MX:
        """The DOF rate state as a raw MX scalar (promotion-aware)."""
        _, rate_name = self.dof_state_names()
        return scalar_mx(getattr(self, rate_name))

    def applied_dof_force(self):
        """Viscous damping only, as a raw MX scalar — the internal
        body↔subtree exchange entering this joint's generalized-force
        row. Actuation models (`CommandedDOF`, `Motor`) call `super()`
        and add their effort on top. The reaction on the body needs no
        explicit bookkeeping: it lives in the mass-matrix coupling of
        the joint-space solve."""
        return -float(self.damping) * self._dof_rate_mx()

    # ----- subtree geometry (numeric rest-pose snapshots) -------------------

    @property
    def I_joint_tensor(self) -> np.ndarray:
        """Subtree inertia tensor (3×3) about the joint origin, in the
        joint's input-frame coords at rest pose — a numeric snapshot
        (zero nested DOFs) used only for the live-DOF test and
        introspection; the dynamics use the fully symbolic
        configuration-dependent mass matrix."""
        total = np.zeros((3, 3))
        for descendant in self.walk():
            if descendant is self or not descendant.contributes_inertia:
                continue
            m = float(declared_attr(descendant, "mass", 0.0))
            moi_diag = declared_attr(descendant, "moi", (0.0, 0.0, 0.0))
            I_own = np.diag([float(moi_diag[0]),
                             float(moi_diag[1]),
                             float(moi_diag[2])])
            # Rest pose relative to the joint origin, composed through
            # every intermediate mount_orientation: r for the parallel-
            # axis lift, R for expressing the descendant's diagonal moi
            # in the joint's input frame (R·I·Rᵀ). A rotated bracket's
            # inertia would otherwise land on the wrong axes — and
            # I_axial gates whether this DOF is live at all.
            r, R = rest_pose_from(self, descendant)
            I_lifted = (R @ I_own @ R.T
                        + m * (float(r @ r) * np.eye(3) - np.outer(r, r)))
            total += I_lifted
        return total

    @property
    def I_axial(self) -> float:
        """Rest-pose subtree MOI about the joint's axis:
        `axisᵀ · I_joint · axis` — a revolute DOF with zero axial inertia
        is locked out of the joint-space solve."""
        axis = np.array(self.axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n <= 0.0:
            return 0.0
        axis_unit = axis / n
        return float(axis_unit @ self.I_joint_tensor @ axis_unit)

    @property
    def subtree_mass(self) -> float:
        """Total mass of the subtree riding the joint — a prismatic DOF
        with zero subtree mass is locked out of the joint-space solve."""
        total = 0.0
        for descendant in self.walk():
            if descendant is self or not descendant.contributes_inertia:
                continue
            total += float(declared_attr(descendant, "mass", 0.0))
        return total

    # ----- update() --------------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        # The joint emits no wrench of its own and does not integrate here:
        # children (Mass, etc.) supply the external wrench, the world tick's
        # joint-space solve produces q̈, and a central integrator advances
        # the DOF states. The zero wrench itself comes from CompositePart —
        # this override exists for the RETURN TYPE. Joints are the parts
        # users subclass to add state (a rev counter on a Motor, say), and
        # such a subclass extends `super().update(ctx).new_state`, which a
        # bare Wrench has no room for.
        return PartUpdate(wrench=super().update(ctx), new_state={})


class CommandedDOF(ArticulatedJoint):
    """Direct-command actuation, shared by `RevoluteJoint` and
    `PrismaticJoint`: a `mode` (passive | saturating) and, in saturating
    mode, a command clamped to the stall limit, applied on top of the
    base damping.

    Parameters:
        mode          — "passive" or "saturating". Default "passive".
    """

    mode: str = Parameter(_PASSIVE)

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        # Post-super read so a subclass overriding the declared default
        # is validated too (a pre-super `overrides.get` would silently
        # validate the base default instead).
        mode = self.declared_value("mode")
        if mode not in _MODES:
            raise ValueError(
                f"{type(self).__name__}({name!r}): mode must be one of "
                f"{_MODES}, got {mode!r}")

    def _actuator_cmd(self):
        """The bound command Input (torque_cmd / force_cmd)."""
        raise NotImplementedError

    def _stall_limit(self) -> float:
        """Saturating-mode clamp magnitude (stall_torque / stall_force)."""
        raise NotImplementedError

    def applied_dof_force(self):
        f = super().applied_dof_force()
        if self.mode == _SATURATING:
            stall = float(self._stall_limit())
            cmd_mx = scalar_mx(self._actuator_cmd())
            f = f + ca.fmin(ca.fmax(cmd_mx, -stall), stall)
        return f


class RevoluteDOF(ArticulatedJoint):
    """The revolute *kinematic* DOF — rotation about `axis`, state
    (`angle`, `rate`). Carries no actuation model of its own; the
    framework's joint dispatch (kinematic pass, inertia rollup,
    rest-pose snapshot) keys on this class, so every subclass —
    `RevoluteJoint` (direct torque command) and `Motor` (DC electrical
    model) — inherits the exact same articulated dynamics and differs
    only in `applied_dof_force()`.

    State:
        angle         — joint angle, rad.
        rate          — joint angular rate (rotor spin relative to body),
                        rad/s.
    """

    angle = State(init=0.0, manifold="R1")
    rate  = State(init=0.0, manifold="R1")

    def dof_state_names(self) -> tuple[str, str]:
        return ("angle", "rate")


class RevoluteJoint(RevoluteDOF, CommandedDOF):
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

    def _actuator_cmd(self):
        return self.torque_cmd

    def _stall_limit(self) -> float:
        return self.stall_torque


class PrismaticJoint(CommandedDOF):
    """1-DOF prismatic (sliding) joint carrying a subtree of Mass children.

    Parameters:
        axis          — input-frame unit vector along the slide axis.
                        Default (0, 0, 1).
        mode          — "passive" or "saturating". Default "passive".
        stall_force   — saturating-mode force clamp magnitude (N).
                        Ignored in passive mode. Default 1.0.
        damping       — viscous slide friction (N·s/m). Default 0.

    Inputs:
        force_cmd     — commanded force along `axis`. Clamped to
                        ±stall_force in saturating mode; ignored
                        entirely in passive mode.

    State:
        displacement  — slide displacement along `axis`, m.
        rate          — slide rate (relative to the mount), m/s.
    """

    stall_force: float = Parameter(1.0)
    force_cmd:   float = Input(default=0.0)

    displacement = State(init=0.0, manifold="R1")
    rate         = State(init=0.0, manifold="R1")

    def dof_state_names(self) -> tuple[str, str]:
        return ("displacement", "rate")

    def _actuator_cmd(self):
        return self.force_cmd

    def _stall_limit(self) -> float:
        return self.stall_force
