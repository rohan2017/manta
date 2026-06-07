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

The name `Motor` stays reserved for a future part that models real
motor dynamics (back-EMF, thermal limits, current/voltage curves) on
top of a RevoluteJoint.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from ..base import CompositePart, Input, Parameter, PartUpdate, State, unit_axis
from ...ir.wrench import Wrench


_PASSIVE    = "passive"
_SATURATING = "saturating"
_MODES      = (_PASSIVE, _SATURATING)


def _offset_from(ancestor, descendant) -> np.ndarray:
    """Sum the static `transform` of every part on the path from
    `ancestor` (exclusive) down to `descendant` (inclusive) — the
    rest-pose offset of a child relative to a joint origin, used for the
    parallel-axis lift in the numeric `I_joint_tensor` snapshot."""
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
    (axis, mode, damping, subtree-geometry snapshots, the no-op update);
    `RevoluteJoint` / `PrismaticJoint` declare their own DOF states and
    actuator input/limit.

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
        # The kinematic pass and the joint-space rows both assume a unit
        # axis — normalize once here (a zero axis is a config error).
        self.axis = unit_axis(self.axis,
                              who=f"{type(self).__name__} {name!r}",
                              what="axis")

    # ----- per-DOF-type hooks ----------------------------------------------

    def dof_state_names(self) -> tuple[str, str]:
        """(position-like, rate) state names of this joint's DOF —
        ("angle", "rate") for revolute, ("displacement", "rate") for
        prismatic. The central integrator, the kinematic/inertia passes,
        and the zero-pose snapshot all enumerate DOF states through this
        hook instead of hard-coding names."""
        raise NotImplementedError

    def _actuator_cmd(self):
        """The bound command Input (torque_cmd / force_cmd)."""
        raise NotImplementedError

    def _stall_limit(self) -> float:
        """Saturating-mode clamp magnitude (stall_torque / stall_force)."""
        raise NotImplementedError

    # ----- generalized applied force ----------------------------------------

    def applied_dof_force(self):
        """Actuator (clamped, saturating mode only) + viscous damping, as
        a raw MX scalar — the internal body↔subtree exchange entering
        this joint's generalized-force row. Its reaction on the body
        needs no explicit bookkeeping: it lives in the mass-matrix
        coupling of the joint-space solve."""
        _, rate_name = self.dof_state_names()
        rate_attr = getattr(self, rate_name)
        rate_mx = (rate_attr._mx if hasattr(rate_attr, "_mx")
                   else ca.MX(float(rate_attr)))
        f = -float(self.damping) * rate_mx
        if self.mode == _SATURATING:
            stall = float(self._stall_limit())
            cmd = self._actuator_cmd()
            cmd_mx = cmd._mx if hasattr(cmd, "_mx") else ca.MX(float(cmd))
            f = f + ca.fmin(ca.fmax(cmd_mx, -stall), stall)
        return f

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
            if descendant is self:
                continue
            m = float(getattr(descendant, "mass", 0.0) or 0.0)
            moi_diag = getattr(descendant, "moi", (0.0, 0.0, 0.0))
            I_own = np.diag([float(moi_diag[0]),
                             float(moi_diag[1]),
                             float(moi_diag[2])])
            r = _offset_from(self, descendant)
            I_lifted = I_own + m * (float(r @ r) * np.eye(3) - np.outer(r, r))
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
            if descendant is self:
                continue
            total += float(getattr(descendant, "mass", 0.0) or 0.0)
        return total

    # ----- update() --------------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        # The joint emits no wrench of its own and does not integrate
        # here. Its children (Mass, etc.) supply the external wrench; the
        # world tick's joint-space solve produces q̈ and a central
        # integrator advances the DOF states.
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

    def _actuator_cmd(self):
        return self.torque_cmd

    def _stall_limit(self) -> float:
        return self.stall_torque


class PrismaticJoint(ArticulatedJoint):
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
