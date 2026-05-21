"""Joint — 1-DOF revolute joint (mirrors legacy `Motor` in motor.hpp).

A `Joint` is a kinematic abstraction: one rotational degree of freedom
about a body-frame `axis`, with internal `angle` and `rate` state. It
hosts a list of `Mass` children whose inertial properties become the
joint's rotor. The joint produces:

  * Reaction torque on the parent body: `-clamped_torque · axis`.
  * Gyroscopic correction: `-ω_body × (I_axial · rate · axis)`.
  * Gravity on each child mass (lifted to the joint origin).

Modes:
  * "passive"     — no commanded torque; rotor free-spins under whatever
                    initial rate is set. The gyroscopic correction still
                    couples the rotor's angular momentum into the body.
  * "saturating"  — commanded torque clipped to `±stall_torque`. Beyond
                    that, the actuator stalls and only the clamped torque
                    is applied.

v1 restriction: child Mass parts must have `transform=(0,0,0)` (rotor
balanced about the joint origin). To place the assembly elsewhere on
the body, set the **Joint's** `transform`, not the children's. This
keeps the rolled-up MOI diagonal and lets the standard body aggregation
machinery work unchanged. Multi-DOF joints and unbalanced rotors land
in a future milestone.

The name `Joint` (vs. `Motor`) is intentional: `Motor` is reserved for a
future part that models real motor dynamics (back-EMF, thermal limits,
current/voltage curves) on top of a Joint.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ...ir.frames import CraftFrame
from ...ir.types import Scalar, Vec3
from ..base import Input, Parameter, Part, PartUpdate, State
from ...math.wrench import Wrench


_PASSIVE    = "passive"
_SATURATING = "saturating"
_MODES      = (_PASSIVE, _SATURATING)


class Joint(Part):
    """1-DOF revolute joint with an axial rotor (set of Mass children).

    Parameters:
        axis          — body-frame unit vector along the rotation axis.
                        Default (0, 0, 1).
        mode          — "passive" or "saturating". Default "passive".
        stall_torque  — saturating-mode torque clamp magnitude (N·m).
                        Ignored in passive mode. Default 1.0.

    Inputs:
        torque_cmd    — commanded torque about `axis`. Clamped to
                        ±stall_torque in saturating mode; ignored
                        entirely in passive mode.

    State:
        angle         — joint angle, rad.
        rate          — joint angular rate (rotor spin relative to body),
                        rad/s.
    """

    axis:          tuple = Parameter((0.0, 0.0, 1.0))
    mode:          str   = Parameter(_PASSIVE)
    stall_torque:  float = Parameter(1.0)
    torque_cmd:    float = Input(default=0.0)

    angle = State(init=0.0, manifold="R1")
    rate  = State(init=0.0, manifold="R1")

    def __init__(self, name: str, **overrides) -> None:
        mode = overrides.get("mode", _PASSIVE)
        if mode not in _MODES:
            raise ValueError(
                f"Joint {name!r}: mode must be one of {_MODES}, got {mode!r}")
        super().__init__(name, **overrides)
        self._children: list = []

    # ----- Child Masses ----------------------------------------------------

    def add(self, child) -> "Joint":
        """Attach a `Mass` child to the rotor. The child's mass and MOI
        contribute to the joint's `I_axial` and to the body's aggregated
        inertia. v1 requires `child.transform == (0, 0, 0)`."""
        # Lazy-import to avoid a circular dependency at module load time.
        from ..structure.mass import Mass
        if not isinstance(child, Mass):
            raise TypeError(
                f"Joint.add: only Mass children supported in v1, got "
                f"{type(child).__name__}")
        if tuple(child.transform) != (0.0, 0.0, 0.0):
            raise ValueError(
                f"Joint('{self.name}'): child Mass '{child.name}' has "
                f"nonzero transform {child.transform}. v1 requires children "
                f"at the joint origin; offset the Joint's transform instead.")
        self._children.append(child)
        return self

    @property
    def children(self) -> tuple:
        return tuple(self._children)

    # ----- Body aggregation: expose mass + moi like a regular Mass ---------

    @property
    def mass(self) -> float:
        """Sum of children masses. Read by Craft._aggregate_inertials."""
        return float(sum(float(c.mass) for c in self._children))

    @property
    def moi(self) -> tuple:
        """Children's combined diagonal MOI about the joint origin.
        Children must be at the joint origin (enforced in add()), so
        this is just the sum of their own MOIs."""
        Ixx = Iyy = Izz = 0.0
        for c in self._children:
            mi = c.moi
            Ixx += float(mi[0])
            Iyy += float(mi[1])
            Izz += float(mi[2])
        return (Ixx, Iyy, Izz)

    @property
    def I_axial(self) -> float:
        """Children's MOI about the joint axis, derived from `moi` + axis."""
        axis = np.array(self.axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n <= 0.0:
            return 0.0
        axis_unit = axis / n
        Ixx, Iyy, Izz = self.moi
        I_diag = np.diag([Ixx, Iyy, Izz])
        return float(axis_unit @ I_diag @ axis_unit)

    # ----- update() --------------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        I_ax = self.I_axial
        axis_v = Vec3[CraftFrame].constant(tuple(self.axis))
        axis_mx = axis_v._mx

        # Torque per mode. We work in MX so the saturating clamp can use
        # ca.fmin / ca.fmax for a branch-free symbolic clip.
        if self.mode == _PASSIVE:
            tau_mx = ca.MX(0.0)
        else:   # _SATURATING (validated at __init__)
            tau_in_mx = self.torque_cmd._mx
            stall = float(self.stall_torque)
            tau_mx = ca.fmin(ca.fmax(tau_in_mx, -stall), stall)

        # Joint dynamics. Skip the angular-velocity integration if the
        # rotor has no inertia (massless joint) — its angle just stays
        # frozen at whatever the user wrote.
        rate_mx  = self.rate._mx
        angle_mx = self.angle._mx
        dt_mx    = ctx.dt._mx
        if I_ax > 0.0:
            accel_mx = tau_mx / I_ax
        else:
            accel_mx = ca.MX(0.0)
        new_rate_mx  = rate_mx + accel_mx * dt_mx
        new_angle_mx = (angle_mx + rate_mx * dt_mx
                        + 0.5 * accel_mx * dt_mx * dt_mx)

        # --- Wrench on the body ---
        # Reaction torque (Newton's 3rd, only meaningful in saturating mode
        # where commanded torque is nonzero).
        reaction_mx = axis_mx * (-tau_mx)
        # Gyroscopic correction: a spinning rotor that the body tries to
        # tilt off-axis pushes back via -ω × L_rotor.
        L_rotor_mx = axis_mx * (I_ax * rate_mx)
        omega_mx = ctx.angular_velocity._mx
        # ca.cross handles 3-vec × 3-vec.
        tau_gyro_mx = -ca.cross(omega_mx, L_rotor_mx)
        torque_mx = reaction_mx + tau_gyro_mx
        torque = Vec3[CraftFrame].from_mx(torque_mx)

        # Gravity on rotor children (each Mass would emit m·g via update,
        # but they're not in craft._parts — so the Joint applies it
        # collectively here, at the joint origin).
        total_grav_mass = sum(
            float(c.mass) for c in self._children
            if getattr(c, "apply_gravity", True))
        force = ctx.gravity * total_grav_mass

        return PartUpdate(
            wrench=Wrench(force=force, torque=torque),
            new_state={"angle": Scalar(new_angle_mx),
                       "rate":  Scalar(new_rate_mx)},
        )
