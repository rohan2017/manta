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
from ..base import CompositePart, Input, Parameter, Part, PartUpdate, State
from ...math.wrench import Wrench


_PASSIVE    = "passive"
_SATURATING = "saturating"
_MODES      = (_PASSIVE, _SATURATING)


def _offset_from(ancestor, descendant) -> np.ndarray:
    """Sum the static `transform` of every part on the path from
    `ancestor` (exclusive) down to `descendant` (inclusive). Used to
    compute the rest-pose offset of a child relative to a joint origin
    so the rotor's I_axial can apply a parallel-axis lift.

    This is the **at-zero-angle** rest offset — it ignores any nested
    joint rotations, since `I_axial` is the rotor's inertia about its
    own axis and the parallel-axis lift only depends on perpendicular
    distance, which is invariant under spins about the rotor's own
    axis when child masses are themselves axisymmetric. For
    cross-product moments-of-inertia that DO depend on inner-joint
    angle, the full symbolic body-aggregate handles it; `I_axial` is
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


class Joint(CompositePart):
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

    # ----- Child Masses ----------------------------------------------------

    def add(self, child) -> Part:
        """Attach a Part to the rotor.

        Children may be `Mass` parts (rotor inertia) or nested `Joint`s
        (e.g., pan–tilt gimbals). Child Masses can sit anywhere — the
        symbolic kinematic and inertia passes lift their position
        through the joint chain. Other leaf-part types (Thruster,
        sensors, drag surfaces) need their `update()` to rotate any
        input-frame quantities through `ctx.R_craft_from_input` before
        they will read correctly on a moving rotor; until they do,
        only `Mass` and `Joint` are accepted here.

        Returns the child (so chained construction reads naturally,
        matching `CompositePart.add`).
        """
        # Lazy-import to avoid a circular dependency at module load time.
        from ..structure.mass import Mass
        if not isinstance(child, (Mass, Joint)):
            raise TypeError(
                f"Joint.add: only Mass and Joint children supported, got "
                f"{type(child).__name__}. Leaf parts that emit wrenches "
                f"in their own input frame need to be made articulation-"
                f"aware (rotate by ctx.R_craft_from_input) before they "
                f"can ride on a moving rotor.")
        return super().add(child)

    # ----- Rotor I_axial computed from children ---------------------------

    @property
    def I_axial(self) -> float:
        """Rotor MOI about the joint's spin axis, with parallel-axis lifts.

        Walks the subtree below this joint, summing each Mass's
        diagonal MOI about the joint origin. The lift accounts for
        children offset from the joint axis (`r_⊥² · m`) but treats
        each child as if its frame is aligned with the joint's input
        frame at zero angle — which matches the gyroscopic-correction
        usage in `update()`, where I_axial only ever multiplies the
        rate-along-axis. A child Mass's own diagonal MOI rotates with
        any nested Joint above it, but the axial projection along the
        SAME axis is rotation-invariant for symmetric rotors and a
        good approximation otherwise."""
        axis = np.array(self.axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n <= 0.0:
            return 0.0
        axis_unit = axis / n

        total = 0.0
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
            total += float(axis_unit @ I_lifted @ axis_unit)
        return total

    # ----- update() --------------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        I_ax = self.I_axial
        # The Joint's spin axis is given in its INPUT-frame coords.
        # Rotate to body-frame via R_craft_from_input so the reaction
        # torque and gyroscopic term land in the same coords the body
        # aggregation expects. For a Joint mounted directly on root
        # this matrix is identity; for a nested inner Joint it carries
        # the outer joint's angle.
        axis_local_mx = ca.MX(list(self.axis))
        R_in_mx = ctx.R_craft_from_input._mx
        axis_mx = ca.mtimes(R_in_mx, axis_local_mx)

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

        # --- Wrench on the mount ---
        # Reaction torque (Newton's 3rd, only meaningful in saturating
        # mode where commanded torque is nonzero). Expressed in body
        # frame via the body-rotated axis.
        reaction_mx = axis_mx * (-tau_mx)
        # Gyroscopic correction: a spinning rotor that the body tries to
        # tilt off-axis pushes back via -ω × L_rotor. ω is the joint's
        # INPUT-frame ω (kinematic_pass gives us this directly in
        # body-frame coords) — for a top-level joint that's body ω; for
        # a nested joint it's the outer joint's output ω.
        L_rotor_mx = axis_mx * (I_ax * rate_mx)
        omega_mx = ctx.angular_velocity._mx
        # ca.cross handles 3-vec × 3-vec.
        tau_gyro_mx = -ca.cross(omega_mx, L_rotor_mx)
        torque_mx = reaction_mx + tau_gyro_mx
        torque = Vec3[CraftFrame].from_mx(torque_mx)

        # The Joint itself contributes no translational force: children
        # are in the craft's part walk and each Mass child applies its
        # own gravity via Mass.update(). Joint only emits the reaction
        # torque on the body plus the rotor's gyroscopic correction.
        zero_force = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))

        return PartUpdate(
            wrench=Wrench(force=zero_force, torque=torque),
            new_state={"angle": Scalar(new_angle_mx),
                       "rate":  Scalar(new_rate_mx)},
        )
