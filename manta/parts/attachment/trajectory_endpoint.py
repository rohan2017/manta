"""TrajectoryEndpoint — slew a craft along a prescribed pose trajectory.

A `TrajectoryEndpoint` exerts a full SE(3) spring-damper wrench (force +
torque) that drags its craft toward a reference pose ``(p_ref(t),
q_ref(t))`` and reference twist ``(v_ref(t), ω_ref(t))`` supplied as a
symbolic function of the world clock. It is the "playback" actuator: bolt
it to an otherwise actuator-free craft and the craft *follows the
trajectory*, so the rest of the world (cameras, magnetometers, field
sources riding the craft) generates fake-but-consistent sensor data along
that path.

It is NOT a realistic controller — it applies whatever idealized wrench
the spring law asks for, with no actuator authority limit. That is the
point: a perfect-tracking ghost for data generation and scenario scripting.
For *realistic* reference tracking with a craft's real actuators, use the
LQR machinery against a reference instead.

Control law (world frame), with the part contractually mounted at the
craft origin (``mount_offset=(0,0,0)``, on the root — so ``PartFrame ≡
CraftFrame``)::

    F_world = kp_pos·(p_ref − p) + kd_pos·(v_ref − v)   [+ mass·a_ref − mass·g]
    τ_world = kp_att·(q_ref ⊟ q) + kd_att·(ω_ref − ω)

The bracketed feedforward is enabled by passing ``mass=`` (the craft's
mass): it cancels gravity and tracks an accelerating trajectory exactly,
so the spring only has to mop up residuals — "perfect" playback with
modest gains. ``q_ref ⊟ q`` is the SO(3) box-minus (a rotation vector).
The wrench is rotated into the part frame for the framework, exactly as
`Mass` does with gravity.

Reference trajectories are plain Python callables ``f(t) -> Trajectory
Sample`` evaluated with the *symbolic* clock ``ctx.t``, so the path is
baked into the compiled tick (no per-step plumbing). `LinearTrajectory`
and `hover` cover the common analytic cases; write your own callable for
anything closed-form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ...fields import GravityField, gravity_at
from ...ir.frames import PartFrame, WorldFrame
from ...ir.manifold import SO3Manifold
from ...ir.types import Quat, Vec3
from ...ir.wrench import Wrench
from .._declarations import Output, Parameter, PartUpdate
from ..base import Part, RootPart


@dataclass
class TrajectorySample:
    """One instant of a reference trajectory, in world coordinates.

    Only `position` is required; the rest default to a held attitude and
    zero twist/acceleration. All values are symbolic IR built from the
    clock passed to the trajectory callable.

    Frames:
      * ``position`` / ``velocity`` / ``acceleration`` — ``Vec3[WorldFrame]``.
      * ``orientation`` — ``Quat[WorldFrame, *]``; the to-frame is
        reinterpreted as the part frame (the endpoint rides the craft
        root), so authoring it as ``Quat[WorldFrame, CraftFrame]`` reads
        naturally.
      * ``angular_velocity`` — ``Vec3[WorldFrame]`` (the world-frame body
        rate, matching ``ctx.angular_velocity[WorldFrame]``).
    """

    position:         Vec3
    orientation:      Optional[Quat] = None
    velocity:         Optional[Vec3] = None
    angular_velocity: Optional[Vec3] = None
    acceleration:     Optional[Vec3] = None


class TrajectoryEndpoint(Part):
    """Spring-damper that slews its craft along a reference pose path.

    Parameters:
        trajectory — callable ``f(t) -> TrajectorySample`` taking the
                     symbolic clock (a `Scalar`) and returning the
                     reference at that time. Required.
        kp_pos, kd_pos — position spring / damping gains (N per m, N per
                     m/s). With ``mass`` set, critical damping is
                     ``kd_pos = 2·sqrt(kp_pos·mass)``.
        kp_att, kd_att — attitude spring / damping gains (N·m per rad,
                     N·m per rad/s).
        mass       — craft mass (kg). When > 0, enables gravity + linear-
                     acceleration feedforward for tight tracking. 0 (the
                     default) is a pure spring — fine for light craft and
                     a zero-gravity world, but it will droop under gravity.

    Mount it on the craft root at the origin (the default transform). An
    off-origin mount would inject a force×lever-arm torque that corrupts
    the attitude channel.
    """

    trajectory: Callable = Parameter(None)
    kp_pos: float = Parameter(40.0)
    kd_pos: float = Parameter(12.0)
    kp_att: float = Parameter(8.0)
    kd_att: float = Parameter(4.0)
    mass:   float = Parameter(0.0)

    # World-frame position tracking error (p_ref − p), for telemetry.
    pos_error = Output()

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        if self.trajectory is None or not callable(self.trajectory):
            raise ValueError(
                f"{type(self).__name__}({name!r}): `trajectory` must be a "
                f"callable f(t) -> TrajectorySample.")
        if float(self.mass) < 0.0:
            raise ValueError(
                f"{type(self).__name__}({name!r}): mass must be >= 0.")

    def on_world_finalize(self, world, craft) -> None:
        # The control law treats PartFrame ≡ CraftFrame (the reference
        # quaternion is reinterpreted into PartFrame in update()). A
        # nested endpoint (under a joint/composite) silently breaks that
        # — reject it at finalize, before any tracing.
        if not isinstance(self.parent, RootPart):
            raise ValueError(
                f"{type(self).__name__}({self.name!r}): must be mounted "
                f"directly on the craft root (got parent "
                f"'{self.parent.name if self.parent else None}'). The "
                f"SE(3) spring law assumes PartFrame ≡ CraftFrame.")

    def update(self, ctx) -> PartUpdate:
        sample = self.trajectory(ctx.t)
        if not isinstance(sample, TrajectorySample):
            raise TypeError(
                f"{type(self).__name__}({self.name!r}): trajectory(t) must "
                f"return a TrajectorySample, got {type(sample).__name__}.")

        # --- references, with held-attitude / zero-twist defaults --------
        zero_w = Vec3[WorldFrame].constant((0.0, 0.0, 0.0))
        p_ref = sample.position
        v_ref = sample.velocity if sample.velocity is not None else zero_w
        w_ref = (sample.angular_velocity
                 if sample.angular_velocity is not None else zero_w)
        a_ref = (sample.acceleration
                 if sample.acceleration is not None else zero_w)

        # ctx.orientation is Quat[WorldFrame, PartFrame]; for a root-mounted
        # endpoint PartFrame ≡ CraftFrame. Reinterpret the user's reference
        # quaternion (authored as [WorldFrame, CraftFrame]) into PartFrame so
        # the SO(3) box-minus frames line up.
        q = ctx.orientation
        if sample.orientation is None:
            q_ref = Quat.identity(from_frame=WorldFrame, to_frame=PartFrame)
        else:
            q_ref = Quat[WorldFrame, PartFrame].from_mx(sample.orientation._mx)

        p = ctx.position[WorldFrame]
        v = ctx.velocity[WorldFrame]
        w = ctx.angular_velocity[WorldFrame]

        # --- SE(3) spring-damper in world coordinates --------------------
        pos_err = p_ref - p
        F_world = pos_err * float(self.kp_pos) + (v_ref - v) * float(self.kd_pos)
        att_err = SO3Manifold().boxminus(q_ref, q)        # Vec3[WorldFrame]
        tau_world = att_err * float(self.kp_att) + (w_ref - w) * float(self.kd_att)

        # Feedforward: cancel gravity and track linear acceleration exactly.
        m = float(self.mass)
        if m > 0.0:
            g_world = gravity_at(ctx, p)
            F_world = F_world + a_ref * m - g_world * m

        # --- rotate the world-frame wrench into the part frame -----------
        R_pw = q.conjugate()                              # Quat[PartFrame, WorldFrame]
        return PartUpdate(
            wrench=Wrench(force=R_pw.apply(F_world),
                          torque=R_pw.apply(tau_world)),
            outputs={"pos_error": pos_err},
        )


# ---------------------------------------------------------------------------
# Convenience trajectory builders
# ---------------------------------------------------------------------------

def hover(position, orientation=None) -> Callable:
    """A constant-pose reference: park the craft at `position` (a length-3
    tuple, world frame) holding `orientation` (a (w,x,y,z) tuple, or None
    for identity)."""
    px, py, pz = (float(c) for c in position)
    quat = None if orientation is None else tuple(float(c) for c in orientation)

    def _traj(t):
        pos = Vec3[WorldFrame].constant((px, py, pz))
        ori = (None if quat is None
               else Quat[WorldFrame, PartFrame].constant(quat))
        return TrajectorySample(position=pos, orientation=ori)

    return _traj


class LinearTrajectory:
    """Constant-velocity (optionally constant angular-velocity) reference.

    ``p(t) = start + velocity·t``; attitude is the fixed `orientation`
    (a (w,x,y,z) tuple, default identity) and `angular_velocity` is held
    constant. Closed-form velocity/accel feedforward come for free.
    """

    def __init__(self, start, velocity, orientation=None, angular_velocity=None):
        self.start = tuple(float(c) for c in start)
        self.velocity = tuple(float(c) for c in velocity)
        self.orientation = (None if orientation is None
                            else tuple(float(c) for c in orientation))
        self.angular_velocity = (None if angular_velocity is None
                                 else tuple(float(c) for c in angular_velocity))

    def __call__(self, t):
        s = Vec3[WorldFrame].constant(self.start)
        vel = Vec3[WorldFrame].constant(self.velocity)
        pos = s + vel * t
        ori = (None if self.orientation is None
               else Quat[WorldFrame, PartFrame].constant(self.orientation))
        w = (None if self.angular_velocity is None
             else Vec3[WorldFrame].constant(self.angular_velocity))
        return TrajectorySample(position=pos, orientation=ori,
                                velocity=vel, angular_velocity=w)
