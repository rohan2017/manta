"""ex2 — quadcopter X-config with airfoil-driven propellers.

Each rotor is a `Motor` (revolute joint about body +z) with two
`Naca00xx` blades attached as children at 180° apart. Aerodynamic lift
on the blades replaces the old `Thruster1`'s direct thrust model; the
reaction torque comes naturally from the blade drag about the motor
axis. The user main commands per-motor torque via a velocity PID.

Body has a `Collider` so the craft can rest on a `CollisionField`
ground plane instead of falling through. A `FluidField` set to
sea-level air provides ρ and ambient velocity for the airfoils.

Library workflow: codegen emits the Craft type + telemetry; the user's
main.cpp does the rate PIDs, throttle → ω_target mixing, velocity P
control per motor, and Zenoh wiring.

Generate from the repo root:

    .venv/bin/python -m manta_codegen.cli examples/ex2_quadcopter/config.py --workflow library
"""

from __future__ import annotations

import math

from manta_codegen import Craft, MantaConfig, Target, World, tf
from manta_codegen.parts  import IMU, Mass, Motor, Naca00xx, Collider
from manta_codegen.fields import CollisionField, FluidField, GravityField


# ---- Craft scale ----------------------------------------------------------
MASS                = 1.0           # kg — body mass
ARM_L               = 0.25          # half-diagonal of the X-config (m)
BODY_COLLIDER_R     = 0.05          # m — sphere radius for ground contact

# ---- Propeller geometry --------------------------------------------------
# Bigger blades vs. the canonical micro-quad geometry: lift scales as
# b³ (blade-element integration) and a slower hover ω means the rotor
# dynamics integrate cleanly with the default 1 ms dt.
BLADE_CHORD         = 0.03          # m
BLADE_SPAN          = 0.20          # m   (each blade: motor center → tip)
BLADE_THICKNESS     = 0.12          # NACA 0012
BLADE_PITCH_DEG     = 12.0          # well below stall (15°)
BLADE_PITCH_RAD     = math.radians(BLADE_PITCH_DEG)
N_SEGMENTS          = 4
MOTOR_STALL_TORQUE  = 2.0           # N·m  (capped: cheaper to keep ω small)

# ---- Atmosphere -----------------------------------------------------------
# ISA sea-level: R_air = 287.05 J/(kg·K), T = 288.15 K, p = 101325 Pa
# ⇒ ρ ≈ 1.225 kg/m³.
ISA_R, ISA_T, ISA_P = 287.05, 288.15, 101325.0

# ---- Ground contact (soft enough for dt=0.001 stability) -----------------
# ω_n = √(k/m) ≈ √(5e3/1) ≈ 70 rad/s ⇒ T ≈ 90 ms ≫ 10·dt. Stable.
GROUND_K, GROUND_D  = 5.0e3, 50.0

# ---- X-config rotor layout. Looking down +z, "cw" is the rotation sense
# the motor needs to spin in to produce upward thrust given the blade
# installation handedness below. CCW props at fr/bl, CW at fl/br (the
# standard "+" pattern).
ROTORS: list[tuple[str, tuple[float, float, float], bool]] = [
    ("fr", (+ARM_L, -ARM_L, 0.0), False),
    ("fl", (+ARM_L, +ARM_L, 0.0), True),
    ("bl", (-ARM_L, +ARM_L, 0.0), False),
    ("br", (-ARM_L, -ARM_L, 0.0), True),
]


# ---- Small quaternion helpers --------------------------------------------
def _quat_z(angle: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for rotation by `angle` about +z."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def _quat_y(angle: float) -> tuple[float, float, float, float]:
    """Quaternion for rotation about +y."""
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def _quat_mul(a, b):
    """Quaternion product a · b (apply b first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )


def _blade(motor: Motor, blade_idx: int, cw: bool, name: str) -> None:
    """Attach a single Naca00xx blade to `motor`. blade_idx 0 sits on +x of
    the motor's rotating frame, blade_idx 1 sits on -x (180° apart).

    Installation: the blade's part-frame chord lies along the motor's
    tangent direction (so the relative wind from rotation hits the LE),
    the airfoil span runs from motor center out to the tip, and the
    airfoil lift axis (+z airfoil) maps to motor +z (thrust up).

    For a CCW motor (positive ω about +z), the blade at +x_motor moves
    in +y_motor as it spins. The leading edge must point along +y_motor
    for the wind to hit it correctly — that's a +90° rotation about
    motor +z applied to the airfoil's +x. Blade 1 at −x_motor is the
    mirror: it moves in −y_motor when the prop spins CCW, so its LE is
    at −y_motor (a −90° rotation). CW props reverse both."""
    sign = +1.0 if blade_idx == 0 else -1.0
    pos = (sign * BLADE_SPAN / 2.0, 0.0, 0.0)

    # Rotor-z component of the install. CCW: blade 0 → +90°, blade 1 → −90°.
    # CW swaps them.
    if cw:
        rot_z = -math.pi / 2 if blade_idx == 0 else +math.pi / 2
    else:
        rot_z = +math.pi / 2 if blade_idx == 0 else -math.pi / 2
    q_rot   = _quat_z(rot_z)
    # Pitch: rotate the airfoil about its +y by −α so LE tilts toward +z
    # in motor frame (positive thrust for the chosen rotation direction).
    q_pitch = _quat_y(-BLADE_PITCH_RAD)
    q_install = _quat_mul(q_rot, q_pitch)

    motor.add(Naca00xx(name=name,
                       chord=BLADE_CHORD,
                       span=BLADE_SPAN,
                       thickness_ratio=BLADE_THICKNESS,
                       n_sample_points=N_SEGMENTS,
                       transform=tf(position=pos, quaternion=q_install)))


def make_config() -> MantaConfig:
    c = Craft("ex2")
    c.add(Mass("body", mass=MASS, moi=(0.01, 0.01, 0.02)))
    # Ground contact via a soft single-sphere collider on the body root.
    c.add(Collider("hull",
                   radius=BODY_COLLIDER_R,
                   k_normal=GROUND_K,
                   d_normal=GROUND_D))
    # IMU at body origin.
    c.add(IMU("imu"))

    # Four propeller assemblies. Each Motor's child blades rotate with
    # the joint; the user-facing `craft.<name>_motor()` accessor exposes
    # `set_torque(τ)` and `rate()` for the velocity-control PID.
    #
    # A small hub Mass per rotor gives the joint a non-zero axial
    # inertia. Without it Motor::resolve detects I_axial ≈ 0 and locks
    # the joint (no spin-up under torque). 1e-4 kg·m² is ballpark for a
    # 0.02 kg hub at ~0.07 m radius — same order as a real micro-prop +
    # motor armature combined.
    HUB_MASS = 0.05         # kg
    HUB_MOI  = (1.0e-3, 1.0e-3, 1.0e-3)   # axial 1 mN·m·s²/rad — slow enough
                                          # for dt=1ms stability under ~2 N·m
    for name, pos, cw in ROTORS:
        m = Motor(f"{name}_motor",
                  axis=(0.0, 0.0, 1.0),
                  stall_torque=MOTOR_STALL_TORQUE,
                  damping=0.0,
                  transform=tf(pos))
        m.add(Mass(f"{name}_hub", mass=HUB_MASS, moi=HUB_MOI))
        _blade(m, blade_idx=0, cw=cw, name=f"{name}_blade_a")
        _blade(m, blade_idx=1, cw=cw, name=f"{name}_blade_b")
        c.add(m)

    # World: gravity, air, ground.
    air = FluidField().add_uniform_gas(R=ISA_R, temperature=ISA_T,
                                        pressure=ISA_P)
    ground = CollisionField().add_infinite_plane(
        point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        k=GROUND_K, d=GROUND_D)
    grav = GravityField(g=(0.0, 0.0, -9.81))

    w = (World()
         .add_field(grav)
         .add_field(air)
         .add_field(ground)
         .add_craft(c))
    return MantaConfig(targets=[Target("ex2", drives=[w])])
