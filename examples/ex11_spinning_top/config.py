"""ex11 — Spinning-top precession on a flat ground plane.

Library workflow: codegen emits the Ex11Craft type + telemetry struct +
cmake fragment. The user's main.cpp pre-spins the flywheel, drives the
brief lateral kick at t=3 s, and publishes telemetry over Zenoh for
viewer.py to render in Rerun.

Craft layout (body frame: +z up, +x forward, +y left):

    +z (vertical)
     │
     ●  top_col          (small sphere collider at top of stick)
     │
     ●  thruster         (lateral kick, +x direction, near top)
     │
     ●─ flywheel         (heavy disk, on Motor whose axis = body +z)
     │
     ●  stick body       (thin rod along body z, mass at origin)
     │
     ●  bottom_col       (small sphere collider — pivot point on ground)

The flywheel pre-spins to 200 rad/s about body +z. Without the rotor-gyro
torque correction in `Craft::sense_and_aggregate`, the small lateral kick
at t=3 s would tip the stick straight over in +x. With it, the body
precesses around the vertical at Ω = M·g·h / (I·θ̇) ≈ 2.7 rad/s
(period ≈ 2.3 s).

Generate from the repo root:

    .venv/bin/python -m manta_codegen.cli \\
        examples/ex11_spinning_top/config.py --workflow library
"""

from __future__ import annotations

from manta_codegen import Craft, MantaConfig, Target, World, tf
from manta_codegen.parts  import Collider, Mass, Motor, Thruster
from manta_codegen.fields import CollisionField, GravityField


# ---- Stick geometry / mass ----
L_STICK     = 0.8                    # m — full length, body z axis
STICK_MASS  = 0.1                    # kg — thin rod, light
# Transverse MOI of a thin rod about its center: m·L²/12.
STICK_I_PERP = STICK_MASS * L_STICK * L_STICK / 12.0   # ≈ 5.3e-3 kg·m²

# ---- Flywheel: thin disk centered on body z axis ----
FLY_MASS    = 2.0                    # kg
FLY_R       = 0.15                   # m
FLY_I_AXIAL = 0.5  * FLY_MASS * FLY_R * FLY_R     # ½·m·r² = 0.0225 kg·m²
FLY_I_PERP  = 0.25 * FLY_MASS * FLY_R * FLY_R     # ¼·m·r²
MOTOR_Z     = 0.20                   # m — flywheel mount above body origin
FLY_RATE_0  = 200.0                  # rad/s pre-spin (set in main.cpp)

# ---- Thruster: lateral kick ----
# Intermediate impulse: large enough to give a visible (~15°) tilt
# amplitude, small enough that the gyro keeps it from immediately
# tipping over. Tilt magnitude is set by the kick impulse divided by
# the body's transverse MOI; precession rate is set by the flywheel
# spin (see FLY_RATE_0 in main.cpp).
#   impulse                = 4.0 N · 0.05 s = 0.2 N·s
#   torque-impulse (bottom) ≈ 4 · 0.8 · 0.05 = 0.16 N·m·s
#   initial tilt rate      ≈ 0.16 / 0.73    ≈ 0.22 rad/s (≈ 13 °/s)
THRUSTER_Z       = +L_STICK / 2.0    # at the top end of the stick
THRUSTER_FORCE   = 4.0               # N at full throttle

# ---- End-cap colliders for ground interaction ----
COLLIDER_R       = 0.02              # m — small spheres at the rod ends

# ---- Ground contact PD ----
# ω_n = √(k/m_total) ≈ √(5000/2.1) ≈ 49 rad/s ⇒ T ≈ 130 ms ≫ 10·dt at 1 kHz.
GROUND_K, GROUND_D = 5.0e3, 50.0
MU_S,    MU_K      = 0.9,   0.7

# Initial body z so the BOTTOM collider sphere's surface rests on z=0.
# Sphere center is at z_body − L/2; surface at z_body − L/2 − r_collider.
# For surface at z=0: z_body = L/2 + r_collider.
SPAWN_Z = L_STICK / 2.0 + COLLIDER_R


def make_config() -> MantaConfig:
    c = Craft("ex11")

    # Body root: thin rod along z. Transverse inertia from the rod
    # formula, axial inertia ~0 (a thin rod doesn't resist spinning
    # about its own length). Tiny non-zero Izz keeps the matrix
    # invertible for the body Euler equation.
    c.add(Mass("stick",
               mass=STICK_MASS,
               moi=(STICK_I_PERP, STICK_I_PERP, 1.0e-6)))

    # Two end-cap colliders. Each is a small sphere; the Collider's part
    # frame attaches to the root via a static transform. Both sit in the
    # same CollisionField pool — they collide independently with the
    # ground plane (and each other, though that won't happen in this sim).
    c.add(Collider("bottom_col",
                   radius=COLLIDER_R,
                   k_normal=GROUND_K, d_normal=GROUND_D,
                   mu_static=MU_S, mu_kinetic=MU_K,
                   transform=tf((0.0, 0.0, -L_STICK / 2.0))))
    c.add(Collider("top_col",
                   radius=COLLIDER_R,
                   k_normal=GROUND_K, d_normal=GROUND_D,
                   mu_static=MU_S, mu_kinetic=MU_K,
                   transform=tf((0.0, 0.0, +L_STICK / 2.0))))

    # Flywheel: a Motor whose joint axis is body +z, with one heavy disk
    # Mass child sitting at the motor's joint origin. Passive mode
    # (stall=0): no actuator drive, the joint coasts under inertia and
    # whatever reactions it gets.
    motor = Motor("fly_motor",
                  axis=(0.0, 0.0, 1.0),
                  stall_torque=0.0,
                  damping=0.0,
                  transform=tf((0.0, 0.0, MOTOR_Z)))
    motor.add(Mass("flywheel",
                   mass=FLY_MASS,
                   moi=(FLY_I_PERP, FLY_I_PERP, FLY_I_AXIAL)))
    c.add(motor)

    # Lateral thruster at the top of the stick. Direction = body +x.
    # Throttle is driven by main.cpp during the kick window.
    c.add(Thruster("kick",
                   max_thrust=THRUSTER_FORCE,
                   direction=(1.0, 0.0, 0.0),
                   transform=tf((0.0, 0.0, THRUSTER_Z))))

    # ---- World: gravity + flat ground ----
    grav   = GravityField(g=(0.0, 0.0, -9.81))
    ground = CollisionField().add_infinite_plane(
        point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        k=GROUND_K, d=GROUND_D, mu_static=MU_S, mu_kinetic=MU_K)

    w = (World()
         .add_field(grav)
         .add_field(ground)
         .add_craft(c, pos=(0.0, 0.0, SPAWN_Z)))

    return MantaConfig(targets=[Target("ex11", drives=[w])])
