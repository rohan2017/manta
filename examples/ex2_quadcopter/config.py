"""ex2 — quadcopter X-config with 4 tensor-style Thruster1s, IMU.

Library workflow: codegen emits the Craft type and telemetry; the user's
main.cpp does the rate PIDs, X-config mixing, and Zenoh wiring.

Each rotor is a `Thruster1` with explicit force / torque coefficient
vectors:
    F_1 = (0, 0, max_thrust)
    τ_1 = (0, 0, ±k_t · max_thrust)     # sign per CCW/CW
This is the new tensor API replacement for the deleted `PropThruster`.

Generate from the repo root:

    PYTHONPATH=python/manta_codegen/src \
        python -m manta_codegen.cli examples/ex2_quadcopter/config.py --workflow library
"""

from manta_codegen import Craft, MantaConfig, Target, World, tf
from manta_codegen.parts  import IMU, Mass, Thruster1
from manta_codegen.fields import GravityField


MASS  = 1.0
ARM_L = 0.25     # half-diagonal in m
KT    = 0.02     # counter-torque coefficient
MAX_THRUST_PER_PROP = 2 * (MASS * 9.81) / 4  # 2× hover thrust per prop


# X-config rotor layout. Looking down +z (top-down), CCW props twist
# the body in -z and CW props twist it in +z (Thruster1 reaction torque
# follows the τ_1 sign in _prop). Geometry is exported as a module-level
# list so the viewer can render rotor markers + thrust arrows at the
# same positions without duplicating the magic numbers.
ROTORS: list[tuple[str, tuple[float, float, float], bool]] = [
    # name, (x, y, z) body-frame offset, cw?
    ("fr", (+ARM_L, -ARM_L, 0.0), False),
    ("fl", (+ARM_L, +ARM_L, 0.0), True),
    ("bl", (-ARM_L, +ARM_L, 0.0), False),
    ("br", (-ARM_L, -ARM_L, 0.0), True),
]


def _prop(name: str, x: float, y: float, cw: bool) -> Thruster1:
    """X-config rotor: vertical thrust + reaction torque about z."""
    sign = +1.0 if cw else -1.0
    F1 = (0.0, 0.0, MAX_THRUST_PER_PROP)
    T1 = (0.0, 0.0, sign * KT * MAX_THRUST_PER_PROP)
    return Thruster1(name, force_coefs=[F1], torque_coefs=[T1],
                     transform=tf((x, y, 0)))


def make_config() -> MantaConfig:
    c = Craft("ex2")

    # Pencil-shaped body inertia: lump everything into one Mass at the root.
    # Auto-gravity is on by default (Mass auto-applies F=m·g when a
    # GravityField is registered).
    c.add(Mass("body", mass=MASS, moi=(0.01, 0.01, 0.02)))

    for name, (x, y, _z), cw in ROTORS:
        c.add(_prop(name, x, y, cw=cw))

    c.add(IMU("imu"))

    # Library workflow: user main does pub/sub manually.
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81))).add_craft(c)
    return MantaConfig(targets=[Target("ex2", drives=[w])])
