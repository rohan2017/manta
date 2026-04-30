"""ex4 — satellite-style craft with a single reaction wheel.

A central body (PointMass) hosts a Motor whose joint output carries a
flywheel (heavy PointMass with high MOI about the motor axis). Commanding
+τ on the motor spins the flywheel one way and, by Newton's third, the
body the opposite way. That's the basic principle of a reaction wheel for
satellite attitude control.

Codegen:

    PYTHONPATH=python/manta_codegen/src \\
        python -m manta_codegen.cli examples/ex4_reaction_wheel/craft.py \\
            --workflow binary

Workflow: binary — the generated main subscribes to a single command topic
'manta/ex4/wheel/cmd' (a JSON float-array carrying torque) and publishes
'manta/ex4/state' with the body pose, velocities, and the wheel angle/rate.
"""

from manta_codegen import Craft
from manta_codegen.parts import Motor, PointMass


# Body inertia: cylinder-ish, lumped into a single PointMass with a small
# diagonal MOI. Mass dominates linear motion (1 kg).
BODY_MASS = 1.0
BODY_MOI  = (0.02, 0.02, 0.02)   # kg·m²

# Flywheel: heavy, concentrated MOI about its spin axis (z). MOI = m * r²
# for a thin ring; we pick numbers so the wheel's I_zz dominates.
WHEEL_MASS = 0.5
WHEEL_R    = 0.10
WHEEL_IZZ  = WHEEL_MASS * WHEEL_R * WHEEL_R   # 0.005 kg·m² ring approximation
WHEEL_MOI  = (WHEEL_IZZ * 0.5, WHEEL_IZZ * 0.5, WHEEL_IZZ)  # disc: I_xx = I_yy = Izz/2


def make_craft() -> Craft:
    c = Craft("ex4")  # no fields — free space, no gravity

    c.root.add(PointMass("body", mass=BODY_MASS, moi=BODY_MOI))

    motor = c.root.add(Motor(
        "wheel",
        axis=(0.0, 0.0, 1.0),
        stall_torque=2.0,        # N·m
        damping=0.0,
        subscribe_command=True,  # accept torque commands on manta/ex4/wheel/cmd
    ))
    # Attach the flywheel as a child of the motor's joint output.
    motor.add(PointMass("flywheel", mass=WHEEL_MASS, moi=WHEEL_MOI))

    return c
