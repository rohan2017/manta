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

from manta_codegen import Craft, World
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


def make_world() -> World:
    body  = PointMass("body", mass=BODY_MASS, moi=BODY_MOI)
    motor = Motor("wheel",
                  axis=(0.0, 0.0, 1.0),
                  stall_torque=2.0,
                  damping=0.0)
    flywheel = PointMass("flywheel", mass=WHEEL_MASS, moi=WHEEL_MOI)

    c = Craft("ex4")
    c.add(body)
    c.add(motor)
    motor.add(flywheel)          # flywheel rides the motor's joint output

    c.publish({
        "p": c.position,         "q": c.orientation,
        "v": c.vel_linear,       "w": c.vel_angular,
        "wheel_angle": motor.angle,
        "wheel_rate":  motor.rate,
        "wheel_accel": motor.accel,
    }, "manta/ex4/state")
    c.subscribe(motor.set_torque, "manta/ex4/wheel/cmd")

    # Free space, no gravity, default initial state.
    return World().add_craft(c).run(dt=0.001, sim_rate_mult=1.0)
