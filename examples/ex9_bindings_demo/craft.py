"""ex9 — minimal demo of the new explicit Binding API (phase 2 of the API
redesign). Exercises both single-signal and bundled-struct topics, on
both the publish and subscribe sides.

Regenerate from the repo root:

    PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \
        examples/ex9_bindings_demo/craft.py --workflow binary \
        --out examples/ex9_bindings_demo/generated/ex9
"""

from manta_codegen import Craft, World
from manta_codegen.parts import IMU, PointMass, Thruster


def make_world() -> World:
    body = PointMass("body", mass=1.0)
    imu  = IMU("imu")
    thr  = Thruster("thrust", max_thrust=5.0, direction=(1.0, 0.0, 0.0))

    c = Craft("ex9")
    c.add(body)
    c.add(imu)
    c.add(thr)

    # Explicit bindings — replaces the legacy publish_state / subscribe_command
    # flag-based defaults. Single-signal binding picks up the default topic;
    # bundled-struct binding requires an explicit topic.
    #
    # Craft-level signals (c.position, c.orientation, c.vel_linear,
    # c.vel_angular) cover the kinematic state that the legacy bundled-state
    # topic used to expose; mix them with part signals in any combination.
    c.publish(imu.last_accel)                         # → manta/ex9/imu/last_accel
    c.publish(imu.last_gyro)                          # → manta/ex9/imu/last_gyro
    c.publish({                                       # → manta/ex9/state (bundled)
        "p": c.position,
        "q": c.orientation,
        "v": c.vel_linear,
        "w": c.vel_angular,
        "throttle": thr.throttle,
    }, "manta/ex9/state")
    c.subscribe(thr.set_throttle, "manta/ex9/cmd")    # explicit topic name

    return World().add_craft(c).run(dt=0.001, sim_rate_mult=1.0)
