"""ex10 — multi-craft codegen demo: two independent crafts in one binary.

This is the simplest possible multi-craft scenario — two free-flying point
masses with thrusters, each on their own state/cmd topics, ticked by a
single sim main.

Demonstrates:
  * `World(name=...)` — required when the world owns multiple crafts
    (the binary's name shouldn't be tied to a single craft).
  * Two distinct Craft objects added to the same World; each gets its own
    .hpp / .cpp / _telemetry.hpp generated, plus per-craft bindings.
  * The codegen-emitted main instantiates `craft_0`, `craft_1`, adds them
    both to the scene, and ticks the world once per loop iteration.

Regenerate from the repo root:

    PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \
        examples/ex10_multi_craft/craft.py --workflow binary \
        --out examples/ex10_multi_craft/generated/ex10
"""

from manta_codegen import Craft, World
from manta_codegen.parts import PointMass, Thruster


def make_drone(name: str, thrust_dir: tuple) -> Craft:
    body = PointMass("body", mass=1.0)
    thr  = Thruster("thrust", max_thrust=2.0, direction=thrust_dir)

    c = Craft(name)
    c.add(body)
    c.add(thr)

    c.publish({
        "t": c.time,
        "p": c.position, "q": c.orientation,
        "v": c.vel_linear, "w": c.vel_angular,
        "throttle": thr.throttle,
    }, f"manta/ex10/{name}/state")
    c.subscribe(thr.set_throttle, f"manta/ex10/{name}/cmd")

    return c


def make_world() -> World:
    drone_a = make_drone("alpha", thrust_dir=(1.0, 0.0, 0.0))   # +x thrust
    drone_b = make_drone("beta",  thrust_dir=(0.0, 1.0, 0.0))   # +y thrust

    return (World(name="ex10")
            .add_craft(drone_a, pos=(0.0, 0.0, 0.0))
            .add_craft(drone_b, pos=(2.0, 0.0, 0.0))   # 2 m apart on x
            .run(dt=0.001, sim_rate_mult=1.0))
