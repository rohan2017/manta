"""ex7 — multi-craft tether codegen demo.

Two free-flying drones connected by a single Tether. The leader drone
has a thrust-controlled thruster on its +x axis; the follower has none.
When the leader thrusts forward, the tether pulls the follower along.

This is the codegen-driven equivalent of ex8's hand-written tether
chain: Python `Tether(...)` + `World.add_tether(...)` produces the C++
`manta::coupling::Tether` instance plus the two TetherEndpoint
attachments at world-main time.

Regenerate from the repo root:

    PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \
        examples/ex7_tethered_pair/craft.py --workflow binary \
        --out examples/ex7_tethered_pair/generated/ex7
"""

from manta_codegen import Craft, Tether, World
from manta_codegen.parts import Mass, Thruster


def make_drone(name: str, with_thruster: bool) -> Craft:
    body = Mass("body", mass=1.0)
    c = Craft(name)
    c.add(body)
    if with_thruster:
        thr = Thruster("thrust", max_thrust=2.0, direction=(1.0, 0.0, 0.0))
        c.add(thr)
        c.subscribe(thr.set_throttle, "manta/ex7/leader/cmd")
        c.publish({
            "t": c.time,
            "p": c.position, "v": c.vel_linear,
            "throttle": thr.throttle,
        }, f"manta/ex7/{name}/state")
    else:
        c.publish({
            "t": c.time,
            "p": c.position, "v": c.vel_linear,
        }, f"manta/ex7/{name}/state")
    return c


def make_world() -> World:
    leader   = make_drone("leader",   with_thruster=True)
    follower = make_drone("follower", with_thruster=False)

    tether = Tether(rest_length=2.0, stiffness=20.0, damping=2.0)

    return (World(name="ex7")
            .add_craft(leader,   pos=(0.0, 0.0, 0.0))
            .add_craft(follower, pos=(2.0, 0.0, 0.0))
            .add_tether(tether,
                        endpoint_a=(leader,   "hook"),
                        endpoint_b=(follower, "hook"))
            .run(dt=0.001, sim_rate_mult=1.0))
