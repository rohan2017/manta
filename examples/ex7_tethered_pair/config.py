"""ex7 — multi-craft tether codegen demo.

Two free-flying drones connected by a single Tether. The leader drone
has a thrust-controlled thruster on its +x axis; the follower has none.
When the leader thrusts forward, the tether pulls the follower along.

Python `Tether(...)` + `World.add_tether(...)` produces the C++
`manta::coupling::Tether` instance plus the two TetherEndpoint
attachments at world-main time. All bindings live at the World level
since they connect signals across two crafts.

Regenerate from the repo root:

    PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \
        examples/ex7_tethered_pair/config.py --workflow binary \
        --out examples/ex7_tethered_pair/generated/ex7
"""

from manta_codegen import (Craft, MantaConfig, Target, Tether, World, publish,
                           subscribe)
from manta_codegen.parts import Mass, Thruster


def make_drone(name: str, with_thruster: bool) -> tuple[Craft, "Thruster | None"]:
    body = Mass("body", mass=1.0)
    c = Craft(name)
    c.add(body)
    thr = None
    if with_thruster:
        thr = Thruster("thrust", max_thrust=2.0, direction=(1.0, 0.0, 0.0))
        c.add(thr)
    return c, thr


def make_config() -> MantaConfig:
    leader,   leader_thr   = make_drone("leader",   with_thruster=True)
    follower, _            = make_drone("follower", with_thruster=False)

    tether = Tether(rest_length=2.0, stiffness=20.0, damping=2.0)

    w = (World(name="ex7")
            .add_craft(leader,   pos=(0.0, 0.0, 0.0))
            .add_craft(follower, pos=(2.0, 0.0, 0.0))
            .add_tether(tether,
                        endpoint_a=(leader,   "hook"),
                        endpoint_b=(follower, "hook")))

    subscribe(leader_thr.set_throttle, "manta/ex7/leader/cmd")
    publish({
        "t": leader.time,
        "p": leader.position, "v": leader.vel_linear,
        "throttle": leader_thr.throttle,
    }, "manta/ex7/leader/state")
    publish({
        "t": follower.time,
        "p": follower.position, "v": follower.vel_linear,
    }, "manta/ex7/follower/state")

    return MantaConfig(targets=[
        Target("ex7", drives=[w], dt=0.001, sim_rate_mult=1.0),
    ])
