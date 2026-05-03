"""connect_demo — minimal demo of the new make_config() entry point + connect().

Single Target containing one World. Two thrusters on one craft; the
follower mirrors the leader via in-process `connect()`. The leader's
throttle is driven from Zenoh.

Regenerate from the repo root:

    PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \\
        examples/connect_demo/craft.py --workflow binary \\
        --out examples/connect_demo/generated/connect_demo
"""

from manta_codegen import (Craft, MantaConfig, Target, World, connect, publish,
                           subscribe)
from manta_codegen.parts import Mass, Thruster


def make_config() -> MantaConfig:
    body     = Mass("body", mass=1.0, apply_gravity=False)
    leader   = Thruster("leader",   max_thrust=1.0, direction=(1.0, 0.0, 0.0))
    follower = Thruster("follower", max_thrust=1.0, direction=(0.0, 1.0, 0.0))

    c = Craft("connect_demo")
    c.add(body)
    c.add(leader)
    c.add(follower)

    w = World("connect_demo").add_craft(c)

    subscribe(leader.set_throttle, "manta/connect_demo/cmd")
    connect(leader.throttle, follower.set_throttle)
    publish({
        "leader":   leader.throttle,
        "follower": follower.throttle,
    }, "manta/connect_demo/state")

    return MantaConfig(targets=[
        Target("connect_demo", drives=[w], dt=0.001, sim_rate_mult=1.0),
    ])
