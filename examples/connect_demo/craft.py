"""connect_demo — minimal demo of world.connect() for in-process
signal-to-signal binding.

One craft with two thrusters. The leader's set_throttle is fed from
Zenoh; the follower mirrors the leader's current throttle via
`w.connect(leader.throttle, follower.set_throttle)` — no Zenoh
round-trip, just a per-tick assignment in the generated main.

Regenerate from the repo root:

    PYTHONPATH=python/manta_codegen/src python -m manta_codegen.cli \\
        examples/connect_demo/craft.py --workflow binary \\
        --out examples/connect_demo/generated/connect_demo
"""

from manta_codegen import Craft, World
from manta_codegen.parts  import Mass, Thruster


def make_world() -> World:
    body     = Mass("body", mass=1.0, apply_gravity=False)
    leader   = Thruster("leader",   max_thrust=1.0, direction=(1.0, 0.0, 0.0))
    follower = Thruster("follower", max_thrust=1.0, direction=(0.0, 1.0, 0.0))

    c = Craft("connect_demo")
    c.add(body)
    c.add(leader)
    c.add(follower)

    w = World().add_craft(c)

    # Leader's throttle comes in over Zenoh.
    w.subscribe(leader.set_throttle, "manta/connect_demo/cmd")

    # Follower mirrors leader, in-process. Codegen emits a per-tick copy
    # right after w.update().
    w.connect(leader.throttle, follower.set_throttle)

    # Publish both throttles so a smoke test can verify equality.
    w.publish({
        "leader":   leader.throttle,
        "follower": follower.throttle,
    }, "manta/connect_demo/state")

    return w.run(dt=0.001, sim_rate_mult=1.0)
