"""Programmatic smoke test for ex8 swarm.

Pulses the leader's thruster, verifies all three drones move forward
(Tether-coupled chain). Confirms Scene's three-phase update is
correctly threading state between sibling crafts.

Usage:
    Terminal A: ./build/examples/ex8_swarm
    Terminal B: python examples/ex8_swarm/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque

import zenoh


LATEST = [deque(maxlen=1) for _ in range(3)]


def make_handler(idx: int):
    def on_state(sample: zenoh.Sample) -> None:
        try:
            LATEST[idx].append(json.loads(bytes(sample.payload)))
        except Exception:
            pass
    return on_state


def main() -> int:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    subs = [
        session.declare_subscriber(f"manta/ex8/{i}/state", make_handler(i))
        for i in range(3)
    ]
    pub = session.declare_publisher("manta/ex8/leader/cmd")

    print("waiting up to 3s for first telemetry from all 3 drones...")
    for _ in range(60):
        if all(LATEST[i] for i in range(3)):
            break
        time.sleep(0.05)
    if not all(LATEST[i] for i in range(3)):
        print("FAIL: missing telemetry. Is ex8_swarm running?", file=sys.stderr)
        return 1

    initial = [LATEST[i][-1] for i in range(3)]
    print("initial positions:")
    for i in range(3):
        print(f"  drone{i}: p={initial[i]['p']}")

    # Pulse leader thrust at full throttle for 2 s, then drift 2 s.
    print("pulsing leader thrust (full) for 2 s, drift 2 s...")
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        pub.put(json.dumps([1.0]))
        time.sleep(0.02)
    pub.put(json.dumps([0.0]))
    time.sleep(2.0)

    final = [LATEST[i][-1] for i in range(3)]
    print("final positions:")
    for i in range(3):
        print(f"  drone{i}: p={final[i]['p']}")

    failures = 0
    # All three drones must have moved forward (positive Δx).
    for i in range(3):
        dx = final[i]["p"][0] - initial[i]["p"][0]
        if dx <= 0.05:
            print(f"FAIL: drone{i} did not move forward (Δx = {dx:.4f}).",
                  file=sys.stderr)
            failures += 1

    # Followers must have moved less than the leader (tethered chain takes
    # time to stretch and pull the others along).
    leader_dx = final[0]["p"][0] - initial[0]["p"][0]
    for i in (1, 2):
        follower_dx = final[i]["p"][0] - initial[i]["p"][0]
        if follower_dx >= leader_dx:
            print(f"FAIL: drone{i} moved {follower_dx:.3f}, more than leader "
                  f"{leader_dx:.3f}.", file=sys.stderr)
            failures += 1

    for s in subs: s.undeclare()
    session.close()

    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — leader pulled all 3 drones along the tether chain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
