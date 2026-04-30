"""Programmatic smoke test for ex0's pilot loop.

What it does:
    1. Subscribes to 'manta/ex0/state'.
    2. Pulses each thruster (+x, +y, +z) for a brief command window.
    3. Reads back the published telemetry stream.
    4. Asserts the resulting velocity vector points in the commanded
       direction.

Usage:
    Terminal A:  ./build/examples/ex0
    Terminal B:  python examples/ex0_free_flight/smoke_test.py

Exits 0 on success, non-zero on failure. Designed for CI / quick sanity
checks; a human-in-the-loop pilot run uses keyboard_controller.py +
viewer.py instead.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque

import zenoh


LATEST: deque = deque(maxlen=1)


def on_state(sample: zenoh.Sample) -> None:
    try:
        d = json.loads(bytes(sample.payload))
    except Exception:
        return
    LATEST.append(d)


def pulse(session: zenoh.Session, name: str, throttle: float, duration: float) -> None:
    pub = session.declare_publisher(f"manta/ex0/{name}/cmd")
    end = time.monotonic() + duration
    while time.monotonic() < end:
        pub.put(json.dumps([throttle]))
        time.sleep(0.02)
    pub.put(json.dumps([0.0]))


def latest_velocity() -> list[float]:
    if not LATEST:
        return [0.0, 0.0, 0.0]
    return LATEST[-1]["v"]


def main() -> int:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    sub = session.declare_subscriber("manta/ex0/state", on_state)

    print("waiting up to 3s for first state message...")
    for _ in range(60):
        if LATEST:
            break
        time.sleep(0.05)
    if not LATEST:
        print("FAIL: no telemetry received from ex0. Is the binary running?",
              file=sys.stderr)
        sub.undeclare()
        session.close()
        return 1

    print(f"got initial state: t={LATEST[-1]['t']:.3f}s p={LATEST[-1]['p']}")

    expectations = [
        # (thruster_name, expected_axis, expected_sign)
        ("tx_p", 0, +1),
        ("ty_p", 1, +1),
        ("tz_p", 2, +1),
    ]

    failures = 0
    for name, axis, sign in expectations:
        v0 = latest_velocity()
        pulse(session, name, throttle=1.0, duration=0.5)
        time.sleep(0.3)
        v1 = latest_velocity()

        delta = [v1[i] - v0[i] for i in range(3)]
        print(f"thruster {name}: dv = ({delta[0]:+.3f}, {delta[1]:+.3f}, "
              f"{delta[2]:+.3f})")

        # Expect motion along the commanded axis with the expected sign,
        # and negligible motion on other axes.
        if sign * delta[axis] < 0.5:
            print(f"  FAIL: expected dv on axis {axis} with sign {sign}, "
                  f"got {delta[axis]:.3f}", file=sys.stderr)
            failures += 1
        for other in range(3):
            if other == axis:
                continue
            if abs(delta[other]) > 0.3:
                print(f"  FAIL: unexpected motion on axis {other}: "
                      f"{delta[other]:.3f}", file=sys.stderr)
                failures += 1

    sub.undeclare()
    session.close()

    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — all 3 axes responded as commanded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
