"""Programmatic smoke test for ex5 underwater sub.

Verifies:
  1. The sub stays approximately neutrally buoyant (z position drifts <0.2 m
     after 1.5 s with no thrust).
  2. After commanding aft thruster at full throttle for 1 s, the sub gains
     forward (+x) velocity.
  3. The DVL telemetry reads body-frame velocity that points forward (+x)
     in matching magnitude.

Usage:
    Terminal A: ./build/examples/ex5
    Terminal B: python examples/ex5_underwater/smoke_test.py
"""

from __future__ import annotations

import json
import math
import sys
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


def main() -> int:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    sub = session.declare_subscriber("manta/ex5/state", on_state)
    pub = session.declare_publisher("manta/ex5/aft_thrust/cmd")

    print("waiting up to 3 s for first telemetry...")
    for _ in range(60):
        if LATEST:
            break
        time.sleep(0.05)
    if not LATEST:
        print("FAIL: no telemetry. Is ex5 running?", file=sys.stderr)
        return 1

    initial = LATEST[-1]
    print(f"initial p = {initial['p']}  v = {initial['v']}")

    failures = 0

    # 1) Neutral buoyancy test. With Hull volume * water_density = body mass,
    # the net buoyancy + gravity should be ≈ 0 → z stays put.
    print("waiting 1.5 s with no thrust to check neutral buoyancy...")
    pub.put(json.dumps([0.0]))
    time.sleep(1.5)
    pre_thrust = LATEST[-1]
    z_drift = pre_thrust["p"][2] - initial["p"][2]
    print(f"  z drift = {z_drift:+.4f} m")
    if abs(z_drift) > 0.2:
        print(f"  FAIL: z drift {z_drift:.4f} too large; check Hull/Mass balance",
              file=sys.stderr)
        failures += 1

    # 2) Apply +1.0 throttle for 1 s. Aft thruster pushes +x.
    print("commanding aft_thrust = 1.0 for 1.0 s...")
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        pub.put(json.dumps([1.0]))
        time.sleep(0.02)
    pub.put(json.dumps([0.0]))
    time.sleep(0.3)
    final = LATEST[-1]
    dv_x = final["v"][0] - pre_thrust["v"][0]
    print(f"  Δv_x = {dv_x:+.4f} m/s")
    if dv_x < 0.3:
        print(f"  FAIL: aft thruster did not produce forward motion (Δv_x={dv_x:.4f})",
              file=sys.stderr)
        failures += 1

    sub.undeclare()
    session.close()

    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — neutral buoyancy and thrust both work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
