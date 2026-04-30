"""Programmatic smoke test for ex4 reaction wheel.

Pulses a positive torque command on the motor and verifies that:
  - the wheel spins up (positive joint rate),
  - the craft body counter-rotates (negative angular velocity about z),
  - total angular momentum stays approximately conserved (zero, started
    from rest).

Usage:
    Terminal A: ./build/examples/ex4
    Terminal B: python examples/ex4_reaction_wheel/smoke_test.py
"""

from __future__ import annotations

import json
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
    sub = session.declare_subscriber("manta/ex4/state", on_state)
    pub = session.declare_publisher("manta/ex4/wheel/cmd")

    print("waiting up to 3s for first telemetry...")
    for _ in range(60):
        if LATEST:
            break
        time.sleep(0.05)
    if not LATEST:
        print("FAIL: no telemetry from ex4. Is the binary running?",
              file=sys.stderr)
        return 1

    initial = LATEST[-1]
    print(f"initial body angular vel z = {initial['w'][2]:+.4f} rad/s")
    print(f"initial wheel rate          = {initial.get('wheel', {}).get('rate', 0):+.4f} rad/s")

    # Pulse +0.5 N·m for 1 second, then 0 for 0.3s.
    print("pulsing +0.5 N·m motor torque for 1.0 s...")
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        pub.put(json.dumps([0.5]))
        time.sleep(0.02)
    pub.put(json.dumps([0.0]))
    time.sleep(0.3)

    final = LATEST[-1]
    body_wz   = final["w"][2]
    wheel_rate = final.get("wheel", {}).get("rate", 0.0)
    print(f"final body angular vel z = {body_wz:+.4f} rad/s")
    print(f"final wheel rate          = {wheel_rate:+.4f} rad/s")

    failures = 0
    if wheel_rate <= 0.5:
        print(f"FAIL: wheel did not spin up (rate {wheel_rate:.3f}).",
              file=sys.stderr)
        failures += 1
    if body_wz >= -0.001:
        print(f"FAIL: body did not counter-rotate (ω_z {body_wz:.4f}).",
              file=sys.stderr)
        failures += 1

    # Total angular momentum about z = I_body * ω_body + I_wheel * (ω_body + ω_wheel_rel)
    # We don't know I exactly here, but conservation gives:
    #   L_z = I_body * body_wz + I_wheel * (body_wz + wheel_rate)
    # which should be ≈ 0 (started at rest, no external torque on the system).
    # Use the values we baked into craft.py.
    I_body  = 0.02       # body MOI z (kg·m²)
    I_wheel = 0.005      # flywheel ring MOI z (kg·m²)
    Lz_total = I_body * body_wz + I_wheel * (body_wz + wheel_rate)
    print(f"L_z total (Σ I·ω) = {Lz_total:+.5f} N·m·s")
    if abs(Lz_total) > 0.05:
        print(f"FAIL: angular momentum drift {Lz_total:.5f} too large.",
              file=sys.stderr)
        failures += 1

    sub.undeclare()
    session.close()

    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — reaction wheel works as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
