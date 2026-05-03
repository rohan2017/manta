"""Programmatic smoke test for ex8 (submarine sim+EKF).

Verifies:
  1. Truth and estimate streams both publish under Zenoh.
  2. Without thrust the sub stays near its initial pose (it sinks slowly
     from negative buoyancy, but shouldn't translate horizontally).
  3. The EKF position estimate tracks the truth within ~0.5 m on each
     axis after a few seconds — the sensors are noisy and the est-side
     dynamics is force-free, but the measurement updates should keep
     things bounded.

Usage:
    Terminal A: ./build/examples/ex8
    Terminal B: python examples/ex8_submarine/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque

import zenoh


TRUTH:    deque = deque(maxlen=1)
ESTIMATE: deque = deque(maxlen=1)


def on_truth(s: zenoh.Sample) -> None:
    try:
        TRUTH.append(json.loads(bytes(s.payload)))
    except Exception:
        pass


def on_estimate(s: zenoh.Sample) -> None:
    try:
        ESTIMATE.append(json.loads(bytes(s.payload)))
    except Exception:
        pass


def main() -> int:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    sub_t = session.declare_subscriber("manta/ex8/state",    on_truth)
    sub_e = session.declare_subscriber("manta/ex8/estimate", on_estimate)

    print("waiting up to 3 s for streams...")
    for _ in range(60):
        if TRUTH and ESTIMATE:
            break
        time.sleep(0.05)
    if not (TRUTH and ESTIMATE):
        print("FAIL: no truth/estimate streams from ex8.", file=sys.stderr)
        return 1

    # Let the sim run ~3 s with no commanded thrust.
    time.sleep(3.0)

    truth    = TRUTH[-1]
    estimate = ESTIMATE[-1]

    print(f"truth p={truth['p']} v={truth['v']}")
    print(f"est   p={estimate['p']} v={estimate['v']}")

    failures = 0
    # No thrust → x/y position should stay near zero.
    if abs(truth["p"][0]) > 0.3:
        print(f"FAIL: truth x = {truth['p'][0]:.3f} drifted",  file=sys.stderr); failures += 1
    if abs(truth["p"][1]) > 0.3:
        print(f"FAIL: truth y = {truth['p'][1]:.3f} drifted",  file=sys.stderr); failures += 1
    # z drifts slowly downward from negative buoyancy — bound it loosely.
    if truth["p"][2] > -3.0 or truth["p"][2] < -10.0:
        print(f"FAIL: truth z = {truth['p'][2]:.3f} out of [-10, -3]",
              file=sys.stderr); failures += 1

    # Estimate should track truth on each axis within ~0.5 m.
    for axis, name in enumerate("xyz"):
        diff = abs(estimate["p"][axis] - truth["p"][axis])
        if diff > 0.5:
            print(f"FAIL: est-truth {name} mismatch {diff:.3f} m",
                  file=sys.stderr); failures += 1

    sub_t.undeclare(); sub_e.undeclare()
    session.close()
    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — sub stays put + EKF tracks truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
