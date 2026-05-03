"""Programmatic smoke test for ex5 sim + EKF demo.

Verifies:
  1. Truth and estimate streams both publish under Zenoh.
  2. After 5 s of free fall + a thruster pulse, the EKF estimated
     velocity tracks the truth velocity within 0.1 m/s on each axis.
  3. The velocity stddev has shrunk below sqrt(initial covariance).

Usage:
    Terminal A: ./build/examples/ex5
    Terminal B: python examples/ex5_estimator_demo/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque

import zenoh


TRUTH:    deque = deque(maxlen=1)
ESTIMATE: deque = deque(maxlen=1)


def on_truth(sample: zenoh.Sample) -> None:
    try:
        TRUTH.append(json.loads(bytes(sample.payload)))
    except Exception:
        pass


def on_estimate(sample: zenoh.Sample) -> None:
    try:
        ESTIMATE.append(json.loads(bytes(sample.payload)))
    except Exception:
        pass


def main() -> int:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    sub_t = session.declare_subscriber("manta/ex5/state",    on_truth)
    sub_e = session.declare_subscriber("manta/ex5/estimate", on_estimate)
    pub   = session.declare_publisher("manta/ex5/thrust/cmd")

    print("waiting up to 3s for first messages...")
    for _ in range(60):
        if TRUTH and ESTIMATE:
            break
        time.sleep(0.05)
    if not TRUTH or not ESTIMATE:
        print("FAIL: missing truth or estimate stream", file=sys.stderr)
        return 1

    # Pulse thrust at half throttle for 1 s, then let it fall freely for 4 s.
    print("pulsing thrust 0.5 for 1.0 s, then drifting 4.0 s...")
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        pub.put(json.dumps([0.5]))
        time.sleep(0.02)
    pub.put(json.dumps([0.0]))
    time.sleep(4.0)

    truth = TRUTH[-1]
    est   = ESTIMATE[-1]

    # Both should report at roughly the same time.
    print(f"truth t={truth['t']:.2f}  v={truth['v']}")
    print(f"est   t={est['t']:.2f}    v={est['v']}    P_diag={est['P']}")

    failures = 0
    for axis in range(3):
        diff = est["v"][axis] - truth["v"][axis]
        if abs(diff) > 0.10:
            print(f"FAIL: estimate v[{axis}] off by {diff:+.4f}", file=sys.stderr)
            failures += 1
    # Velocity covariance entries are P[3..5]; should be well below the
    # initial 1.0 (identity).
    for i in (3, 4, 5):
        if est["P"][i] > 0.1:
            print(f"FAIL: P[{i}] = {est['P'][i]:.4f} did not shrink",
                  file=sys.stderr)
            failures += 1

    sub_t.undeclare()
    sub_e.undeclare()
    session.close()
    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — sim + EKF tracking and converging.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
