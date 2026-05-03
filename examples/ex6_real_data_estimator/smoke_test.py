"""Programmatic smoke test for ex7 (real-data-only estimator).

Runs `fake_sensors.py` for ~5 s alongside the ex7 binary, subscribes to
`manta/ex6/estimate`, and verifies the EKF velocity estimate converged on
the known truth (`vx = 1.5 m/s`, others zero).

Usage:
    Terminal A: ./build/examples/ex6_real_data_estimator
    Terminal B: python examples/ex6_real_data_estimator/fake_sensors.py
    Terminal C: python examples/ex6_real_data_estimator/smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import zenoh


LATEST: deque = deque(maxlen=1)


def on_estimate(sample: zenoh.Sample) -> None:
    try:
        LATEST.append(json.loads(bytes(sample.payload)))
    except Exception:
        pass


def main() -> int:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    sub = session.declare_subscriber("manta/ex6/estimate", on_estimate)

    # Spawn fake_sensors.py for 5 seconds.
    fake = subprocess.Popen([
        sys.executable,
        str(Path(__file__).parent / "fake_sensors.py"),
        "--seconds", "5",
    ])

    print("waiting up to 3 s for first estimate message...")
    for _ in range(60):
        if LATEST:
            break
        time.sleep(0.05)
    if not LATEST:
        fake.terminate()
        print("FAIL: no estimate stream from ex7. Is the binary running?",
              file=sys.stderr)
        return 1

    initial = LATEST[-1]
    print(f"initial estimate v={initial['v']}")

    # Let fake_sensors run its full 5 seconds.
    fake.wait(timeout=8)
    time.sleep(0.5)   # let the EKF process the last messages

    final = LATEST[-1]
    print(f"final estimate p={final['p']} v={final['v']} "
          f"P_diag={final['P']}")

    failures = 0
    expected_vx = 1.5
    if abs(final["v"][0] - expected_vx) > 0.10:
        print(f"FAIL: v_x off — got {final['v'][0]:.4f}, "
              f"expected {expected_vx}.", file=sys.stderr)
        failures += 1
    for axis in (1, 2):
        if abs(final["v"][axis]) > 0.10:
            print(f"FAIL: v[{axis}] should be ~0, got {final['v'][axis]:.4f}",
                  file=sys.stderr)
            failures += 1

    # Velocity covariance should have shrunk (EKF converged).
    for i in (3, 4, 5):
        if final["P"][i] > 0.1:
            print(f"FAIL: P[{i}] = {final['P'][i]:.4f} did not shrink",
                  file=sys.stderr)
            failures += 1

    sub.undeclare()
    session.close()

    if failures:
        print(f"smoke test: {failures} failure(s)", file=sys.stderr)
        return 1
    print("smoke test: OK — estimator tracks fake-sensor truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
