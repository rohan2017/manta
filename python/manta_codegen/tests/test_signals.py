"""Phase 2 tests: Signal / BoundSignal / Craft.publish / Craft.subscribe.

Pure-Python tests — no codegen, no compiled C++. Verifies that:
  * Each part's class-level `signals` table attaches BoundSignal attributes
    on instances.
  * Craft.publish / Craft.subscribe validate direction and record bindings.
  * The default topic naming follows the documented pattern.

Run with: pytest python/manta_codegen/tests/  (or just `python -m unittest`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from manta_codegen import ...` when invoked without an install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from manta_codegen.core import Craft
from manta_codegen.parts.sensor.imu import IMU
from manta_codegen.parts.actuator.thruster import Thruster
from manta_codegen.signal import BoundSignal


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_imu_signals_attached() -> None:
    imu = IMU("imu0")
    _check(isinstance(imu.last_accel, BoundSignal), "last_accel not bound")
    _check(isinstance(imu.last_gyro,  BoundSignal), "last_gyro not bound")
    _check(isinstance(imu.set_measurement, BoundSignal), "set_measurement not bound")
    _check(imu.last_accel.direction == "out", "last_accel must be out")
    _check(imu.set_measurement.direction == "in", "set_measurement must be in")
    _check(imu.last_accel.part_name == "imu0", "part_name not propagated")


def test_thruster_signals_attached() -> None:
    t = Thruster("forward", max_thrust=2.0)
    _check(isinstance(t.throttle,     BoundSignal), "throttle not bound")
    _check(isinstance(t.set_throttle, BoundSignal), "set_throttle not bound")
    _check(t.throttle.direction == "out", "throttle must be out")
    _check(t.set_throttle.direction == "in", "set_throttle must be in")


def test_publish_records_binding_with_default_topic() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    c.publish(imu.last_accel)
    _check(len(c.bindings) == 1, "binding not recorded")
    sig, topic, proto = c.bindings[0]
    _check(sig is imu.last_accel, "wrong signal in binding")
    _check(topic == "manta/ex_test/imu0/last_accel", f"unexpected default topic: {topic}")
    _check(proto == "zenoh", "default protocol must be zenoh")


def test_publish_explicit_topic() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    c.publish(imu.last_gyro, "custom/topic/path")
    _check(c.bindings[0][1] == "custom/topic/path", "explicit topic ignored")


def test_subscribe_records_binding() -> None:
    t = Thruster("forward", max_thrust=2.0)
    c = Craft("ex_test")
    c.subscribe(t.set_throttle, "manta/ex_test/cmd")
    _check(len(c.bindings) == 1, "subscribe didn't record")


def test_publish_rejects_in_signal() -> None:
    t = Thruster("forward", max_thrust=2.0)
    c = Craft("ex_test")
    try:
        c.publish(t.set_throttle)   # set_throttle is direction='in'
        raise AssertionError("publish on in-signal should have raised")
    except ValueError:
        pass


def test_subscribe_rejects_out_signal() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    try:
        c.subscribe(imu.last_accel)
        raise AssertionError("subscribe on out-signal should have raised")
    except ValueError:
        pass


def test_chaining_returns_craft() -> None:
    imu = IMU("imu0")
    t = Thruster("fwd", max_thrust=1.0)
    c = (Craft("chained")
         .publish(imu.last_accel)
         .publish(imu.last_gyro)
         .subscribe(t.set_throttle))
    _check(len(c.bindings) == 3, "chained calls didn't accumulate")


if __name__ == "__main__":
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
