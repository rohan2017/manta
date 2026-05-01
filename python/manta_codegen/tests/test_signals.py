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
from manta_codegen.parts.sensor.dvl import DVL
from manta_codegen.parts.actuator.thruster import Thruster
from manta_codegen.parts.actuator.prop_thruster import PropThruster
from manta_codegen.parts.actuator.gimbaled_thruster import GimbaledThruster
from manta_codegen.parts.articulation.motor import Motor
from manta_codegen.signal import Binding, BoundSignal


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
    b = c.bindings[0]
    _check(isinstance(b, Binding), "binding must be a Binding instance")
    _check(list(b.members.keys()) == ["last_accel"], "single-signal struct must have one member named after signal")
    _check(b.members["last_accel"] is imu.last_accel, "wrong signal in binding")
    _check(b.topic == "manta/ex_test/imu0/last_accel", f"unexpected default topic: {b.topic}")
    _check(b.protocol == "zenoh", "default protocol must be zenoh")
    _check(b.encoding == "json", "default encoding must be json")
    _check(b.direction == "out", "single-signal direction must propagate")
    _check(b.total_floats == 3, f"vec3 signal should have 3 floats, got {b.total_floats}")


def test_publish_explicit_topic() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    c.publish(imu.last_gyro, "custom/topic/path")
    _check(c.bindings[0].topic == "custom/topic/path", "explicit topic ignored")


def test_publish_struct_bundle() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    c.publish({"accel": imu.last_accel, "gyro": imu.last_gyro}, "manta/ex_test/imu_bundle")
    b = c.bindings[0]
    _check(set(b.members) == {"accel", "gyro"}, f"unexpected members: {list(b.members)}")
    _check(b.total_floats == 6, f"two vec3 should be 6 floats, got {b.total_floats}")
    _check(b.direction == "out", "all-out bundle must have direction=out")


def test_publish_struct_requires_explicit_topic() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    try:
        c.publish({"accel": imu.last_accel, "gyro": imu.last_gyro})
        raise AssertionError("multi-member bundle without topic should have raised")
    except ValueError:
        pass


def test_struct_with_mixed_directions_rejected() -> None:
    imu = IMU("imu0")
    t = Thruster("fwd", max_thrust=1.0)
    c = Craft("ex_test")
    try:
        c.publish({"accel": imu.last_accel, "throttle": t.set_throttle},
                  "manta/ex_test/mixed")
        raise AssertionError("mixed-direction struct should have raised")
    except ValueError:
        pass


def test_subscribe_struct_bundle() -> None:
    t1 = Thruster("a", max_thrust=1.0)
    t2 = Thruster("b", max_thrust=1.0)
    c = Craft("ex_test")
    c.subscribe({"left": t1.set_throttle, "right": t2.set_throttle},
                "manta/ex_test/throttles")
    _check(len(c.bindings) == 1, "bundled subscribe didn't record")
    b = c.bindings[0]
    _check(b.direction == "in", "all-in bundle must have direction=in")


def test_explicit_encoding() -> None:
    imu = IMU("imu0")
    c = Craft("ex_test")
    c.publish(imu.last_accel, encoding="binary")
    _check(c.bindings[0].encoding == "binary", "encoding override ignored")


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


def test_dvl_signals_attached() -> None:
    d = DVL("dvl0")
    _check(d.last_velocity.direction == "out", "DVL last_velocity must be out")
    _check(d.set_measurement.direction == "in", "DVL set_measurement must be in")
    _check(d.last_velocity.signal.n_floats == 3, "DVL velocity should be 3 floats")


def test_motor_signals_attached() -> None:
    m = Motor("hinge")
    for name in ("angle", "rate", "accel"):
        _check(getattr(m, name).direction == "out", f"Motor {name} must be out")
    _check(m.set_torque.direction == "in", "Motor set_torque must be in")


def test_prop_thruster_inherits_thruster_signals() -> None:
    p = PropThruster("rotor", max_thrust=2.0)
    _check(p.throttle.direction == "out", "PropThruster inherits throttle")
    _check(p.set_throttle.direction == "in", "PropThruster inherits set_throttle")


def test_gimbaled_thruster_extends_thruster_signals() -> None:
    g = GimbaledThruster("gimbal", max_thrust=2.0)
    # Inherited from Thruster:
    _check(g.throttle.direction == "out", "GimbaledThruster inherits throttle")
    _check(g.set_throttle.direction == "in", "GimbaledThruster inherits set_throttle")
    # Added by GimbaledThruster:
    _check(g.pitch.direction == "out", "pitch must be out")
    _check(g.yaw.direction == "out", "yaw must be out")
    _check(g.set_gimbal.direction == "in", "set_gimbal must be in")
    _check(g.set_gimbal.signal.n_floats == 2, "set_gimbal should be 2 floats")


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
