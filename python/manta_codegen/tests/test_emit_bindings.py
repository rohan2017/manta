"""Phase 2f: codegen emits Zenoh sub/pub from Craft.bindings.

Exercises emit_main_cpp on a synthetic craft that uses ONLY the new binding
API. Verifies the legacy bundled-state topic is suppressed and that each
binding produces a publisher/subscriber + main-loop apply/publish snippet.

No C++ build — pure-Python output inspection. The matching compile-time
sync test (tests/test_codegen_signals.gen.cpp from phase 2e) handles the
"Python signal expressions actually compile against the C++ part API"
side; this file handles "the emitter wires bindings into the right
locations in the generated main."
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from manta_codegen.core import Craft
from manta_codegen.emit.main import emit_main_cpp
from manta_codegen.parts.actuator.thruster import Thruster
from manta_codegen.parts.sensor.imu import IMU


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _build_craft_with_bindings() -> Craft:
    imu = IMU("imu0")
    thr = Thruster("fwd", max_thrust=2.0)
    c = Craft("ex_bind")
    c.root.add(imu)
    c.root.add(thr)
    c.publish(imu.last_accel, "ex/imu/accel")
    c.publish({"accel": imu.last_accel, "gyro": imu.last_gyro}, "ex/imu/bundle")
    c.subscribe(thr.set_throttle, "ex/cmd/throttle")
    return c


def test_emit_main_with_bindings_produces_publishers() -> None:
    c = _build_craft_with_bindings()
    src = emit_main_cpp(c)
    # One publisher per out-binding (we registered two).
    _check('declare_publisher(zenoh::KeyExpr("ex/imu/accel"))' in src,
           "single-signal publisher missing")
    _check('declare_publisher(zenoh::KeyExpr("ex/imu/bundle"))' in src,
           "bundled publisher missing")
    # JSON encoding strings present.
    _check('"\\"accel\\":[' in src, "bundled JSON should include accel key")
    _check('"\\"gyro\\":[' in src, "bundled JSON should include gyro key")


def test_emit_main_with_bindings_produces_subscriber() -> None:
    c = _build_craft_with_bindings()
    src = emit_main_cpp(c)
    _check('declare_subscriber(' in src, "subscriber declaration missing")
    _check('"ex/cmd/throttle"' in src, "subscribe topic missing")
    _check('craft.fwd().set_throttle(' in src, "set_throttle apply missing")


def test_emit_main_with_bindings_suppresses_legacy_state_topic() -> None:
    c = _build_craft_with_bindings()
    src = emit_main_cpp(c)
    # No bundled `state_pub` for the auto-bundled `manta/<craft>/state` topic.
    _check("state_pub" not in src,
           "with bindings active, the legacy state_pub must not be emitted")
    _check("ex_bindTelemetry telem" not in src and "capture_ex_bind_telemetry" not in src,
           "legacy telemetry capture must be suppressed under bindings")


def test_emit_main_legacy_path_still_active_when_no_bindings() -> None:
    # Sanity: a craft with no bindings still gets the legacy bundled-state path
    # so existing examples keep building.
    imu = IMU("imu0")
    c = Craft("ex_legacy")
    c.root.add(imu)
    src = emit_main_cpp(c)
    _check("state_pub" in src, "legacy state_pub disappeared without bindings")


def test_apply_substitutes_payload_indices_per_member() -> None:
    """Bundle two scalar in-signals → expect bind_0_payload[0] for the first
    member and bind_0_payload[1] for the second."""
    t1 = Thruster("a", max_thrust=1.0)
    t2 = Thruster("b", max_thrust=1.0)
    c = Craft("ex_pair")
    c.root.add(t1)
    c.root.add(t2)
    c.subscribe({"left": t1.set_throttle, "right": t2.set_throttle},
                "ex/throttles")
    src = emit_main_cpp(c)
    _check("craft.a().set_throttle(bind_0_payload[0])" in src,
           "first member should consume payload[0]")
    _check("craft.b().set_throttle(bind_0_payload[1])" in src,
           "second member should consume payload[1]")


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
