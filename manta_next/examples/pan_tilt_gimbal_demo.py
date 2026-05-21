"""Pan-tilt gimbal demo — nested Joints driving two independent rotors.

A camera-style gimbal: an outer `pan` Joint (yaw, about +z) hosts an
inner `tilt` Joint (pitch, about +y). The tilt joint hosts the camera
mass at the gimbal head. Both joints accept commanded torques (in
saturating mode), and each joint's reaction propagates correctly
through the chain — the tilt actuator pushes against the pan rotor,
and the pan actuator pushes against the body.

Demonstrates the M20 milestone capabilities:
  * Nested CompositePart hierarchy (Joint hosting Joint).
  * Symbolic kinematic pass tracking the joint chain.
  * Symbolic inertia rollup: the camera's body-frame position and
    inertia contribution depend on both joint angles.
  * Two-stage wrench aggregation: tilt actuator → pan rotor → body.

Run::

    .venv/bin/python -m manta_next.examples.pan_tilt_gimbal_demo
"""

from __future__ import annotations

import numpy as np

from manta_next import Craft
from manta_next.parts import Joint, Mass


def _build_gimbal() -> Craft:
    c = Craft("gimbal")

    # Heavy chassis — large MOI so the body's response to gimbal
    # reactions is small. The print-out below tracks both joint angles
    # and the body's angular velocity to show the (small) coupling.
    c.add(Mass("chassis", mass=2.0, moi=(0.05, 0.05, 0.05)))

    # Pan stage: outer Joint about +z. Hosts the tilt-stage mount.
    pan = Joint("pan", mode="saturating", stall_torque=1.0,
                axis=(0.0, 0.0, 1.0))
    # The pan rotor's own inertia (motor housing, bearings, etc.).
    pan.add(Mass("pan_housing",
                 mass=0.1, moi=(0.0005, 0.0005, 0.001)))

    # Tilt stage: inner Joint about +y. Mounted at z=0.05 above pan
    # so the rotor chain has a real lever arm. Hosts the camera.
    tilt = Joint("tilt", mode="saturating", stall_torque=0.5,
                 axis=(0.0, 1.0, 0.0),
                 transform=(0.0, 0.0, 0.05))
    tilt.add(Mass("camera",
                  mass=0.2, moi=(0.0008, 0.0008, 0.0002),
                  transform=(0.05, 0.0, 0.0)))
    pan.add(tilt)
    c.add(pan)

    return c


def main() -> None:
    c = _build_gimbal()
    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, 0.0))
    state = c.initial_state()

    # Phase 1: drive PAN only (no tilt command) — pan accelerates, body
    # gets a small counter-yaw reaction.
    state["pan.torque_cmd"]  = 0.3
    state["tilt.torque_cmd"] = 0.0
    print("=== Phase 1: pan.torque_cmd = 0.3, tilt.torque_cmd = 0 ===")
    print(f"{'t (s)':>6} {'pan.angle':>10} {'pan.rate':>10} "
          f"{'tilt.angle':>11} {'tilt.rate':>10} {'body ω_z':>10}")
    for step in range(1, 501):
        out = tick(dt=0.001, **state)
        state = {**state, **out}
        if step % 100 == 0:
            t = step * 0.001
            print(f"{t:6.2f} "
                  f"{state['pan.angle']:10.4f} {state['pan.rate']:10.4f} "
                  f"{state['tilt.angle']:11.4f} {state['tilt.rate']:10.4f} "
                  f"{state['angular_velocity'][2]:10.4f}")

    # Phase 2: zero pan, drive TILT — tilt accelerates, pan rotor is
    # nudged by the tilt reaction (because the tilt actuator pushes on
    # the pan rotor's frame). Watch pan.rate move a touch.
    state["pan.torque_cmd"]  = 0.0
    state["tilt.torque_cmd"] = 0.2
    print()
    print("=== Phase 2: pan.torque_cmd = 0, tilt.torque_cmd = 0.2 ===")
    print(f"{'t (s)':>6} {'pan.angle':>10} {'pan.rate':>10} "
          f"{'tilt.angle':>11} {'tilt.rate':>10} {'body ω_z':>10}")
    for step in range(501, 1001):
        out = tick(dt=0.001, **state)
        state = {**state, **out}
        if step % 100 == 0:
            t = step * 0.001
            print(f"{t:6.2f} "
                  f"{state['pan.angle']:10.4f} {state['pan.rate']:10.4f} "
                  f"{state['tilt.angle']:11.4f} {state['tilt.rate']:10.4f} "
                  f"{state['angular_velocity'][2]:10.4f}")


if __name__ == "__main__":
    main()
