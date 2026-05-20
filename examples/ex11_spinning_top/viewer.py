"""Rerun viewer for ex11 spinning-top precession demo.

Renders the stick body, the heavy flywheel (spinning with the motor
angle), the two end colliders, the ground plane, and a body-frame axes
gizmo. Telemetry comes in on Zenoh topic `manta/ex11/state` — JSON with
`t`, `p` (position), `q` (orientation w,x,y,z), `v`, `w`, plus a
`fly_motor` block carrying the joint's `angle`, `rate`, `accel`.

Run alongside `./build/examples/ex11_spinning_top`:

    .venv/bin/python examples/ex11_spinning_top/viewer.py
"""

import importlib.util
import json
import math
import os
import pathlib
import signal
import sys

# rr.init(spawn=True) shells out to a `rerun` binary that ships next to
# rerun-sdk inside the venv. Stitch the interpreter's directory onto
# PATH so the viewer works whether or not `.venv` is activated.
_interpreter_dir = pathlib.Path(sys.executable).parent
if str(_interpreter_dir) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_interpreter_dir}{os.pathsep}{os.environ.get('PATH', '')}"

import numpy as np
import rerun as rr
import zenoh

# ---- Pull geometry constants from the sim config so the viewer matches. ----
_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("_ex11_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_cfg)
except ModuleNotFoundError as e:
    sys.exit(f"viewer.py: failed to load config ({e}). "
             "Run via .venv/bin/python (it has manta_codegen installed).")

L_STICK     = _cfg.L_STICK
FLY_R       = _cfg.FLY_R
MOTOR_Z     = _cfg.MOTOR_Z
COLLIDER_R  = _cfg.COLLIDER_R
THRUSTER_Z  = _cfg.THRUSTER_Z


def main() -> None:
    rr.init("manta_ex11", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Ground grid — flat plane at z=0.
    rr.log("world/ground",
           rr.Boxes3D(half_sizes=[[2.0, 2.0, 0.005]],
                      centers=[[0, 0, -0.005]],
                      colors=[[60, 80, 60]]),
           static=True)

    # Static craft geometry, logged under world/top. Re-parented each
    # tick by the body Transform3D so everything moves together.
    #
    # Stick: a thin tall box along body z, full length L_STICK, light
    # color so the flywheel reads against it.
    STICK_HALF_W = 0.012
    rr.log("world/top/stick",
           rr.Boxes3D(half_sizes=[[STICK_HALF_W, STICK_HALF_W, L_STICK / 2.0]],
                      centers=[[0.0, 0.0, 0.0]],
                      colors=[[180, 180, 200]]),
           static=True)

    # End-cap colliders: small spheres at +L/2 and -L/2 along body z.
    rr.log("world/top/colliders",
           rr.Points3D(positions=[[0.0, 0.0, +L_STICK / 2.0],
                                  [0.0, 0.0, -L_STICK / 2.0]],
                       radii=[COLLIDER_R, COLLIDER_R],
                       colors=[[100, 100, 220], [220, 100, 100]]),
           static=True)

    # Thruster nozzle marker at the top of the stick (offset in +x to
    # show which way the kick fires).
    rr.log("world/top/thruster",
           rr.Boxes3D(half_sizes=[[0.025, 0.015, 0.015]],
                      centers=[[0.04, 0.0, THRUSTER_Z]],
                      colors=[[220, 140, 50]]),
           static=True)

    # Body-frame axes gizmo at body origin.
    rr.log("world/top",
           rr.archetypes.TransformAxes3D(0.2),
           static=True)

    # Flywheel: a flat disk of radius FLY_R in the xy-plane of the
    # *motor* frame (so it spins with the joint angle). Two halves
    # colored differently so the rotation is visible. Each half is a
    # half-disk — implemented here as a thin box approximation rotated
    # in main loop. Simpler version: render as a thin cylinder via
    # Boxes3D in xy, half +y vs half -y.
    DISK_THICK = 0.005
    rr.log("world/top/fly_motor/disk_a",
           rr.Boxes3D(half_sizes=[[FLY_R, FLY_R / 2.0, DISK_THICK]],
                      centers=[[0.0, +FLY_R / 2.0, 0.0]],
                      colors=[[230, 200, 60]]),
           static=True)
    rr.log("world/top/fly_motor/disk_b",
           rr.Boxes3D(half_sizes=[[FLY_R, FLY_R / 2.0, DISK_THICK]],
                      centers=[[0.0, -FLY_R / 2.0, 0.0]],
                      colors=[[60, 180, 200]]),
           static=True)
    # Hub marker so the spin direction reads even when the disk is at
    # a near-symmetric pose.
    rr.log("world/top/fly_motor/hub",
           rr.Points3D(positions=[[FLY_R * 0.65, 0.0, 0.0]],
                       radii=[0.012],
                       colors=[[40, 40, 40]]),
           static=True)

    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    state = {"n": 0, "kick_armed": True}

    def on_state(sample: zenoh.Sample) -> None:
        payload = bytes(sample.payload).decode("utf-8", errors="replace")
        try:
            d = json.loads(payload)
        except Exception as e:
            print(f"viewer: bad payload ({e}): {payload[:200]}", file=sys.stderr)
            return

        state["n"] += 1
        if state["n"] == 1:
            print(f"viewer: first state received  t={d.get('t', '?')}  "
                  f"keys={sorted(d.keys())}", file=sys.stderr)

        t = float(d["t"])
        rr.set_time("sim_time", duration=t)

        # Body pose. Telemetry q is (w, x, y, z); Rerun wants (x, y, z, w).
        p = d["p"]
        q = d["q"]
        rr.log("world/top",
               rr.Transform3D(translation=p,
                              rotation=rr.Quaternion(
                                  xyzw=[q[1], q[2], q[3], q[0]])))

        # Motor transform: translate to mount, rotate about local +z by
        # the joint angle. The static disk geometry under fly_motor
        # inherits this transform, so it spins.
        angle = float(d.get("fly_motor", {}).get("angle", 0.0))
        half  = angle * 0.5
        rr.log("world/top/fly_motor",
               rr.Transform3D(
                   translation=[0.0, 0.0, MOTOR_Z],
                   rotation=rr.Quaternion(
                       xyzw=[0.0, 0.0, math.sin(half), math.cos(half)])))

        # Tilt arrow: a red vector at the body origin pointing along
        # body +z in scene frame, showing how the stick is leaning over
        # time. Useful for reading the precession at a glance.
        # body +z in scene = R · ẑ. Quaternion (w, x, y, z) rotating ẑ:
        #   x' = 2(qx·qz + qw·qy)
        #   y' = 2(qy·qz - qw·qx)
        #   z' = 1 - 2(qx² + qy²)
        z_x = 2.0 * (q[1] * q[3] + q[0] * q[2])
        z_y = 2.0 * (q[2] * q[3] - q[0] * q[1])
        z_z = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        tilt_len = 0.6
        rr.log("world/tilt_arrow",
               rr.Arrows3D(origins=[p],
                           vectors=[[z_x * tilt_len, z_y * tilt_len,
                                     z_z * tilt_len]],
                           colors=[[230, 80, 80]]))

        # Flywheel rate (for the scalar plot panel).
        rate = float(d.get("fly_motor", {}).get("rate", 0.0))
        rr.log("plots/fly_rate", rr.Scalars(rate))
        # Tilt angle from vertical.
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, z_z))))
        rr.log("plots/tilt_deg", rr.Scalars(tilt_deg))

    sub = session.declare_subscriber("manta/ex11/state", on_state)

    def shutdown(*_) -> None:
        sub.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("viewer: listening on manta/ex11/state. Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
