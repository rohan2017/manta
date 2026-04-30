"""Rerun viewer for ex0 free-flight craft.

Subscribes to 'manta/ex0/state' on Zenoh, logs craft pose + per-thruster
state to rerun. Telemetry JSON is the codegen-emitted nested form:

    { "t":..., "p":[x,y,z], "q":[w,x,y,z], "v":[...], "w":[...],
      "tx_p": {"throttle": 0.5}, ... }

Usage:
    pip install zenoh rerun-sdk
    python viewer.py
"""

import json
import signal
import sys

import rerun as rr
import zenoh


# Each thruster's name → unit direction in part frame (must match craft.py).
THRUSTERS: dict[str, tuple[float, float, float]] = {
    "tx_p": ( 1.0,  0.0,  0.0),
    "tx_n": (-1.0,  0.0,  0.0),
    "ty_p": ( 0.0,  1.0,  0.0),
    "ty_n": ( 0.0, -1.0,  0.0),
    "tz_p": ( 0.0,  0.0,  1.0),
    "tz_n": ( 0.0,  0.0, -1.0),
}


def main() -> None:
    rr.init("manta_ex0", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/grid",
           rr.Boxes3D(half_sizes=[[5, 5, 0.01]], colors=[[60, 60, 60]]),
           static=True)

    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    def on_state(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload))
        except Exception as e:
            print(f"bad payload: {e}", file=sys.stderr)
            return
        rr.set_time_seconds("sim_time", d["t"])

        rr.log(
            "world/craft",
            rr.Transform3D(
                translation=d["p"],
                rotation=rr.Quaternion(
                    xyzw=[d["q"][1], d["q"][2], d["q"][3], d["q"][0]]),
                axis_length=0.3,
            ),
        )
        rr.log("world/craft/body",
               rr.Boxes3D(half_sizes=[[0.15, 0.15, 0.05]],
                          colors=[[200, 200, 80]]))

        # Thruster plumes — points opposite the thrust direction, length grows
        # with throttle. Lives under world/craft/<name> so it inherits the
        # craft transform.
        for name, direction in THRUSTERS.items():
            sub = d.get(name) or {}
            thr = float(sub.get("throttle", 0.0))
            rr.log(f"plots/throttle/{name}", rr.Scalar(thr))
            if thr > 1e-3:
                length = 0.4 * thr
                vec = [-direction[0] * length,
                       -direction[1] * length,
                       -direction[2] * length]
                rr.log(f"world/craft/{name}_plume",
                       rr.Arrows3D(origins=[[0, 0, 0]],
                                   vectors=[vec],
                                   colors=[[255, 120, 50]]))
            else:
                # Clear stale plume from the previous tick.
                rr.log(f"world/craft/{name}_plume", rr.Clear(recursive=False))

        rr.log("plots/vel/x", rr.Scalar(d["v"][0]))
        rr.log("plots/vel/y", rr.Scalar(d["v"][1]))
        rr.log("plots/vel/z", rr.Scalar(d["v"][2]))

    sub = session.declare_subscriber("manta/ex0/state", on_state)

    def shutdown(*_) -> None:
        print("viewer: shutting down")
        sub.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("viewer: listening on manta/ex0/state. Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
