"""Rerun viewer for ex2 quadcopter."""

import json
import signal
import sys

import rerun as rr
import zenoh


def main() -> None:
    rr.init("manta_ex2", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/ground",
           rr.Boxes3D(half_sizes=[[10, 10, 0.01]], colors=[[60, 80, 60]]),
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
            "world/quad",
            rr.Transform3D(
                translation=d["p"],
                rotation=rr.Quaternion(xyzw=[d["q"][1], d["q"][2], d["q"][3], d["q"][0]]),
                axis_length=0.3,
            ),
        )
        rr.log("world/quad/body",
               rr.Boxes3D(half_sizes=[[0.3, 0.3, 0.04]], colors=[[200, 80, 80]]))
        rr.log("plots/gyro/wx", rr.Scalar(d["w"][0]))
        rr.log("plots/gyro/wy", rr.Scalar(d["w"][1]))
        rr.log("plots/gyro/wz", rr.Scalar(d["w"][2]))
        rr.log("plots/pos/z",   rr.Scalar(d["p"][2]))

    sub = session.declare_subscriber("manta/ex2/state", on_state)

    def shutdown(*_) -> None:
        sub.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("viewer: listening on manta/ex2/state. Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
