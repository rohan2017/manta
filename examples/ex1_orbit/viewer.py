"""Rerun viewer for ex1 orbital craft.

Renders Earth as a sphere and the craft's orbit trail. Positions are scaled
by SCALE so the scene fits comfortably in rerun's default view.
"""

import collections
import json
import math
import signal
import sys

import rerun as rr
import zenoh

EARTH_RADIUS = 6.371e6
SCALE = 1.0 / 1.0e6  # 1 unit per Mm so Earth ≈ 6.371 units


def main() -> None:
    rr.init("manta_ex1", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # Earth sphere: rerun has no native sphere primitive in v0.x; use a Points3D
    # with a fat radius, or better, an icosphere mesh. For simplicity use a
    # large opaque point and an arrow for the rotation axis.
    rr.log("world/earth",
           rr.Points3D(
               positions=[[0, 0, 0]],
               radii=[EARTH_RADIUS * SCALE],
               colors=[[60, 100, 180]]),
           static=True)
    rr.log("world/earth_axis",
           rr.Arrows3D(
               origins=[[0, 0, 0]],
               vectors=[[0, 0, EARTH_RADIUS * 1.3 * SCALE]],
               colors=[[230, 230, 80]]),
           static=True)

    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    trail: collections.deque = collections.deque(maxlen=2000)

    def on_state(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload))
        except Exception as e:
            print(f"bad payload: {e}", file=sys.stderr)
            return
        rr.set_time_seconds("sim_time", d["t"])

        p_scaled = [d["p"][0] * SCALE, d["p"][1] * SCALE, d["p"][2] * SCALE]
        trail.append(p_scaled)

        rr.log(
            "world/craft",
            rr.Transform3D(
                translation=p_scaled,
                rotation=rr.Quaternion(xyzw=[d["q"][1], d["q"][2], d["q"][3], d["q"][0]]),
                axis_length=0.3,
            ),
        )
        rr.log("world/craft/body",
               rr.Points3D(positions=[[0, 0, 0]], radii=[0.05], colors=[[230, 230, 230]]))
        if len(trail) >= 2:
            rr.log("world/orbit_trail",
                   rr.LineStrips3D([list(trail)], colors=[[150, 230, 150]]))

        # Altitude in km (above mean radius)
        r = math.sqrt(d["p"][0]**2 + d["p"][1]**2 + d["p"][2]**2)
        rr.log("plots/altitude_km", rr.Scalar((r - EARTH_RADIUS) / 1000.0))
        # Speed
        v = math.sqrt(d["v"][0]**2 + d["v"][1]**2 + d["v"][2]**2)
        rr.log("plots/speed_mps", rr.Scalar(v))

    sub = session.declare_subscriber("manta/ex1/state", on_state)

    def shutdown(*_) -> None:
        sub.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("viewer: listening on manta/ex1/state. Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
