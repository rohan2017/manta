"""rerun.io viewer for ex6 — overlays truth and EKF estimate.

Subscribes to both 'manta/ex6/state' (truth) and 'manta/ex6/estimate' (EKF
output) and logs them to rerun side by side:
  - white craft pose + trajectory line for truth
  - cyan trajectory line for estimate, plus 1-σ position uncertainty
    as colored boxes whose half-extents are sqrt(P_diag[0..2])
  - scalar plots for per-axis position error and velocity covariance

Usage:
    pip install zenoh rerun-sdk  # (already in the project venv)
    python examples/ex6_estimator_demo/viewer.py
"""

from __future__ import annotations

import json
import math
import signal
import sys

import rerun as rr
import zenoh


def _log_pose(path: str, p, q, color, axis_length: float = 0.2) -> None:
    rr.log(
        path,
        rr.Transform3D(
            translation=p,
            rotation=rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]]),
            axis_length=axis_length,
        ),
    )
    rr.log(
        f"{path}/body",
        rr.Boxes3D(half_sizes=[[0.15, 0.15, 0.05]], colors=[color]),
    )


def main() -> None:
    rr.init("manta_ex6", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/grid",
        rr.Boxes3D(half_sizes=[[20, 20, 0.01]], colors=[[60, 60, 60]]),
        static=True,
    )

    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    truth_traj: list[list[float]] = []
    est_traj:   list[list[float]] = []
    MAX_TRAIL = 600   # ~12 s at 50 Hz

    def on_truth(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload))
        except Exception:
            return
        rr.set_time_seconds("sim_time", d["t"])
        _log_pose("world/truth", d["p"], d["q"], [240, 240, 240])
        truth_traj.append(d["p"])
        if len(truth_traj) > MAX_TRAIL:
            del truth_traj[: len(truth_traj) - MAX_TRAIL]
        if len(truth_traj) >= 2:
            rr.log(
                "world/truth_trail",
                rr.LineStrips3D(
                    [truth_traj], colors=[[240, 240, 240]], radii=[0.02]),
            )
        rr.log("plots/truth/vx", rr.Scalar(d["v"][0]))
        rr.log("plots/truth/vy", rr.Scalar(d["v"][1]))
        rr.log("plots/truth/vz", rr.Scalar(d["v"][2]))

    def on_estimate(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload))
        except Exception:
            return
        rr.set_time_seconds("sim_time", d["t"])
        # The EKF currently does not estimate orientation (state is p, v
        # only); show pose with identity quaternion.
        _log_pose("world/estimate", d["p"], [1, 0, 0, 0], [80, 200, 230],
                  axis_length=0.15)

        est_traj.append(d["p"])
        if len(est_traj) > MAX_TRAIL:
            del est_traj[: len(est_traj) - MAX_TRAIL]
        if len(est_traj) >= 2:
            rr.log(
                "world/estimate_trail",
                rr.LineStrips3D(
                    [est_traj], colors=[[80, 200, 230]], radii=[0.018]),
            )

        # 1-σ position uncertainty as a transparent box centered on estimate.
        p_var = d["P"][0:3]
        sigmas = [math.sqrt(max(v, 0.0)) for v in p_var]
        rr.log(
            "world/estimate/uncertainty",
            rr.Boxes3D(
                centers=[[0, 0, 0]],
                half_sizes=[sigmas],
                colors=[[80, 200, 230, 80]],
            ),
        )
        rr.log("plots/cov/pz", rr.Scalar(p_var[2]))
        rr.log("plots/cov/vz", rr.Scalar(d["P"][5]))
        rr.log("plots/est/vx", rr.Scalar(d["v"][0]))
        rr.log("plots/est/vy", rr.Scalar(d["v"][1]))
        rr.log("plots/est/vz", rr.Scalar(d["v"][2]))

    sub_t = session.declare_subscriber("manta/ex6/state",    on_truth)
    sub_e = session.declare_subscriber("manta/ex6/estimate", on_estimate)

    def shutdown(*_: object) -> None:
        sub_t.undeclare()
        sub_e.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("ex6 viewer: subscribed to manta/ex6/state and manta/ex6/estimate. "
          "Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
