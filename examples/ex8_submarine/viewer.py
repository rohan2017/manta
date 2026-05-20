"""Rerun viewer for ex8 — underwater AUV sim + EKF.

Renders TWO craft poses side by side:
  * Truth (solid, red)         from `manta/ex8/state`
  * EKF estimate (wireframe)   from `manta/ex8/estimate`,
                               surrounded by a sphere proportional to the
                               filter's position stddev.

Plus scalar plots for the per-axis position error and stddev.

Run alongside `./build/examples/ex8`:

    .venv/bin/python examples/ex8_submarine/viewer.py
"""

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

import rerun as rr
import zenoh


# Submarine geometry (rough cylinder along body +x).
SUB_LEN    = 1.5
SUB_RADIUS = 0.15
# Color: truth = warm red, estimate = cool cyan.
COLOR_TRUTH    = [220, 80,  80]
COLOR_ESTIMATE = [80,  200, 220]


def _quat_xyzw_from_telem(q_wxyz):
    """Telemetry q is (w, x, y, z); Rerun wants (x, y, z, w)."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def main() -> None:
    rr.init("manta_ex8", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ---- Static scene ----
    # Water surface at z = 0 (large flat blue plane).
    rr.log("world/surface",
           rr.Boxes3D(half_sizes=[[15.0, 15.0, 0.01]],
                      centers=[[0, 0, 0.0]],
                      colors=[[60, 110, 180, 80]]),
           static=True)

    # Sea-floor sketch ~12 m down to anchor depth perception.
    rr.log("world/seafloor",
           rr.Boxes3D(half_sizes=[[15.0, 15.0, 0.02]],
                      centers=[[0, 0, -12.0]],
                      colors=[[80, 70, 50]]),
           static=True)

    # Depth markers along -z to make the dive visible.
    for d in (-2.0, -5.0, -8.0, -10.0):
        rr.log(f"world/depth_marks/m{abs(int(d))}",
               rr.LineStrips3D(strips=[[[-2.0, 0.0, d], [2.0, 0.0, d]]],
                               colors=[[80, 100, 130]],
                               radii=[0.005]),
               static=True)

    # ---- Static hull geometry under each pose's transform ----
    # Truth body: cylinder along +x. Half-sizes for the bounding box —
    # Rerun doesn't ship a primitive cylinder so a thin box stands in.
    rr.log("world/truth/hull",
           rr.Boxes3D(half_sizes=[[SUB_LEN / 2.0, SUB_RADIUS, SUB_RADIUS]],
                      colors=[COLOR_TRUTH]),
           static=True)
    rr.log("world/truth/nose",
           rr.Points3D(positions=[[SUB_LEN / 2.0, 0.0, 0.0]],
                       radii=[SUB_RADIUS * 0.6],
                       colors=[[255, 230, 200]]),
           static=True)
    rr.log("world/truth",
           rr.archetypes.TransformAxes3D(0.4),
           static=True)

    # Estimate body: smaller wireframe-style box so the two don't fully
    # overlap when they coincide. Lighter alpha to read as "estimate".
    rr.log("world/estimate/hull",
           rr.Boxes3D(half_sizes=[[SUB_LEN / 2.0 + 0.02,
                                   SUB_RADIUS + 0.02,
                                   SUB_RADIUS + 0.02]],
                      colors=[[*COLOR_ESTIMATE, 120]]),
           static=True)
    rr.log("world/estimate",
           rr.archetypes.TransformAxes3D(0.35),
           static=True)

    # ---- Zenoh wiring ----
    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    state = {"truth": None, "estimate": None, "n_truth": 0, "n_est": 0}

    def _log_truth(d: dict) -> None:
        rr.log("world/truth",
               rr.Transform3D(translation=d["p"],
                              rotation=rr.Quaternion(
                                  xyzw=_quat_xyzw_from_telem(d["q"]))))
        # Velocity arrow at body origin.
        rr.log("world/truth/velocity",
               rr.Arrows3D(origins=[[0, 0, 0]],
                           vectors=[d["v"]],
                           colors=[[230, 200, 80]]))

    def _log_estimate(d: dict) -> None:
        p = d["p"]
        rr.log("world/estimate",
               rr.Transform3D(translation=p,
                              rotation=rr.Quaternion(
                                  xyzw=_quat_xyzw_from_telem(d["q"]))))
        rr.log("world/estimate/velocity",
               rr.Arrows3D(origins=[[0, 0, 0]],
                           vectors=[d["v"]],
                           colors=[[200, 230, 80]]))
        # 1-σ position uncertainty sphere, in world frame (not under the
        # estimate transform — its radius doesn't rotate with the body).
        psd = d.get("p_stddev", [0.0, 0.0, 0.0])
        r_unc = max(psd[0], psd[1], psd[2])
        rr.log("world/uncertainty",
               rr.Points3D(positions=[p],
                           radii=[r_unc],
                           colors=[[*COLOR_ESTIMATE, 60]]))

    def _log_plots(truth: dict, est: dict) -> None:
        # Position error per axis (estimate − truth).
        for i, axis in enumerate("xyz"):
            err = est["p"][i] - truth["p"][i]
            rr.log(f"plots/p_err_{axis}", rr.Scalars(err))
            rr.log(f"plots/p_stddev_{axis}",
                   rr.Scalars(est.get("p_stddev", [0, 0, 0])[i]))
        # Total position error magnitude.
        err_mag = math.sqrt(sum((est["p"][i] - truth["p"][i]) ** 2
                                for i in range(3)))
        rr.log("plots/p_err_mag", rr.Scalars(err_mag))
        # Depth (z) of both, for a side-by-side trace.
        rr.log("plots/depth_truth", rr.Scalars(truth["p"][2]))
        rr.log("plots/depth_est",   rr.Scalars(est["p"][2]))

    def on_truth(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload).decode("utf-8"))
        except Exception as e:
            print(f"viewer: bad truth payload ({e})", file=sys.stderr); return
        state["truth"] = d
        state["n_truth"] += 1
        if state["n_truth"] == 1:
            print(f"viewer: first truth sample  keys={sorted(d.keys())}",
                  file=sys.stderr)
        # We don't have a sim time on the truth feed (no `t` key), so the
        # viewer drives Rerun's time axis from a sample counter scaled
        # to the publish rate (~50 Hz).
        rr.set_time("sim_time", duration=state["n_truth"] / 50.0)
        _log_truth(d)
        if state["estimate"] is not None:
            _log_plots(d, state["estimate"])

    def on_estimate(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload).decode("utf-8"))
        except Exception as e:
            print(f"viewer: bad estimate payload ({e})", file=sys.stderr); return
        state["estimate"] = d
        state["n_est"] += 1
        if state["n_est"] == 1:
            print(f"viewer: first estimate sample  keys={sorted(d.keys())}",
                  file=sys.stderr)
        rr.set_time("sim_time", duration=state["n_est"] / 50.0)
        _log_estimate(d)
        if state["truth"] is not None:
            _log_plots(state["truth"], d)

    sub_t = session.declare_subscriber("manta/ex8/state",    on_truth)
    sub_e = session.declare_subscriber("manta/ex8/estimate", on_estimate)

    def shutdown(*_):
        sub_t.undeclare()
        sub_e.undeclare()
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("viewer: listening on manta/ex8/{state,estimate}. Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
