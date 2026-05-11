"""Rerun viewer for ex2 quadcopter.

Renders the craft body, body-frame axes, a ground grid, and one force
arrow per rotor scaled by the live throttle (telemetry on
`manta/ex2/state`). Run alongside `./build/examples/ex2_quadcopter`.
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

import rerun as rr
import zenoh

# ---- Pull rotor geometry from the sim config. ----
_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("_ex2_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_cfg)
except ModuleNotFoundError as e:
    sys.exit(f"viewer.py: failed to load config ({e}). "
             "Run via .venv/bin/python (it has manta_codegen installed).")

ROTORS              = _cfg.ROTORS                 # [(name, (x,y,z), cw?), ...]
MAX_THRUST_PER_PROP = _cfg.MAX_THRUST_PER_PROP    # N
ARM_L               = _cfg.ARM_L                  # m

ARROW_PER_N = 0.05          # rerun units per N. Full-throttle arrow ≈
                            # MAX_THRUST_PER_PROP * 0.05 ≈ 0.25 m, same
                            # order as the rotor arm so it reads.


def main() -> None:
    rr.init("manta_ex2", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Ground grid — flat plane at z=0, big enough to bound mild flight.
    rr.log("world/ground",
           rr.Boxes3D(half_sizes=[[5.0, 5.0, 0.005]],
                      centers=[[0, 0, -0.005]],
                      colors=[[60, 80, 60]]),
           static=True)

    # Craft body (procedural — flat-ish plate, X across +x/-x, Y across +y/-y).
    rr.log("world/quad/body",
           rr.Boxes3D(half_sizes=[[0.3, 0.3, 0.04]],
                      colors=[[200, 80, 80]]),
           static=True)
    # Body-frame coordinate axes.
    rr.log("world/quad",
           rr.archetypes.TransformAxes3D(0.3),
           static=True)
    # Rotor origin markers (body-local meters).
    for name, offset, _cw in ROTORS:
        rr.log(f"world/quad/rotors/{name}/origin",
               rr.Points3D(positions=[list(offset)],
                           radii=[0.04],
                           colors=[[80, 80, 80]]),
               static=True)

    cfg = zenoh.Config()
    session = zenoh.open(cfg)

    def on_state(sample: zenoh.Sample) -> None:
        try:
            d = json.loads(bytes(sample.payload))
        except Exception as e:
            print(f"bad payload: {e}", file=sys.stderr)
            return
        rr.set_time("sim_time", duration=float(d["t"]))

        rr.log("world/quad",
               rr.Transform3D(
                   translation=d["p"],
                   rotation=rr.Quaternion(xyzw=[d["q"][1], d["q"][2],
                                                d["q"][3], d["q"][0]])))

        # Per-rotor thrust arrows. Thrust is along body +z; arrow length
        # ∝ throttle. The codegen-emitted telemetry exposes each rotor's
        # throttle under `<name>.throttle`.
        origins: list[list[float]] = []
        vectors: list[list[float]] = []
        colors:  list[list[int]]   = []
        for name, offset, _cw in ROTORS:
            thr = float(d.get(name, {}).get("throttle", 0.0))
            length = thr * MAX_THRUST_PER_PROP * ARROW_PER_N
            origins.append(list(offset))
            vectors.append([0.0, 0.0, length])
            colors.append([255, 100, 100])
        rr.log("world/quad/rotors/forces",
               rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))

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
