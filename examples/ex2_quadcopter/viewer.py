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

ROTORS      = _cfg.ROTORS                        # [(name, (x,y,z), cw?), ...]
ARM_L       = _cfg.ARM_L                         # m
BLADE_CHORD = _cfg.BLADE_CHORD                   # m
BLADE_SPAN  = _cfg.BLADE_SPAN                    # m (motor center → tip)
OMEGA_REF   = 100.0                              # rad/s — typical hover
ARROW_AT_OMEGA_REF = 0.20                        # rerun units (≈ ARM_L)


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
    # Per-rotor static geometry: an origin marker plus two flat-plate
    # blades extending in ±x of the motor's *rotating* frame, each
    # `BLADE_SPAN × BLADE_CHORD` in the disk plane (thickness ≈ 0).
    # The rotor's Transform3D (logged per tick) translates to the body-
    # frame attach point AND rotates by the motor's live angle about
    # +z — so children logged here in motor coords visibly spin.
    THIN_Z = 0.001
    blade_half = [BLADE_SPAN / 2.0, BLADE_CHORD / 2.0, THIN_Z]
    for name, _offset, cw in ROTORS:
        rr.log(f"world/quad/rotors/{name}/origin",
               rr.Points3D(positions=[[0.0, 0.0, 0.0]],
                           radii=[0.02],
                           colors=[[80, 80, 80]]),
               static=True)
        # Two flat-plate blades, 180° apart in the disk plane. Color
        # one yellow, the other cyan so the rotation is visible.
        rr.log(f"world/quad/rotors/{name}/blade_a",
               rr.Boxes3D(centers=[[+BLADE_SPAN / 2.0, 0.0, 0.0]],
                          half_sizes=[blade_half],
                          colors=[[235, 220, 80]]),
               static=True)
        rr.log(f"world/quad/rotors/{name}/blade_b",
               rr.Boxes3D(centers=[[-BLADE_SPAN / 2.0, 0.0, 0.0]],
                          half_sizes=[blade_half],
                          colors=[[80, 200, 220]]),
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

        # Per-rotor Transform3D: position the rotor at its body-frame
        # mount AND spin it by the motor's live angle about +z. The
        # static blade plates logged earlier inherit this transform, so
        # they rotate with the motor.
        for name, offset, _cw in ROTORS:
            angle = float(d.get(f"{name}_motor", {}).get("angle", 0.0))
            # AngleAxis(angle, +z) as a quaternion (w, x, y, z) =
            # (cos α/2, 0, 0, sin α/2).
            half = angle * 0.5
            rr.log(f"world/quad/rotors/{name}",
                   rr.Transform3D(
                       translation=list(offset),
                       rotation=rr.Quaternion(xyzw=[0.0, 0.0,
                                                    math.sin(half),
                                                    math.cos(half)])))

        # Per-rotor "thrust" arrows. Logged at body-frame level (above
        # the spinning rotor frame) so the arrows don't sweep with the
        # blades. Proxy: arrow length ∝ (ω / OMEGA_REF)² since lift
        # scales as ω². Color: red CCW, blue CW.
        origins: list[list[float]] = []
        vectors: list[list[float]] = []
        colors:  list[list[int]]   = []
        for name, offset, _cw in ROTORS:
            rate = float(d.get(f"{name}_motor", {}).get("rate", 0.0))
            mag2 = (rate / OMEGA_REF) ** 2
            length = ARROW_AT_OMEGA_REF * mag2
            origins.append(list(offset))
            vectors.append([0.0, 0.0, length])
            colors.append([255, 100, 100] if rate >= 0.0 else [100, 140, 255])
        rr.log("world/quad/rotor_forces",
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
