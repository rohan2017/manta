"""Rerun viewer for ex1_orbit (lunar orbit, 100 m altitude).

Renders only what's needed to see the sim:
  * The Moon as a large sphere at world origin.
  * The craft (Apollo LM model) with body-frame coordinate axes.
  * One arrow per thruster, scaled by current throttle.

The orbit altitude (100 m) is microscopic against the Moon's 1737 km
radius — at physical scale the craft would sit visually on the surface
and you'd see no motion. The viewer exaggerates altitude by
`ALT_EXAGGERATE` so the orbit reads clearly: physical distance above
the Moon's mean surface is multiplied before mapping to rerun units.
The Moon itself is rendered at physical scale via `SCALE`.

Run:
    .venv/bin/python examples/ex1_orbit/viewer.py
"""

import importlib.util
import json
import math
import os
import pathlib
import signal
import sys

# rr.init(spawn=True) needs the `rerun` viewer binary on PATH. When this
# script is run via .venv/bin/python without activating the venv, the
# binary sits next to the interpreter; stitch its dir onto PATH.
_interpreter_dir = pathlib.Path(sys.executable).parent
if str(_interpreter_dir) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_interpreter_dir}{os.pathsep}{os.environ.get('PATH', '')}"

import rerun as rr
import zenoh

# ---- Pull thruster + Moon geometry from the sim config. ----
_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("_ex1_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_cfg)
except ModuleNotFoundError as e:
    sys.exit(f"viewer.py: failed to load config ({e}). "
             "Run via .venv/bin/python (it has manta_codegen installed).")

THRUSTERS   = _cfg.THRUSTERS
MAX_THRUST  = _cfg.MAX_THRUST
LEVER_ARM   = _cfg.LEVER_ARM
MOON_RADIUS = _cfg.MOON_RADIUS
ALTITUDE    = _cfg.ALTITUDE

SCALE          = 1.0 / 1.0e6        # 1 rerun unit per Mm — Moon ≈ 1.737 units
ALT_EXAGGERATE = 5.0e3              # multiplier on (altitude above Moon surface)
                                    # before scaling into rerun units. At 100 m
                                    # physical, this lifts the orbit visibly off
                                    # the Moon's surface.
CRAFT_SCALE    = 0.05               # rerun units per body-frame meter. Sized
                                    # so the 6-m craft is ~0.3 rerun units
                                    # (~17% of Moon radius — visible without
                                    # dwarfing).
ARROW_PER_N    = 8e-3               # rerun units (after CRAFT_SCALE) per N.
                                    # Full-throttle 500 N arrow = 4 body-m
                                    # ≈ 1.3 × LEVER_ARM — easy to read.


def render_position(p_phys: list[float]) -> list[float]:
    """Map a physical 3D position (m, world frame) to a rerun-space
    position with altitude above the Moon surface exaggerated."""
    r = math.sqrt(p_phys[0]**2 + p_phys[1]**2 + p_phys[2]**2)
    if r <= 0.0:
        return [0.0, 0.0, 0.0]
    alt = r - MOON_RADIUS
    r_render = (MOON_RADIUS + alt * ALT_EXAGGERATE) * SCALE
    s = r_render / r
    return [p_phys[0] * s, p_phys[1] * s, p_phys[2] * s]


def main() -> None:
    rr.init("manta_ex1", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Moon (physical scale).
    rr.log("world/moon",
           rr.Points3D(positions=[[0, 0, 0]],
                       radii=[MOON_RADIUS * SCALE],
                       colors=[[170, 170, 170]]),
           static=True)

    # Craft model + body-frame axes. Asset3D/TransformAxes3D are static;
    # the Transform3D logged per-tick provides world pose and scale.
    rr.log("world/craft/body",
           rr.Asset3D(path=str(pathlib.Path(__file__).parent / "Apollo_LM.gltf")),
           static=True)
    rr.log("world/craft",
           rr.archetypes.TransformAxes3D(1.0),
           static=True)
    # Thruster origin markers (body-local).
    for name, _direction, offset in THRUSTERS:
        rr.log(f"world/craft/thrusters/{name}/origin",
               rr.Points3D(positions=[list(offset)],
                           radii=[0.1],
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

        p_render = render_position(d["p"])
        rr.log("world/craft",
               rr.Transform3D(
                   translation=p_render,
                   rotation=rr.Quaternion(xyzw=[d["q"][1], d["q"][2],
                                                d["q"][3], d["q"][0]]),
                   scale=[CRAFT_SCALE, CRAFT_SCALE, CRAFT_SCALE]))

        origins:  list[list[float]] = []
        vectors:  list[list[float]] = []
        colors:   list[list[int]]   = []
        for name, direction, offset in THRUSTERS:
            thr = float(d.get(name, 0.0))
            length = thr * MAX_THRUST * ARROW_PER_N
            origins.append(list(offset))
            vectors.append([direction[0] * length,
                            direction[1] * length,
                            direction[2] * length])
            colors.append([255, 100, 100] if thr >= 0.0 else [80, 140, 255])
        rr.log("world/craft/thrusters/forces",
               rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))

    sub = session.declare_subscriber("manta/ex1/state", on_state)

    def shutdown(*_) -> None:
        sub.undeclare()
        session.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"viewer: listening on manta/ex1/state "
          f"(alt × {ALT_EXAGGERATE:g} exaggeration). Ctrl-C to exit.")
    signal.pause()


if __name__ == "__main__":
    main()
