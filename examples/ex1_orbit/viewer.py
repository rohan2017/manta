"""Rerun viewer for ex1 orbital craft.

Renders Earth, the craft's orbit trail, a body for the craft, and one
arrow per thruster whose length and direction track the live throttle.

The orbit and craft are at wildly different scales (orbit ≈ 6.4 Mm,
craft ≈ 1 m). We log Earth/orbit at world-scale via `SCALE` and use a
separate `rr.Transform3D` `scale` factor on the craft node so per-
thruster geometry (offsets, directions) can be specified in physical
meters in `config.py` and re-used directly here — Rerun applies the
scale automatically to the whole craft subtree.

Swapping in a real model: replace the `rr.Boxes3D` call on
`world/craft/body` with
    rr.log("world/craft/body", rr.Asset3D(path="my_craft.glb"), static=True)
(.glb / .gltf / .obj / .stl are all supported).
"""

import collections
import importlib.util
import json
import math
import pathlib
import signal
import sys

import rerun as rr
import zenoh

# ---- Pull thruster geometry directly from the sim config. ----
# config.py imports `manta_codegen`, so the viewer must be run with the
# same PYTHONPATH the codegen uses:
#     PYTHONPATH=python/manta_codegen/src python examples/ex1_orbit/viewer.py
_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("_ex1_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_cfg)
except ModuleNotFoundError as e:
    sys.exit(f"viewer.py: failed to load config ({e}). "
             "Run with PYTHONPATH=python/manta_codegen/src.")

THRUSTERS  = _cfg.THRUSTERS          # [(name, direction, offset), ...]
MAX_THRUST = _cfg.MAX_THRUST         # N
LEVER_ARM  = _cfg.LEVER_ARM          # m

EARTH_RADIUS = 6.371e6
SCALE        = 1.0 / 1.0e6           # 1 rerun unit per Mm — Earth ≈ 6.371 units
CRAFT_SCALE  = 5e5                   # rerun units per m of body geometry. The
                                     # craft is ~1 m physical; this magnifies
                                     # the body so it's visible against an
                                     # orbit at 6.4 Mm-radius scale.
ARROW_PER_N  = 0.04                  # rerun units (after CRAFT_SCALE) per Newton
                                     # of thrust. With MAX_THRUST=5 N and
                                     # LEVER_ARM=0.5 m, full-throttle arrows
                                     # are 4× the body's half-width — easy
                                     # to read without dwarfing the body.


def main() -> None:
    rr.init("manta_ex1", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ---- Earth + spin axis (world frame, at full sim scale via SCALE). ----
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

    # ---- Static craft body + thruster-origin markers. ----
    # All values below are in body-local meters; the craft node's
    # Transform3D will apply CRAFT_SCALE to render them visibly.
    _model_path = pathlib.Path(__file__).parent / "Apollo_LM.gltf"
    rr.log("world/craft/body",
           rr.Asset3D(path=str(_model_path)),
           static=True)

    for name, _direction, offset in THRUSTERS:
        rr.log(f"world/craft/thrusters/{name}/origin",
               rr.Points3D(positions=[list(offset)],
                           radii=[0.05],
                           colors=[[80, 80, 80]]),
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

        p_world = [d["p"][0] * SCALE, d["p"][1] * SCALE, d["p"][2] * SCALE]
        trail.append(p_world)

        # Craft Transform3D: world-frame translation + rotation, plus a
        # body-frame scale factor that makes the small physical craft
        # visible at the orbit's scale. Children logged under
        # `world/craft/...` are interpreted in body-local meters and
        # rendered scaled.
        rr.log("world/craft",
               rr.Transform3D(
                   translation=p_world,
                   rotation=rr.Quaternion(xyzw=[d["q"][1], d["q"][2],
                                                d["q"][3], d["q"][0]]),
                   scale=[CRAFT_SCALE, CRAFT_SCALE, CRAFT_SCALE],
                   axis_length=0.6))    # body-local meters

        # Per-thruster force arrows. Each arrow's tail sits at the
        # thruster's body-local offset; the vector points along the
        # thrust direction with length |throttle| · MAX_THRUST · scale.
        # Negative throttle reverses the arrow (the part is bipolar)
        # and switches color to blue.
        origins:  list[list[float]] = []
        vectors:  list[list[float]] = []
        colors:   list[list[int]]   = []
        for name, direction, offset in THRUSTERS:
            thr = float(d.get(name, 0.0))
            length = thr * MAX_THRUST * ARROW_PER_N  # signed — negative ⇒ reverse
            origins.append(list(offset))
            vectors.append([direction[0] * length,
                            direction[1] * length,
                            direction[2] * length])
            colors.append([255, 100, 100] if thr >= 0.0 else [80, 140, 255])
        rr.log("world/craft/thrusters/forces",
               rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))

        # Orbit trail (world frame).
        if len(trail) >= 2:
            rr.log("world/orbit_trail",
                   rr.LineStrips3D([list(trail)], colors=[[150, 230, 150]]))

        # Scalar plots.
        r = math.sqrt(d["p"][0]**2 + d["p"][1]**2 + d["p"][2]**2)
        rr.log("plots/altitude_km", rr.Scalar((r - EARTH_RADIUS) / 1000.0))
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
