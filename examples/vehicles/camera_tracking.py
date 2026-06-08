"""Camera tracking — a survey craft sees other vehicles, by field.

The headline demo for the OpticalField + field-source parts. A survey
craft coasts past a small field of "rocks". Each rock carries field
sources, and the survey craft carries the matching sensors:

  * `OpticalSource` on each rock  → `Camera` on the survey craft boxes
    them: it projects each rock's semantic ellipsoid to an image-frame
    bounding box (the standard 2-D detection a perception stack eats).
  * `MagneticSource` on the metallic rock → the survey `Magnetometer`
    reads the dipole swell as it passes closest.
  * `GravitySource` on a massive distant body → its inverse-square pull
    curves the survey craft's coast (no thrust; it just falls along).

Every source rides its craft's pose and is registered onto the world
fields automatically — the survey craft just declares sensors and reads
them. The terminal prints the boxes + field readings each second; with
the viewer you get the 3-D scene (ellipsoids + camera frustum) and a 2-D
image panel with the live bounding boxes.

Run::

    .venv/bin/python -m examples.vehicles.camera_tracking
    .venv/bin/python -m examples.vehicles.camera_tracking --no-viz
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField, MagField, OpticalField
from manta.parts import (
    Camera, GravitySource, MagneticSource, Mass, Magnetometer, OpticalSource,
)

from .._viz import Viz

W_IMG, H_IMG, HFOV = 640, 480, 70.0

# (name, position, semi-axes, label, colour, metallic?)
ROCKS = [
    ("rock_a", (4.0, 0.0, 11.0), (1.5, 1.5, 1.5), 1, (120, 200, 255), False),
    ("rock_b", (0.0, 0.0, 7.0),  (2.0, 1.0, 1.0), 2, (255, 170, 90),  True),
]


def build_world():
    survey = Craft("survey")
    survey.add(Mass("body", mass=50.0, moi=(2.0, 2.0, 2.0)))
    survey.add(Camera("cam", width=W_IMG, height=H_IMG, hfov_deg=HFOV))
    survey.add(Magnetometer("mag"))

    w = (World()
         .add_field(GravityField(g=(0, 0, 0)))   # only the source contributes
         .add_field(MagField())
         .add_field(OpticalField()))
    # Survey coasts in +x, camera (+z) facing the rocks; starts off to the side.
    w.add_craft(survey, position=(-7.0, 0.0, 0.0), velocity=(1.6, 0.0, 0.0))

    crafts = {"survey": survey}
    for name, pos, semi, label, _col, metallic in ROCKS:
        rock = Craft(name)
        rock.add(Mass("body", mass=2.0e3))
        rock.add(OpticalSource("hull", semi_axes=semi, label=label))
        if metallic:
            # A strongly magnetized ore body — readable as the craft passes.
            rock.add(MagneticSource("ore", moment=(0.0, 0.0, 900.0)))
        w.add_craft(rock, position=pos)
        crafts[name] = rock

    # A distant massive world. Its inverse-square pull tugs the whole
    # formation (≈ common-mode at this range — the camera geometry is set
    # by the survey's own lateral coast; gravity adds a slow shared drift).
    planet = Craft("planet")
    planet.add(Mass("core", mass=1.0))
    planet.add(GravitySource("grav", GM=400.0))
    w.add_craft(planet, position=(0.0, -50.0, 0.0))
    crafts["planet"] = planet
    return w, crafts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--viz-addr", default=None)
    p.add_argument("--duration", type=float, default=9.0)
    args = p.parse_args()

    w, crafts = build_world()
    sim = TargetNumpy(Sim(w))
    dt = 0.02
    n = int(args.duration / dt)
    box_keys = [f"cam.{name}_hull" for name, *_ in ROCKS]

    viz = None if args.no_viz else Viz("manta/camera_tracking",
                                       addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=-3.0, size=60.0, color=(55, 60, 70, 120))
        # The 2-D image panel: a static frame rectangle the boxes live in.
        viz.rr.log("image/frame", viz.rr.Boxes2D(
            mins=[[0, 0]], sizes=[[W_IMG, H_IMG]], colors=[(70, 70, 80)]),
            static=True)

    print(f"\n{'t':>5}  {'survey x,z':>14}  "
          + "  ".join(f"{name:>22}" for name, *_ in ROCKS)
          + f"  {'|B| (µT)':>9}")
    for i in range(n):
        sim.step(dt)
        out = sim.outputs()["survey"]
        t = (i + 1) * dt
        boxes = {}
        for (name, _pos, _semi, label, _col, _m), key in zip(ROCKS, box_keys):
            g = lambda s: float(np.asarray(out[f"{key}_{s}"]).ravel()[0])
            vis = g("vis")
            boxes[name] = (g("xmin"), g("ymin"), g("xmax"), g("ymax"), vis,
                           label)
        B = np.linalg.norm(np.asarray(out["mag.B"]).ravel()) * 1e6  # µT

        if (i + 1) % int(1.0 / dt) == 0:
            sp = np.asarray(sim.state["survey"]["position"]).ravel()
            cells = []
            for name, *_ in ROCKS:
                x0, y0, x1, y1, vis, _lab = boxes[name]
                cells.append(f"[{x0:4.0f},{y0:4.0f},{x1:4.0f},{y1:4.0f}]"
                             if vis > 0.5 else f"{'--- out of view ---':>22}")
            print(f"{t:>5.1f}  ({sp[0]:+5.1f},{sp[2]:+5.1f})  "
                  + "  ".join(cells) + f"  {B:>9.3f}")

        if viz is not None:
            sp = np.asarray(sim.state["survey"]["position"]).ravel()
            sq = np.asarray(sim.state["survey"]["orientation"]).ravel()
            viz.t(t)
            viz.pose("world/survey", sp, sq)
            viz.box("world/survey/body", (0.4, 0.4, 0.4), color=(230, 230, 240),
                    static=False)
            viz.arrow("world/survey/los", (0, 0, 0), (0, 0, 3.0),
                      color=(120, 220, 140), radius=0.04)
            viz.trail("world/survey/trail", sp, max_len=2000, min_dist=0.05)
            mins2d, sizes2d, cols2d = [], [], []
            for (name, _p, _s, _lab, col, _m) in ROCKS:
                rp = np.asarray(sim.state[name]["position"]).ravel()
                rs = dict((n, s) for n, _pp, s, *_ in ROCKS)[name]
                viz.rr.log(f"world/{name}", viz.rr.Ellipsoids3D(
                    centers=[rp], half_sizes=[rs], colors=[col],
                    fill_mode="solid"))
                x0, y0, x1, y1, vis, _lab = boxes[name]
                if vis > 0.5:
                    mins2d.append([x0, y0]); sizes2d.append([x1 - x0, y1 - y0])
                    cols2d.append(col)
            viz.rr.log("image/boxes", viz.rr.Boxes2D(
                mins=mins2d or [[0, 0]], sizes=sizes2d or [[0, 0]],
                colors=cols2d or [(0, 0, 0)]))

    print("\nThe boxes slide across the 640×480 image as the survey craft "
          "passes; rock_b's box is wider (a 2.5×1×1 ellipsoid) and its "
          "magnetic ore lifts |B| as the craft nears.")


if __name__ == "__main__":
    main()
