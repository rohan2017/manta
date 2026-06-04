"""Spinning top — gyroscopic stability on a real contact tip.

A nearly massless pole carries a heavy disk at its TOP — an inverted
pendulum, normally unstable. The pole tip stands on a ground plane
through a `Collider` (a frictionless point contact, gravity on). Two
runs, same rig:

  1. no spin → it falls right over within a second (the frictionless
     tip skates out sideways — a rod falling on ice; a crown `Collider`
     lets it lie down instead of poking through the floor);
  2. spinning fast → the contact normal's torque becomes **precession**:
     released leaning 8°, the lean azimuth sweeps a cone while the tilt
     holds — and the contact damping slowly *rights* the top, exactly
     like a real top "going to sleep".

The spin lives in the rigid body itself (like a real toy top — the
whole thing spins), so the gyroscopics come from the exact rigid-body
integrator. The disk is rendered as a two-colour split disc so you can
see it spin (it strobes at the viewer frame rate, like a real top on
camera).

Run::

    .venv/bin/python -m examples.physics.spinning_top
    .venv/bin/python -m examples.physics.spinning_top --no-viz
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import CollisionField, GravityField
from manta.parts import Collider, Mass

from .._viz import Viz

G = 9.81
H = 0.30          # pole height (disk sits on top)
DISK_R = 0.14     # disk radius (viz; I_z = m·r²/2 below)
SPIN = 150.0      # spin rate, rad/s
TILT0 = np.radians(8.0)              # released leaning 8°
R_COM = 0.293     # craft COM height up the pole (pole + disk)
M_TOT = 0.42


def _build_world(spin: float):
    top = Craft("top")
    # The pole is nearly massless — all the inertia is the disk on top,
    # which is what makes the standing top an inverted pendulum.
    top.add(Mass("pole", mass=0.02, moi=(2e-4, 2e-4, 1e-5),
                 transform=(0.0, 0.0, H / 2)))
    top.add(Mass("disk", mass=0.4, moi=(0.002, 0.002, 0.004),
                 transform=(0.0, 0.0, H)))
    # Contact: the tip it stands on + a crown point so a toppled top
    # lies down instead of poking through the floor.
    top.add(Collider("tip", stiffness=2000.0, damping=30.0))
    top.add(Collider("crown", stiffness=2000.0, damping=30.0,
                     transform=(0.0, 0.0, H)))

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -G)))
         .add_field(CollisionField().add_half_space()))

    # Launch: leaning TILT0, spinning about its own (tilted) axis, and
    # already carrying the slow-precession rate about vertical (without
    # it the release just nutates in a deep dive). The initial-velocity
    # term cancels the ω×r_com seeding so the top spins in place.
    omega_p = (M_TOT * G * R_COM / (0.004 * spin)) if spin else 0.0
    axis = np.array([0.0, -np.sin(TILT0), np.cos(TILT0)])
    om0 = spin * axis + np.array([0.02, 0.0, omega_p])
    v0 = -np.cross(om0, [0.0, 0.0, R_COM])
    w.add_craft(top, position=(0.0, 0.0, 0.003),
                orientation=(np.cos(TILT0 / 2), np.sin(TILT0 / 2), 0.0, 0.0),
                velocity=tuple(v0),
                angular_velocity=tuple(om0))
    return w


def _tilt_azimuth(q_wxyz):
    """(tilt from vertical, lean azimuth) of the body z-axis, rad."""
    w, x, y, z = q_wxyz
    zax = np.array([2 * (x*z + w*y), 2 * (y*z - w*x),
                    1 - 2 * (x*x + y*y)])
    return (float(np.arccos(np.clip(zax[2], -1.0, 1.0))),
            float(np.arctan2(zax[1], zax[0])))


def _run(label: str, spin: float, viz: Viz | None, dt: float, n: int):
    sim = TargetNumpy(Sim(_build_world(spin)))

    print(f"\n=== {label}  spin = {spin} rad/s ===")
    print(f"{'t (s)':>5} {'tilt°':>8} {'lean az°':>9} {'|ω| (rad/s)':>12}")
    marks = {round(s / dt) for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                                     5.0, 6.0, 7.0, 8.0)}
    for i in range(n):
        t = i * dt
        sim.step(dt)

        if viz is not None and i % 40 == 0:
            st = sim.state["top"]
            p = np.asarray(st["position"]).ravel()
            q = np.asarray(st["orientation"]).ravel()
            viz.t(t)
            if i == 0:
                # Disc offset pose: on the sim timeline, not at init.
                viz.pose("world/top/disk", (0, 0, H + 0.006))
            viz.pose("world/top", p, q)

        if i in marks:
            st = sim.state["top"]
            q = np.asarray(st["orientation"]).ravel()
            om = np.asarray(st["angular_velocity"]).ravel()
            tilt, az = _tilt_azimuth(q)
            print(f"{t:>5.2f} {np.degrees(tilt):>8.1f} "
                  f"{np.degrees(az):>9.1f} {np.linalg.norm(om):>12.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--duration", type=float, default=8.0)
    args = p.parse_args()

    dt = 2.5e-4
    n = int(args.duration / dt)
    viz = None if args.no_viz else Viz("manta/spinning_top")
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=1.0, color=(70, 110, 70, 200))
        viz.box("world/top/pole", (0.008, 0.008, H / 2),
                center=(0, 0, H / 2), color=(200, 200, 210))
        viz.split_disc("world/top/disk", DISK_R)

    print("Spinning top standing on its tip — same rig, two runs.")
    print("No spin → inverted pendulum, falls right over;")
    print("spinning → precesses on its lean cone and slowly rights")
    print("itself as the contact damping bleeds the wobble.")
    _run("SPINNING", SPIN, viz, dt, n)        # visualized
    _run("STATIONARY", 0.0, None, dt, n)      # terminal only

    print(f"\n(precession estimate Ω ≈ m·g·h/L = "
          f"{M_TOT * G * R_COM / (0.004 * SPIN):.1f} rad/s)")


if __name__ == "__main__":
    main()
