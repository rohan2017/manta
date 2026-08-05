"""Spinning top — a fast rotor `RevoluteJoint` precessing on a real contact tip.

A nearly massless pole carries a heavy disk on a passive `RevoluteJoint` at its
TOP — an inverted pendulum, normally unstable. The pole tip stands on a
ground plane through a `Collider` (a frictionless point contact,
gravity on). Two runs, same rig:

  1. rotor stationary → it falls right over within a second (the
     frictionless tip skates out sideways — a rod falling on ice; a
     crown `Collider` lets it lie down instead of poking the floor);
  2. rotor spinning fast → the contact normal's torque becomes
     **precession**: released leaning 8°, the lean azimuth sweeps a
     cone while the tilt holds — and the contact damping slowly
     *rights* the top, exactly like a real top "going to sleep".

The rotor DOF exercises the coupled body/rotor solve + the momentum-
form angular integrator (a fast rotor on a precessing mount is its
acid test). The disk is rendered as a two-colour split disc so you can
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
from manta.parts import Collider, RevoluteJoint, Mass

from .._viz import Viz

G = 9.81
H = 0.30          # pole height (disk sits on top)
DISK_R = 0.14     # disk radius (viz; I_z = m·r²/2 below)
DISK_H = 0.04     # disk thickness (viz)
SPIN = 150.0      # spin rate, rad/s
TILT0 = np.radians(8.0)              # released leaning 8°
R_COM = 0.293     # craft COM height up the pole (pole + disk)
M_TOT = 0.42


def _build_world(spin: float):
    top = Craft("top")
    # The pole is nearly massless — all the inertia is the disk on top,
    # which is what makes the standing top an inverted pendulum.
    top.add(Mass("pole", mass=0.02, moi=(2e-4, 2e-4, 1e-5),
                 mount_offset=(0.0, 0.0, H / 2)))
    # A touch of bearing friction on the rotor: it locks the (nearly
    # massless) pole into co-rotation with the disk within ~30 ms — a
    # real toy top is rigid — after which the spin lives in the body
    # and the relative joint rate sits near zero.
    rotor = RevoluteJoint("rotor", mode="passive", axis=(0.0, 0.0, 1.0),
                  damping=3.0e-4, mount_offset=(0.0, 0.0, H))
    rotor.add(Mass("disk", mass=0.4, moi=(0.002, 0.002, 0.004)))
    top.add(rotor)
    # Contact: the tip it stands on (tangential friction grips it
    # against skating, and damps the precession sweep — the energy
    # drain that makes the top sink lower and lower) + a crown point
    # so a toppled top lies down instead of poking through the floor.
    top.add(Collider("tip", stiffness=2000.0, damping=30.0, friction=3.0))
    top.add(Collider("crown", stiffness=2000.0, damping=30.0, friction=3.0,
                     mount_offset=(0.0, 0.0, H)))

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -G)))
         .add_field(CollisionField().add_half_space()))

    # Launch: leaning TILT0 with the spin in the rotor DOF, the BODY
    # carrying only the slow-precession rate about vertical (without it
    # the release just nutates in a deep dive). angular_velocity is
    # CraftFrame (body rates); rotate the world-z precession into the
    # tilted body.
    omega_p = (M_TOT * G * R_COM / (0.004 * spin)) if spin else 0.0
    om0 = omega_p * np.array([0.0, np.sin(TILT0), np.cos(TILT0)]) \
        + np.array([0.02, 0.0, 0.0])
    w.add_craft(top, position=(0.0, 0.0, 0.003),
                orientation=(np.cos(TILT0 / 2), np.sin(TILT0 / 2), 0.0, 0.0),
                angular_velocity=tuple(om0),
                **{"rotor.rate": spin})
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
    print(f"{'t (s)':>5} {'tilt°':>8} {'lean az°':>9} {'spin':>8}")
    marks = {round(s / dt) for s in
             [0.5, 1.0, 1.5] + list(range(2, 60, 2))}
    for i in range(n):
        t = i * dt
        sim.step(dt)

        if viz is not None and i % 40 == 0:
            st = sim.state["top"]
            p = np.asarray(st["position"]).ravel()
            q = np.asarray(st["orientation"]).ravel()
            ang = float(st["rotor.angle"])
            viz.t(t)
            viz.pose("world/top", p, q)
            viz.pose("world/top/rotor", (0, 0, H + 0.004 + DISK_H / 2),
                     (np.cos(ang / 2), 0, 0, np.sin(ang / 2)))

        if i in marks:
            st = sim.state["top"]
            q = np.asarray(st["orientation"]).ravel()
            om = np.asarray(st["angular_velocity"]).ravel()   # body rates
            tilt, az = _tilt_azimuth(q)
            spin_now = float(st["rotor.rate"]) + om[2]   # disk total spin
            print(f"{t:>5.2f} {np.degrees(tilt):>8.1f} "
                  f"{np.degrees(az):>9.1f} {spin_now:>8.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--duration", type=float, default=45.0)
    args = p.parse_args()

    dt = 2.5e-4
    n = int(args.duration / dt)
    viz = None if args.no_viz else Viz("manta/spinning_top")
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=1.0, color=(70, 110, 70, 200))
        viz.box("world/top/pole", (0.008, 0.008, H / 2),
                center=(0, 0, H / 2), color=(200, 200, 210))
        viz.split_cylinder("world/top/rotor/disk", DISK_R, DISK_H)

    print("Spinning top standing on its tip — same rig, two runs.")
    print("No spin → inverted pendulum, falls right over;")
    print("spinning → precesses on its lean cone, then sinks lower and")
    print("lower as the bearing friction bleeds the spin — a dying top.")
    _run("SPINNING", SPIN, viz, dt, n)               # visualized
    _run("STATIONARY", 0.0, None, dt, int(6.0 / dt))  # topples in ~2 s

    print(f"\n(precession estimate Ω ≈ m·g·h/L = "
          f"{M_TOT * G * R_COM / (0.004 * SPIN):.1f} rad/s)")


if __name__ == "__main__":
    main()
