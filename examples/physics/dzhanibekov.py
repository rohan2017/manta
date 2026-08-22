"""Dzhanibekov / tennis-racket effect — the unstable intermediate axis.

A torque-free rigid body spins stably about its largest or smallest
principal axis, but spin about the INTERMEDIATE axis is unstable: any
infinitesimal wobble grows exponentially until the body does a clean
180° flip, settles, then flips back — for ever, with no energy input
(Euler's equations conserve both energy and |L|; the flip is the polhode
crossing the separatrix, not a perturbation).

The body here is a T of four equal point masses — a crossbar on a
two-mass stem, no gravity, nothing but inertia. For a planar body the
perpendicular-axis theorem pins I_z = I_x + I_y (z, out of plane, is
always the major axis), so the intermediate axis is the larger in-plane
one — the STEM axis for this geometry. The growth rate collapses to

    λ = ω · √((I_int − I_min) / I_z)

It spins about the stem (the T's handle) with a 0.3 % wobble; in the
body frame ω_y holds steady then snaps sign at each flip — watch the
crossbar's orange/cyan tips swap sides. The stem tip rides near the
spin axis, so its trail hangs still between flips and sweeps an arc
during each one.

Run::

    .venv/bin/python -m examples.physics.dzhanibekov
    .venv/bin/python -m examples.physics.dzhanibekov --no-viz
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass

from .._control import Pacer
from .._geom import rotmat
from .._viz import Viz

BAR = 0.7                            # crossbar half-length (m)
MASSES = {                           # name → body-frame position, 1 kg each
    "bar_l":  (-BAR, 0.0, 0.0),
    "bar_r":  (+BAR, 0.0, 0.0),
    "stem_a": (0.0, -0.4, 0.0),
    "stem_b": (0.0, -0.8, 0.0),
}
TIP = np.array(MASSES["stem_b"])     # the end of the T (trail point)
SPIN = 6.0                           # rad/s about the stem (intermediate) axis
WOBBLE = 0.02                        # rad/s seed perturbation (0.3 %)


def principal_moments():
    """Principal moments about the COM (x = crossbar, y = stem, z = out
    of plane) — the T's axes are already principal by symmetry."""
    pts = np.array(list(MASSES.values()))
    rel = pts - pts.mean(axis=0)
    ix = float((rel[:, 1] ** 2 + rel[:, 2] ** 2).sum())
    iy = float((rel[:, 0] ** 2 + rel[:, 2] ** 2).sum())
    return ix, iy, ix + iy           # planar: I_z = I_x + I_y


def build_world():
    tee = Craft("tee")
    for name, pos in MASSES.items():
        tee.add(Mass(name, mass=1.0, moi=(1e-3, 1e-3, 1e-3), mount_offset=pos))
    w = World().add_field(GravityField.none())                      # no fields: free-floating, torque-free
    # Spin about the stem (body y) + a tiny wobble about the crossbar.
    w.add_craft(tee, position=(0.0, 0.0, 1.2),
                angular_velocity=(WOBBLE, SPIN, 0.0))
    return w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--duration", type=float, default=30.0)
    args = p.parse_args()

    dt = 1e-3
    n = int(args.duration / dt)
    ix, iy, iz = principal_moments()
    lam = SPIN * np.sqrt((iy - ix) / iz)
    print(f"principal moments I = ({ix:.3f}, {iy:.3f}, {iz:.3f}) kg·m²  "
          f"(intermediate: y, the stem)")
    print(f"instability growth λ = {lam:.2f} /s → flips every few seconds\n")

    viz = None if args.no_viz else Viz("manta/dzhanibekov")
    if viz is not None:
        viz.box("world/tee/bar", (BAR, 0.025, 0.025), color=(150, 150, 158))
        viz.box("world/tee/stem", (0.025, 0.4, 0.025),
                center=(0.0, -0.4, 0.0), color=(150, 150, 158))

    sim = TargetNumpy(Sim(build_world()))
    pacer = Pacer() if viz is not None else None

    flips, last_sign = 0, 1.0
    print(f"{'t (s)':>6} {'ω_body (rad/s)':>24} {'flips':>6}")
    for i in range(n):
        t = i * dt
        if pacer is not None:
            pacer.pace(t)
        sim.step(dt)
        st = sim.state["tee"]
        om = np.asarray(st["angular_velocity"]).ravel()

        # A flip = the body reverses its spin about its OWN y-axis (L is
        # fixed in the world; the body turned over underneath it).
        s = np.sign(om[1])
        if s != 0 and s != last_sign and abs(om[1]) > 0.5 * SPIN:
            flips += 1
            last_sign = s
            print(f"{t:>6.2f}  → flip #{flips}  (ω_y now {om[1]:+.2f})")

        if viz is not None and i % 33 == 0:        # ~30 Hz
            q = np.asarray(st["orientation"]).ravel()
            pos = np.asarray(st["position"]).ravel()
            viz.t(t)
            viz.pose("world/tee", pos, q)
            # Crossbar tips coloured to make the 180° swap obvious.
            viz.point("world/tee/m_l", MASSES["bar_l"],
                      color=(255, 140, 50), radius=0.07)
            viz.point("world/tee/m_r", MASSES["bar_r"],
                      color=(70, 200, 255), radius=0.07)
            for name in ("stem_a", "stem_b"):
                viz.point(f"world/tee/m_{name}", MASSES[name],
                          color=(180, 180, 190), radius=0.07)
            viz.trail("world/tip_trail", pos + rotmat(q) @ TIP,
                      color=(235, 200, 80), max_len=3000, min_dist=0.005)

        if (i + 1) % 2000 == 0:
            print(f"{t:>6.2f} {np.round(om, 2) + 0.0!s:>24} {flips:>6}")

    om = np.asarray(sim.state["tee"]["angular_velocity"]).ravel()
    print(f"\n{flips} flips in {args.duration:.0f} s; final ω_body "
          f"{np.round(om, 2) + 0.0}  (|ω| {np.linalg.norm(om):.3f}, "
          f"started {np.linalg.norm([WOBBLE, SPIN, 0.0]):.3f})")


if __name__ == "__main__":
    main()
