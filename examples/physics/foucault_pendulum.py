"""Foucault pendulum — Earth's rotation precesses a swinging bob.

A bob hangs from an apex on a stiff `Tether` (a wire/ball pivot, as real
Foucault pendulums are suspended). The rig sits at the **north pole** on a
four-`Collider` base standing on the `Earth`'s own surface — the planet
registers its sea-level sphere as a `CollisionField` obstacle, so the rig
could stand anywhere on the globe with no per-site ground plane. Released
at rest in the *inertial* frame, the bob swings in an inertially-fixed
plane; viewed in the rotating planet frame that plane precesses at −Ω —
Foucault's effect. Earth's true Ω (7.29e-5 rad/s, a 24 h precession) is
too slow to watch, so we exaggerate it; the *measured* precession rate
still equals Ω.

The viewer shows the scene in the **co-rotating planet frame**, where the
base is fixed and the swing plane visibly rotates — the rosette trail is
the precession. The terminal fits the precession rate and compares it to
−Ω (ratio 1.0 = exact Foucault).

Suspension: a `Tether` (a wire/ball pivot, like the real instrument).
A stacked two-axis `RevoluteJoint` gimbal works too — the joint-space
solve carries the inter-body Coriolis coupling that drives the
precession (see tests/test_foucault_joints.py) — but the wire is the
physically faithful suspension for this rig.

Run::

    .venv/bin/python -m examples.physics.foucault_pendulum
    .venv/bin/python -m examples.physics.foucault_pendulum --no-viz
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.couplings import Tether
from manta.parts import Collider, Mass, TetherEndpoint
from manta.planets import Earth

from .._viz import Viz

OMEGA = 0.5            # rad/s (real Earth: 7.29e-5) — exaggerated to watch
R = Earth.R_EQ        # north pole, z = R
HALF = 1.0            # pyramid base half-width [m]
APEX = 1.5            # pivot height above base [m]
LEN = 0.5             # pendulum length [m]
DEFL = 0.05           # initial deflection [m]
DT = 2.0e-4


def _build_world():
    earth = Earth(rotation_rate=OMEGA)
    base = Craft("base")
    base.add(Mass("hub", mass=150.0, moi=(60.0, 60.0, 90.0)))
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        base.add(Collider(f"foot{i}", stiffness=1.0e5, damping=3200.0,
                          mount_offset=(sx * HALF, sy * HALF, 0.0)))
    base.add(TetherEndpoint("hook", mount_offset=(0.0, 0.0, APEX)))

    bob = Craft("bob")
    bob.add(Mass("bob", mass=1.0, moi=(1e-6, 1e-6, 1e-6)))
    bob.add(TetherEndpoint("hook"))

    # No per-site ground plane: Earth registers its sea-level sphere as
    # the collision surface, so the base's feet land on the planet itself.
    w = World()
    w.add_planet(earth)
    # A local Scene at the north pole (planet at the world origin). Placing
    # the base "at rest" in it gives the co-spin body rate (0, 0, OMEGA) and
    # zero orbital velocity (Ω×r = 0 on the axis). The bob, by contrast, is
    # left inertial (velocity 0) — that's the Foucault effect.
    scene = earth.scene_at((0.0, 0.0, R))
    w.add_craft(base, **scene.at_rest())
    w.add_craft(bob, position=(DEFL, 0.0, R + APEX - LEN),
                velocity=(0.0, 0.0, 0.0))
    w.add_coupling(Tether(base, "hook", bob, "hook",
                          stiffness=8.0e3, damping=6.0, rest_length=LEN))
    return w, earth


def _swing_azimuth(window: np.ndarray) -> float:
    cov = np.cov(window.T)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    return float(np.arctan2(v[1], v[0]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true", help="run headless")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds (default: one precession period 2π/Ω)")
    args = p.parse_args()

    w, earth = _build_world()
    sim = TargetNumpy(Sim(w))

    T_prec = 2.0 * np.pi / OMEGA
    duration = T_prec if args.duration is None else args.duration
    n_steps = int(duration / DT)
    sample_every = 40

    viz = None if args.no_viz else Viz("manta/foucault")
    if viz is not None:
        # Apex pivot, fixed in the (rendered) planet frame.
        viz.point("world/apex", (0, 0, APEX), color=(200, 200, 200),
                  radius=0.03)

    print(f"Foucault pendulum at the north pole (exaggerated Ω = {OMEGA} rad/s)")
    print(f"swing period {2*np.pi*np.sqrt(LEN/9.81):.2f} s; "
          f"precession period 2π/Ω = {T_prec:.2f} s\n")
    print(f"{'t (s)':>7} {'swing azimuth (deg)':>20} {'base spin (deg)':>16}")

    bob_planet, times = [], []
    t = 0.0
    for i in range(n_steps):
        sim.step(DT, t=t)
        t += DT
        if i % sample_every == 0:
            p_planet, _ = earth.world_to_planet(
                sim.state["bob"]["position"], sim.state["bob"]["velocity"], t)
            bob_planet.append((p_planet[0], p_planet[1]))
            times.append(t)
            if viz is not None:
                # Render in the co-rotating planet frame, local to the pole.
                bob_local = (p_planet[0], p_planet[1], p_planet[2] - R)
                viz.t(t)
                viz.point("world/bob", bob_local, color=(240, 180, 60),
                          radius=0.04)
                viz.line("world/tether", [(0, 0, APEX), bob_local],
                         color=(180, 180, 180), radius=0.004)
                viz.trail("world/bob/trail", bob_local, color=(90, 170, 255))

    bob_planet = np.asarray(bob_planet) - np.asarray(bob_planet).mean(axis=0)
    times = np.asarray(times)

    win = 120
    az, tc = [], []
    for s in range(0, len(bob_planet) - win, 20):
        az.append(_swing_azimuth(bob_planet[s:s + win]))
        tc.append(times[s + win // 2])
    if len(az) < 2:
        print("\n(run longer for a precession-rate fit)")
        return
    az = np.unwrap(np.asarray(az) * 2.0) / 2.0
    tc = np.asarray(tc)

    for k in range(0, len(tc), max(1, len(tc) // 10)):
        print(f"{tc[k]:7.2f} {np.degrees(az[k]-az[0]):20.1f} "
              f"{np.degrees(OMEGA*tc[k]):16.1f}")

    rate = float(np.polyfit(tc, az, 1)[0])
    print()
    print(f"measured precession rate : {rate:+.4f} rad/s")
    print(f"theoretical (−Ω at pole) : {-OMEGA:+.4f} rad/s")
    print(f"ratio                    : {rate/-OMEGA:.3f}  (1.0 = exact)")


if __name__ == "__main__":
    main()
