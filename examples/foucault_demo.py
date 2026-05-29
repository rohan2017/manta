"""Foucault pendulum — Earth's rotation precesses a swinging bob.

A Foucault pendulum swings in a plane that stays fixed in inertial space
while the Earth turns beneath it. To an observer on the ground the swing
plane appears to rotate; at the **north pole** it precesses at exactly the
Earth's rotation rate Ω (period = one sidereal day). Away from the pole the
rate is Ω·sin(latitude).

Setup here:
  * A square-pyramid **base of four colliders** rests on the ground at the
    north pole and co-rotates with Earth — it's seeded spinning at Ω, and a
    symmetric base conserves that spin on its own (the collision model is
    frictionless, and none is needed: nothing torques it about the vertical).
  * A bob hangs from the apex on a stiff **tether** — a wire / ball pivot,
    which is exactly how real Foucault pendulums are suspended. The tether's
    damping acts only along the wire (radial), so it tames the stiff
    stretch mode for numerical stability without damping the swing.
  * The bob is released from a small deflection **at rest in the inertial
    frame**, so it swings in an inertially-fixed plane. Viewed in the
    rotating planet frame (via `earth.world_to_planet`), that plane precesses
    at −Ω — the Foucault signature.

Earth's true Ω is 7.29e-5 rad/s (a 24 h precession) — far too slow to watch —
so we exaggerate the rotation rate (as `coriolis_drop_demo` does) to see a
full precession in seconds. The measured precession rate still equals Ω.

Why a tether and not a gimbal: manta's articulated `Joint` is a deliberately
cheap "good enough" model — excellent for reaction wheels, gimbaled
thrusters, and driven or passive boom arms — but its fixed-base approximation
drops the inter-body Coriolis coupling that drives Foucault precession (adding
it correctly needs a coupled multibody solve). A tether is a true two-body
constraint, so it captures that coupling exactly.

Run::

    .venv/bin/python -m examples.foucault_demo
"""

from __future__ import annotations

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.couplings import Tether
from manta.fields import CollisionField
from manta.parts import Collider, Mass, TetherEndpoint
from manta.planets import Earth


# Exaggerated Earth rotation so a full precession takes seconds, not a day.
OMEGA = 0.5            # rad/s  (real Earth: 7.29e-5)
R     = Earth.R_EQ     # place the rig at the north pole, z = R
HALF  = 1.0            # pyramid base half-width [m]
APEX  = 1.5            # pivot height above the base [m]
LEN   = 0.5            # pendulum length [m]
DEFL  = 0.05           # initial deflection [m] (small → clean planar swing)
DT    = 2.0e-4


def _build_world() -> World:
    earth = Earth(rotation_rate=OMEGA)

    # Heavy base: a central mass + four ground colliders at the pyramid
    # corners, with near-critical normal damping so it settles and stays put.
    base = Craft("base")
    base.add(Mass("hub", mass=150.0, moi=(60.0, 60.0, 90.0)))
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        base.add(Collider(f"foot{i}", stiffness=1.0e5, damping=3200.0,
                          transform=(sx * HALF, sy * HALF, 0.0)))
    base.add(TetherEndpoint("hook", transform=(0.0, 0.0, APEX)))   # apex pivot

    bob = Craft("bob")
    bob.add(Mass("bob", mass=1.0, moi=(1e-6, 1e-6, 1e-6)))
    bob.add(TetherEndpoint("hook"))

    w = World()
    w.add_planet(earth)
    w.add_field(CollisionField().add_half_space(origin=(0.0, 0.0, R),
                                                normal=(0.0, 0.0, 1.0)))
    # Base sits at the pole, co-rotating with Earth (spinning at Ω about the
    # local vertical = the planet's axis).
    w.add_craft(base,
                position=earth.position(0.0, 0.0, R),
                velocity=earth.at_rest(),
                angular_velocity=(0.0, 0.0, OMEGA))
    # Bob: deflected, released at rest in the INERTIAL frame → swings in an
    # inertially-fixed plane.
    w.add_craft(bob,
                position=(DEFL, 0.0, R + APEX - LEN),
                velocity=(0.0, 0.0, 0.0))
    # Stiff tether ≈ a rigid wire; damping is radial (along the wire) only.
    w.add_coupling(Tether(base, "hook", bob, "hook",
                          stiffness=8.0e3, damping=6.0, rest_length=LEN))
    return w, earth


def _swing_azimuth(window: np.ndarray) -> float:
    """Orientation (mod π) of the swing line = principal axis of the bob's
    horizontal-position scatter over a short window."""
    cov = np.cov(window.T)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    return float(np.arctan2(v[1], v[0]))


def main() -> None:
    w, earth = _build_world()
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()

    T_prec = 2.0 * np.pi / OMEGA          # one precession period (at the pole)
    n_steps = int(T_prec / DT)
    sample_every = 40

    print(f"Foucault pendulum at the north pole (exaggerated Ω = {OMEGA} rad/s)")
    print(f"pendulum length {LEN} m → swing period "
          f"{2*np.pi*np.sqrt(LEN/9.81):.2f} s; "
          f"precession period 2π/Ω = {T_prec:.2f} s")
    print()
    print(f"{'t (s)':>7} {'swing azimuth (deg)':>20} {'base spin (deg)':>16}")

    bob_planet, times = [], []
    t = 0.0
    next_print = 0.0
    for i in range(n_steps):
        state = sim.step(state, dt=DT, t=t)
        t += DT
        if i % sample_every == 0:
            p_planet, _ = earth.world_to_planet(
                state["bob"]["position"], state["bob"]["velocity"], t)
            bob_planet.append((p_planet[0], p_planet[1]))
            times.append(t)

    bob_planet = np.asarray(bob_planet)
    bob_planet -= bob_planet.mean(axis=0)
    times = np.asarray(times)

    # Sliding-window swing-plane azimuth (a line, so unwrap mod π).
    win = 120
    az, tc = [], []
    for s in range(0, len(bob_planet) - win, 20):
        az.append(_swing_azimuth(bob_planet[s:s + win]))
        tc.append(times[s + win // 2])
    az = np.unwrap(np.asarray(az) * 2.0) / 2.0
    tc = np.asarray(tc)

    az0 = az[0]
    for k in range(0, len(tc), max(1, len(tc) // 10)):
        print(f"{tc[k]:7.2f} {np.degrees(az[k]-az0):20.1f} "
              f"{np.degrees(OMEGA*tc[k]):16.1f}")

    rate = float(np.polyfit(tc, az, 1)[0])
    print()
    print(f"measured precession rate : {rate:+.4f} rad/s")
    print(f"theoretical (−Ω at pole) : {-OMEGA:+.4f} rad/s")
    print(f"ratio                    : {rate/-OMEGA:.3f}  "
          f"(1.0 = exact Foucault)")
    print()
    print("The swing plane stays fixed in inertial space while the base turns "
          "with Earth;\nin the rotating frame it precesses at the Earth's "
          "rotation rate — Foucault's effect.")


if __name__ == "__main__":
    main()
