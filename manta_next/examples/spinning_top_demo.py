"""Spinning top demo — gyroscopic precession via a passive Joint.

Demonstrates the rotor-on-Joint composition: the rotor is a `Mass`
attached to a passive `Joint`. The Joint's update() carries the
gyroscopic-torque-on-body correction (-ω_body × L_rotor), so a body
carrying a spinning rotor subjected to a torque about a perpendicular
axis **precesses** about the rotor's spin axis instead of toppling —
the classic top behavior.

Two scenarios, same rig:

  1. Rotor SPINNING at 100 rad/s — body precesses (ω_x and ω_y
     oscillate in phase quadrature).
  2. Rotor STATIONARY (rate = 0) — body tumbles wildly about the
     impulse axis because it has no angular momentum to resist
     toppling.

Run::

    .venv/bin/python -m manta_next.examples.spinning_top_demo
"""

import numpy as np

from manta_next import Craft, World, TargetNumpy
from manta_next.fields import GravityField
from manta_next.parts import Joint, Mass, Thruster


def _build_top() -> Craft:
    """Build the top craft. Body MOI is comparable to rotor MOI·rate so
    the precession period is many tick-times — keeps the symplectic
    integrator stable. Rotor sized so the precession frequency ω_p ≈
    I_rotor·rate / I_body is well below the sample frequency (1 kHz)."""
    c = Craft("top")
    c.add(Mass("body", mass=0.2, moi=(0.005, 0.005, 0.001)))

    # Rotor on a passive Joint: the Joint integrates angle/rate (no
    # commanded torque in passive mode) and applies the gyroscopic
    # reaction on the body. The Mass child provides the rotor's
    # inertia distribution; its I_zz becomes the joint's I_axial about
    # the +z spin axis.
    rotor_joint = Joint("rotor", mode="passive", axis=(0.0, 0.0, 1.0))
    rotor_joint.add(Mass("rotor_disk",
                          mass=0.05,
                          moi=(0.0001, 0.0001, 0.001)))
    c.add(rotor_joint)

    # Off-axis thruster that we'll pulse briefly to create a lateral
    # body-frame torque (the disturbance whose response we observe).
    c.add(Thruster("kick", force=(0.0, 0.0, 1.0),
                   transform=(0.0, 0.1, 0.0)))
    return c


def _run_scenario(label: str, rotor_rate: float) -> None:
    """Apply a brief lateral impulse and print angular velocity history."""
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))   # gravity off
    c = _build_top()
    w.add_craft(c)
    cw = TargetNumpy(w.compile())

    state = cw.initial_state()
    state["top"]["rotor.rate"] = rotor_rate

    dt = 0.001
    n  = 3000     # 3 seconds

    print(f"\n=== {label}  rotor_rate = {rotor_rate} rad/s ===")
    print(f"{'t (s)':>5} {'ω_x':>10} {'ω_y':>10} {'ω_z':>10} {'rotor':>9}")
    for i in range(n):
        t = i * dt
        thrust = 0.5 if 0.1 <= t < 0.2 else 0.0
        state["top"]["kick.throttle"] = thrust
        state = cw.step(state, dt=dt)

        if i in (99, 199, 499, 999, 1999, 2999):
            omega = np.array(state["top"]["angular_velocity"]).ravel()
            rotor_rate_curr = float(state["top"]["rotor.rate"])
            print(f"{t:>5.2f} {omega[0]:>10.4f} {omega[1]:>10.4f} "
                  f"{omega[2]:>10.4f} {rotor_rate_curr:>9.1f}")


def main() -> None:
    print("Spinning top demo — lateral impulse during t ∈ [0.1, 0.2] s.")
    print("Expected: spinning rotor → precession (ω_x and ω_y oscillate),")
    print("          stationary rotor → toppling (large ω_x accumulates).")
    _run_scenario("SPINNING",  rotor_rate=100.0)
    _run_scenario("STATIONARY", rotor_rate=0.0)


if __name__ == "__main__":
    main()
