"""Glider — a NACA wing flown fly-by-wire under gravity.

An unpowered glider trades altitude for airspeed; the `Naca00xx` wing
turns that airspeed into lift. The wing sits at the centre of mass, so it
makes no pitch moment — the glide is neutrally stable. You steer through a
small **rate-damping inner loop**: the keys command a body angular rate,
and a proportional controller drives three pure-torque control surfaces
(elevator / aileron / rudder) to track it. Release the keys and the loop
damps the rates back to zero, so the glider holds its attitude instead of
tumbling.

Showcases `Naca00xx` aerodynamics in a `FluidField` atmosphere, plus a
tiny stabilizer loop closed on the IMU. Without ``--keyboard`` a scripted
run pitches up, then banks into a turn.

Controls:  W/S pitch up/down   A/D roll left/right   Q/E yaw left/right

Run::

    .venv/bin/python -m examples.vehicles.glider --keyboard
    .venv/bin/python -m examples.vehicles.glider             # scripted
    .venv/bin/python -m examples.vehicles.glider --no-viz    # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import IMU, Mass, Naca00xx, Thruster

from .._control import common_args, make_controller
from .._viz import Viz

TORQUE = 2.0          # N·m at full surface deflection
RATE_CMD = 0.7        # rad/s commanded per held key
KP = 3.0              # rate-loop proportional gain


def build_world():
    g = Craft("glider")
    g.add(Mass("body", mass=5.0, moi=(0.4, 0.6, 0.9)))
    g.add(Naca00xx("wing", area=1.5, CL_max=1.2, CD_0=0.02, induced_k=0.05,
                   chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 0.0, 1.0)))
    g.add(Thruster("elevator", torque=(0.0, TORQUE, 0.0)))   # pitch (body y)
    g.add(Thruster("aileron", torque=(TORQUE, 0.0, 0.0)))    # roll  (body x)
    g.add(Thruster("rudder", torque=(0.0, 0.0, TORQUE)))     # yaw   (body z)
    g.add(IMU("imu"))
    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=1.225)))
    w.add_craft(g, position=(0.0, 0.0, 120.0), velocity=(18.0, 0.0, 0.0))
    return w


def _body_rates(q_wxyz, omega_world):
    """World angular velocity → body frame (R(q)ᵀ · ω)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])
    return R.T @ np.asarray(omega_world).ravel()


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.01
    duration = args.duration or (1e9 if args.keyboard else 14.0)

    sim = TargetNumpy(Sim(build_world()))
    state = sim.initial_state()

    script = [(2.0, 3.5, {"w"}), (5.0, 8.0, {"d"}), (9.0, 10.0, {"a"})]
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  W/S pitch   A/D roll   Q/E yaw   (Ctrl-C to quit)\n")

    viz = None if args.no_viz else Viz("manta/glider")
    if viz is not None:
        viz.box("world/glider/body", (0.4, 1.5, 0.05), color=(210, 210, 220))

    n = int(duration / dt)
    print(f"{'t (s)':>6} {'x (m)':>8} {'z (m)':>8} {'|v| (m/s)':>9} "
          f"{'gamma°':>7} {'roll°':>7}")
    try:
        for i in range(n):
            t = i * dt
            ctrl.update(t)
            q = np.asarray(state["glider"]["orientation"]).ravel()
            omega_b = _body_rates(q, state["glider"]["angular_velocity"])
            # Commanded body rates from the keys, then a P rate-tracker.
            cmd = RATE_CMD * np.array([
                ctrl.held("d") - ctrl.held("a"),      # roll  (body x)
                ctrl.held("w") - ctrl.held("s"),      # pitch (body y)
                ctrl.held("e") - ctrl.held("q")])     # yaw   (body z)
            defl = np.clip(KP * (cmd - omega_b) / TORQUE, -1.0, 1.0)
            state["glider"]["aileron.throttle"] = float(defl[0])
            state["glider"]["elevator.throttle"] = float(defl[1])
            state["glider"]["rudder.throttle"] = float(defl[2])
            state = sim.step(state, dt=dt)

            p = np.asarray(state["glider"]["position"]).ravel()
            v = np.asarray(state["glider"]["velocity"]).ravel()
            if viz is not None:
                viz.t(t)
                viz.pose("world/glider", p, np.asarray(
                    state["glider"]["orientation"]).ravel())
                viz.trail("world/glider/trail", p)
                viz.arrow("world/glider/vel", p, v * 0.3, color=(90, 200, 120),
                          radius=0.1)

            if p[2] < 0:
                print(f"\nlanded at t = {t:.2f} s, range = "
                      f"{np.hypot(p[0], p[1]):.1f} m")
                break
            if (i + 1) % 100 == 0:
                q = np.asarray(state["glider"]["orientation"]).ravel()
                gamma = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
                roll = np.degrees(np.arctan2(
                    2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]**2 + q[2]**2)))
                print(f"{t:>6.2f} {p[0]:>8.1f} {p[2]:>8.1f} "
                      f"{np.linalg.norm(v):>9.2f} {gamma:>+7.1f} {roll:>+7.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
