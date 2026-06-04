"""Hydrofoil boat — drive it around while a gimballed laser tracks a buoy.

A passively-stable surface craft you steer with the keyboard, carrying a
two-axis (pan + tilt) **gimbal** built from nested `Joint`s. A small PD
loop drives the two joints so the mounted "laser" stays locked on a fixed
target buoy no matter how the boat moves — the laser ray is drawn in the
viewer, sweeping to follow the buoy as you drive.

What it showcases:
  * **Nested articulation** — a pan `Joint` (yaw) hosting a tilt `Joint`
    (pitch); each joint's reaction propagates down the chain to the hull.
  * **Passive stability** — four hull contact `Collider`s float the boat
    on a `CollisionField` water plane (pitch/roll-stable), a `Naca00xx`
    foil adds lift, and offset `DragSurface` fins damp yaw, so you only
    have to point-and-go.
  * A simple **PD pointing controller** closed on the joint angles.

Controls:  W/S throttle ahead/astern   A/D steer left/right

Run::

    .venv/bin/python -m examples.vehicles.hydrofoil --keyboard
    .venv/bin/python -m examples.vehicles.hydrofoil            # scripted
    .venv/bin/python -m examples.vehicles.hydrofoil --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import PID, Craft, Sim, TargetNumpy, World
from manta.fields import CollisionField, FluidField, GravityField
from manta.parts import Collider, DragSurface, Joint, Mass, Naca00xx, Thruster

from .._control import common_args, make_controller
from .._viz import Viz

RHO = 1025.0
MASS = 50.0
TARGET = np.array([18.0, 11.0, 1.5])    # the buoy the laser tracks
GIMBAL_MOUNT = np.array([0.6, 0.0, 0.35])   # gimbal base on the foredeck
LASER_LEN = 14.0
FWD, STEER = 450.0, 160.0


def build_world():
    b = Craft("boat")
    b.add(Mass("hull", mass=MASS, moi=(8.0, 20.0, 24.0)))
    # Four hull contact points float the boat on the water plane and make
    # it pitch/roll stable.
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        b.add(Collider(f"hull{i}", stiffness=1500.0, damping=200.0,
                       transform=(sx * 1.2, sy * 0.5, -0.25)))
    # Hull drag (terminal speed) + a lift foil + yaw-damping tail fin.
    b.add(DragSurface.isotropic_quadratic("drag", area=0.6,
                                          drag_coefficient=0.6))
    b.add(Naca00xx("foil", area=0.5, CL_max=1.1, CD_0=0.02, induced_k=0.1,
                   chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 0.0, 1.0),
                   transform=(0.0, 0.0, -0.4)))
    b.add(DragSurface.isotropic_quadratic("fin", area=0.5,
                                          drag_coefficient=1.0,
                                          transform=(-1.6, 0.0, 0.0)))
    b.add(Thruster("prop", force=(1.0, 0.0, 0.0), transform=(-1.6, 0, 0)))
    b.add(Thruster("rudder", torque=(0.0, 0.0, 1.0), transform=(-1.6, 0, 0)))

    # Two-axis laser gimbal: pan (yaw, +z) hosting tilt (pitch, +y).
    pan = Joint("pan", mode="saturating", stall_torque=8.0, damping=0.3,
                axis=(0.0, 0.0, 1.0), transform=tuple(GIMBAL_MOUNT))
    pan.add(Mass("pan_motor", mass=0.3, moi=(0.002, 0.002, 0.003)))
    tilt = Joint("tilt", mode="saturating", stall_torque=6.0, damping=0.3,
                 axis=(0.0, 1.0, 0.0), transform=(0.0, 0.0, 0.15))
    tilt.add(Mass("laser", mass=0.2, moi=(0.001, 0.001, 0.001)))
    pan.add(tilt)
    b.add(pan)

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=RHO))
         .add_field(CollisionField().add_half_space(origin=(0, 0, 0.0),
                                                    normal=(0, 0, 1))))
    w.add_craft(b, position=(0.0, 0.0, 0.18))
    return w


def _R(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.005
    duration = args.duration or (1e9 if args.keyboard else 16.0)

    sim = TargetNumpy(Sim(build_world()))

    # Scripted: drive a curving path so the laser has to sweep to track.
    script = [(1.0, 14.0, {"w"}), (3.0, 5.0, {"a"}), (7.0, 9.0, {"d"}),
              (11.0, 13.0, {"a"})]
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  W/S ahead/astern   A/D steer   (Ctrl-C to quit)\n")

    viz = None if args.no_viz else Viz("manta/hydrofoil")
    if viz is not None:
        viz.plane("world/water", z=0.0, size=30.0, color=(40, 90, 150, 130))
        viz.box("world/boat/hull", (1.4, 0.6, 0.2), color=(210, 190, 120))
        viz.box("world/boat/pan/tilt/laser_body", (0.1, 0.05, 0.05),
                color=(60, 60, 70))
        viz.point("world/target", TARGET, color=(90, 230, 120), radius=0.4)

    # One `PID` block per gimbal axis, sized to the (small) gimbal-joint
    # inertia: the integral trims the steady-state offset; the derivative
    # (on the measured angle) damps the rate. Same block you'd lower to C++
    # with TargetCpp(PID(...)) and run onboard.
    pid = {axis: TargetNumpy(PID(kp=2.0, ki=5.0, kd=0.5, integral_limit=2.0))
           for axis in ("pan", "tilt")}
    n = int(duration / dt)
    print(f"{'t (s)':>6} {'boat xy':>16} {'heading°':>9} {'pan°':>7} "
          f"{'tilt°':>7} {'aim err°':>9}")
    try:
        for i in range(n):
            t = i * dt
            ctrl.update(t)
            sim.state["boat"]["prop.throttle"] = \
                FWD * (ctrl.held("w") - ctrl.held("s"))
            sim.state["boat"]["rudder.throttle"] = \
                STEER * (ctrl.held("a") - ctrl.held("d"))

            # --- laser-pointing PD on the gimbal joints --------------------
            p = np.asarray(sim.state["boat"]["position"]).ravel()
            q = np.asarray(sim.state["boat"]["orientation"]).ravel()
            R = _R(q)
            base_w = p + R @ GIMBAL_MOUNT             # gimbal base, world
            v = R.T @ (TARGET - base_w)               # target in hull frame
            des_pan = np.arctan2(v[1], v[0])
            des_tilt = -np.arctan2(v[2], np.hypot(v[0], v[1]))
            for joint, des in (("pan", des_pan), ("tilt", des_tilt)):
                ang = float(sim.state["boat"][f"{joint}.angle"])
                # Fold the shortest-arc wrap into the setpoint so the PID's
                # error is _wrap(des − ang); its derivative-on-measurement
                # then damps the joint rate.
                sp = ang + _wrap(des - ang)
                sim.state["boat"][f"{joint}.torque_cmd"] = \
                    pid[joint].step(dt, setpoint=sp, measurement=ang)["command"]

            sim.step(dt)

            if viz is not None:
                p = np.asarray(sim.state["boat"]["position"]).ravel()
                q = np.asarray(sim.state["boat"]["orientation"]).ravel()
                pan = float(sim.state["boat"]["pan.angle"])
                tilt = float(sim.state["boat"]["tilt.angle"])
                viz.t(t)
                viz.pose("world/boat", p, q)
                viz.pose("world/boat/pan", GIMBAL_MOUNT,
                         (np.cos(pan/2), 0, 0, np.sin(pan/2)))
                viz.pose("world/boat/pan/tilt", (0, 0, 0.15),
                         (np.cos(tilt/2), 0, np.sin(tilt/2), 0))
                # Laser ray, drawn along +x of the tilt frame.
                viz.arrow("world/boat/pan/tilt/laser", (0, 0, 0),
                          (LASER_LEN, 0, 0), color=(255, 60, 60), radius=0.02)
                viz.trail("world/trail", p)   # world coords: not under the posed boat

            if (i + 1) % 200 == 0:
                p = np.asarray(sim.state["boat"]["position"]).ravel()
                q = np.asarray(sim.state["boat"]["orientation"]).ravel()
                R = _R(q)
                pan = float(sim.state["boat"]["pan.angle"])
                tilt = float(sim.state["boat"]["tilt.angle"])
                # Actual laser direction (world) vs. direction to target.
                d_hull = np.array([np.cos(pan)*np.cos(tilt),
                                   np.sin(pan)*np.cos(tilt), -np.sin(tilt)])
                d_world = R @ d_hull
                to_tgt = TARGET - (p + R @ GIMBAL_MOUNT)
                to_tgt /= np.linalg.norm(to_tgt)
                aim_err = np.degrees(np.arccos(
                    np.clip(d_world @ to_tgt, -1, 1)))
                head = np.degrees(np.arctan2(
                    2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)))
                print(f"{t:>6.2f} {f'({p[0]:.1f},{p[1]:.1f})':>16} "
                      f"{head:>9.1f} {np.degrees(pan):>7.1f} "
                      f"{np.degrees(tilt):>7.1f} {aim_err:>9.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
