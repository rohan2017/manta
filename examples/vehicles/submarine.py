"""Submarine — direct piloting with an Error-State EKF watching along.

A neutrally-buoyant AUV in a seawater `FluidField`: `Mass` + `PointBuoy`
(displacement set to cancel weight) + hydrodynamic `DragSurface`, driven
by a stern propeller, a vertical thruster, and a yaw thruster. The buoy
sits above the centre of mass, so the hull is **pendulum-stable** in
roll/pitch — you only steer heading and depth.

You pilot it directly (no controller in the loop). Meanwhile an `EKF`
fuses the noisy IMU + DVL (body-frame velocity) + position fixes into a
state estimate, and the terminal reports the estimate error so you can
watch the filter stay locked on while you fly. Process noise comes from
the propeller `force_noise`; measurement noise from each sensor's σ — all
model-derived, and the same σ drive the truth via a `NoiseDriver`.

Controls:  W/S ahead/astern   A/D yaw left/right   space/shift rise/dive

Run::

    .venv/bin/python -m examples.vehicles.submarine --keyboard
    .venv/bin/python -m examples.vehicles.submarine            # scripted
    .venv/bin/python -m examples.vehicles.submarine --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, EKF, NoiseDriver, Sim, TargetNumpy, World, wire
from manta.fields import FluidField, GravityField, MagField
from manta.parts import (
    DVL, DragSurface, IMU, Magnetometer, Mass, PointBuoy, PositionSensor,
    Thruster,
)

from .._control import common_args, make_controller
from .._viz import Viz

RHO = 1025.0           # seawater density
MASS = 120.0
FWD, VERT, YAW = 600.0, 250.0, 120.0    # command magnitudes (N, N, N·m)


def build_world():
    sub = Craft("sub")
    sub.add(Mass("hull", mass=MASS, moi=(3.0, 12.0, 12.0)))
    # Neutral buoyancy, buoy above CoM for pendulum (metacentric) stability.
    sub.add(PointBuoy("buoy", volume=MASS / RHO, transform=(0.0, 0.0, 0.15)))
    sub.add(DragSurface.isotropic_quadratic("drag", area=0.09,
                                            drag_coefficient=0.5))
    # Tail fins: an offset drag panel sees the local ω×r velocity, so it
    # damps yaw + pitch rates — the hull's hydrodynamic stability that keeps
    # heading from running away under a held yaw command.
    sub.add(DragSurface.isotropic_quadratic("fin", area=0.35,
                                            drag_coefficient=1.1,
                                            transform=(-1.3, 0.0, 0.0)))
    sub.add(Thruster("prop", force=(1.0, 0.0, 0.0), transform=(-1.0, 0, 0),
                     force_noise_sigma=3.0))
    sub.add(Thruster("vert", force=(0.0, 0.0, 1.0)))
    sub.add(Thruster("yaw", torque=(0.0, 0.0, 1.0)))
    sub.add(IMU("imu", gyro_noise_sigma=0.01))
    sub.add(DVL("dvl", velocity_noise_sigma=0.02))
    sub.add(PositionSensor("gps", position_noise_sigma=0.1))
    sub.add(Magnetometer("mag", B_noise_sigma=1.0e-6))     # compass (heading)
    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=RHO))
         .add_field(MagField().add_uniform((2.0e-5, 0.0, -4.0e-5))))
    w.add_craft(sub, position=(0.0, 0.0, -8.0))     # 8 m depth
    return w, sub


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.02
    duration = args.duration or (1e9 if args.keyboard else 16.0)

    w, c = build_world()
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=5))

    ekf_ir = EKF(w)
    ekf = TargetNumpy(ekf_ir)
    ekf.reset(state={"sub": c.initial_state(position=(0.0, 0.0, -8.0))},
              P=np.eye(ekf.spec.tangent_dim) * 0.1)
    # Wire the sensor + command bus: position fix + DVL body-velocity +
    # gyro + compass. The compass fixes absolute heading — without it,
    # gyro-integrated heading drifts a few degrees through a turn, and that
    # rotates the DVL body-velocity into a growing dead-reckoning error.
    # (We leave the IMU accelerometer out of the attitude solution: under
    # thrust its specific-force reading is dominated by acceleration, not
    # gravity, so it would corrupt heading.)
    wire(sim.out("sub.gps.position"), ekf.meas("sub.gps.position"))
    wire(sim.out("sub.dvl.velocity"), ekf.meas("sub.dvl.velocity"))
    wire(sim.out("sub.imu.gyro"), ekf.meas("sub.imu.gyro"))
    wire(sim.out("sub.mag.B"), ekf.meas("sub.mag.B"))

    # Why the compass earns its place: an observability check on the
    # symbolic F/H shows that at rest GPS + DVL + gyro can't see absolute
    # heading (it's only observable while turning) — so it has no absolute
    # reference and can wander on long straight legs. The magnetometer
    # makes orientation observable everywhere.
    no_compass = ekf_ir.observability(
        sensors=["sub.imu.gyro", "sub.dvl.velocity", "sub.gps.position"])
    full_rep = ekf_ir.observability()
    print(f"observability — full suite: {full_rep.rank}/"
          f"{full_rep.tangent_dim}; without compass: "
          f"{no_compass.rank}/{no_compass.tangent_dim} "
          f"(unobservable: {', '.join(n for n, _ in no_compass.unobservable) or 'none'})\n")

    script = [(1.0, 6.0, {"w"}), (3.0, 5.0, {"d"}), (7.0, 11.0, {"shift"}),
              (9.0, 11.0, {"a"}), (12.0, 15.0, {"w"})]
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  W/S ahead/astern   A/D yaw   space/shift rise/dive"
              "   (Ctrl-C to quit)\n")

    viz = None if args.no_viz else Viz("manta/submarine")
    if viz is not None:
        viz.plane("world/surface", z=0.0, size=40.0, color=(40, 90, 160, 120))
        viz.box("world/sub/hull", (1.0, 0.18, 0.18), color=(230, 210, 80))
        viz.box("world/sub_est/hull", (1.0, 0.18, 0.18),
                color=(120, 120, 120, 120))

    n = int(duration / dt)
    print(f"{'t (s)':>6} {'depth (m)':>10} {'vx (m/s)':>9} {'heading°':>9} "
          f"{'est err (m)':>11}")
    try:
        for i in range(n):
            t = i * dt
            ctrl.update(t)
            cmd = {
                "prop.throttle": FWD * (ctrl.held("w") - ctrl.held("s")),
                "vert.throttle": VERT * (ctrl.held("space") - ctrl.held("shift")),
                "yaw.throttle": YAW * (ctrl.held("a") - ctrl.held("d")),
            }
            for name, val in cmd.items():
                sim.command(name).set(val)
                ekf.command(name).set(val)

            sim.step(dt)
            ekf.step(dt)

            tp = np.asarray(sim.state["sub"]["position"]).ravel()
            tq = np.asarray(sim.state["sub"]["orientation"]).ravel()
            est = ekf.state_dict()["sub"]
            ep = np.asarray(est["position"]).ravel()
            if viz is not None:
                viz.t(t)
                viz.pose("world/sub", tp, tq)
                viz.pose("world/sub_est", ep,
                         np.asarray(est["orientation"]).ravel())
                viz.trail("world/sub/trail", tp)

            if (i + 1) % 50 == 0:
                v = np.asarray(sim.state["sub"]["velocity"]).ravel()
                heading = np.degrees(np.arctan2(
                    2 * (tq[0]*tq[3] + tq[1]*tq[2]),
                    1 - 2 * (tq[2]**2 + tq[3]**2)))
                print(f"{t:>6.2f} {tp[2]:>10.2f} {v[0]:>9.3f} {heading:>9.1f} "
                      f"{np.linalg.norm(tp - ep):>11.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
