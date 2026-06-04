"""Quadcopter — the full estimate→control stack, flown to a setpoint.

This is the payoff demo for manta's sibling-transform architecture. One
model is consumed by three transforms that snap together over the shared
`Linearization` seam and the `Signal` bus:

    Sim(world)   — truth: a 4-rotor quad integrated forward, with process
                   noise (thruster `force_noise`) jittering the real motion
    EKF(world)   — state estimate (position, attitude, velocity, rates, and
                   the IMU gyro bias) from a noisy GPS fix + gyro
    LQR(world)   — a full-state hover regulator; feedback on the *estimate*

`wire(producer, consumer)` connects the runtimes over typed ports: the
sim's noisy GPS + gyro flow to the EKF, the commands flow to both the sim
(truth) and the EKF (known input). The control loop is then just
``control → step → step``.

Both noise covariances are **model-derived** from one σ each: GPS
`position_noise` → the EKF's R; thruster `force_noise` → the process noise
Q (it enters the wrench, so the EKF reads it off ∂f/∂noise). A
`NoiseDriver` draws those same σ to jitter the truth, so the filter's
assumptions and the world it estimates are consistent by construction.

You fly it: WASD moves the horizontal setpoint, space/shift change
altitude — the LQR chases the moving target. Without ``--keyboard`` a
scripted tour flies a box so the demo runs unattended.

Run::

    .venv/bin/python -m examples.vehicles.quadcopter --keyboard
    .venv/bin/python -m examples.vehicles.quadcopter            # scripted
    .venv/bin/python -m examples.vehicles.quadcopter --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import (
    Craft, EKF, LQR, NoiseDriver, Sim, TargetNumpy, World, wire,
)
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

from .._control import common_args, make_controller
from .._viz import Viz

L, M, G = 0.25, 1.0, 9.81          # arm length, mass, gravity
MAXT, KYAW = 6.0, 0.05             # per-rotor max thrust (N), yaw coeff
SIGMA_FORCE = 0.15                 # thruster force σ (N)  → process noise Q
SIGMA_GPS = 0.04                   # GPS σ (m)             → measurement R
SIGMA_GYRO = 0.01                  # gyro σ (rad/s)        → measurement R
ROTORS = ["front", "right", "back", "left"]
ARM = {"front": (+L, 0, 0), "right": (0, +L, 0),
       "back": (-L, 0, 0), "left": (0, -L, 0)}


def build_world():
    c = Craft("quad")
    c.add(Mass("body", mass=M, moi=(0.01, 0.01, 0.02)))
    F = (0.0, 0.0, MAXT)
    chir = {"front": +1, "right": -1, "back": +1, "left": -1}   # CW/CCW
    for name in ROTORS:
        c.add(Thruster(name, force=F,
                       torque=(0, 0, chir[name] * KYAW * MAXT),
                       transform=ARM[name], force_noise_sigma=SIGMA_FORCE))
    c.add(IMU("imu", gyro_noise_sigma=SIGMA_GYRO, gyro_bias_sigma=2e-4))
    c.add(PositionSensor("gps", position_noise_sigma=SIGMA_GPS))
    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(c, position=(0, 0, 2.0))
    return w, c


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.01
    duration = args.duration or (1e9 if args.keyboard else 13.0)
    p0 = np.array([0.0, 0.0, 2.0])         # LQR operating point (hover here)
    hover = M * G / 4.0 / MAXT             # throttle per rotor at trim

    w, c = build_world()

    # Truth, with a noise driver so process + sensor noise are real. The
    # quad starts below and to the side of the hover point and flies up to
    # it under control.
    start = np.array([0.2, -0.15, 1.85])
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=11))
    sim.state["quad"]["position"] = start.copy()

    # Estimator — seeded at the (known) launch pose; R and Q are
    # model-derived. A tight initial P reflects that we know where we
    # started; the filter's job is then to reject the GPS/gyro noise while
    # the LQR flies on its estimate.
    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"quad": c.initial_state(position=tuple(start))},
              P=np.eye(ekf.spec.tangent_dim) * 0.2)

    # Full-state hover regulator (position + attitude + their rates).
    Q = np.diag([8, 8, 8, 0.5, 0.5, 0.5, 2, 2, 1, 0.2, 0.2, 0.2])
    lqr_t = LQR(
        w, x_ref={"quad": {"position": tuple(p0)}},
        u_ref={f"{r}.throttle": hover for r in ROTORS},
        track=["quad.position", "quad.velocity",
               "quad.orientation", "quad.angular_velocity"],
        Q=Q, R=np.eye(4) * 0.5, dt=dt)
    lqr = TargetNumpy(lqr_t)
    print(f"closed-loop |eig|max = "
          f"{np.max(np.abs(lqr_t.closed_loop_eigs)):.4f}  (stable < 1)")

    # Wire the sensor + command bus: GPS + gyro + accel → EKF; commands →
    # truth + EKF. The accelerometer sees the gravity direction, which is
    # what makes the quad's roll/pitch observable (GPS + gyro alone leave
    # attitude weakly observable, and a full-state regulator needs it).
    wire(sim.out("quad.gps.position"), ekf.meas("quad.gps.position"))
    wire(sim.out("quad.imu.gyro"), ekf.meas("quad.imu.gyro"))
    wire(sim.out("quad.imu.accel"), ekf.meas("quad.imu.accel"))

    # Keyboard: WASD = horizontal, space/shift = up/down. (Scripted: a box.)
    script = [(1.5, 3.5, {"w"}), (3.5, 5.5, {"d"}), (5.5, 7.0, {"space"}),
              (7.0, 9.0, {"s"}), (9.0, 11.0, {"a"}), (11.0, 12.5, {"shift"})]
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("\nControls:  W/S forward/back   A/D left/right   "
              "space/shift up/down   (Ctrl-C to quit)\n")

    viz = None if args.no_viz else Viz("manta/quadcopter", addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=12.0, color=(60, 70, 80, 160))
        viz.box("world/quad/body", (L, L, 0.03), color=(80, 140, 220))
        viz.box("world/quad_est/body", (L, L, 0.03), color=(120, 120, 120, 120))

    setpoint = p0.copy()
    sp_rate = 0.8                          # m/s setpoint slew while held
    n = int(duration / dt)
    print(f"\n{'t (s)':>6}  {'true pos':>22}  {'setpoint':>18}  {'est err':>8}")
    try:
        for i in range(n):
            t = i * dt
            ctrl.update(t)
            move = np.array([
                (ctrl.held("w") - ctrl.held("s")),
                (ctrl.held("d") - ctrl.held("a")),
                (ctrl.held("space") - ctrl.held("shift"))], dtype=float)
            setpoint += move * sp_rate * dt

            # Retarget the hover regulator to the moving setpoint by
            # offsetting the position it sees (position enters the control
            # law linearly, so shifting the estimate shifts the goal).
            est = ekf.state_dict()["quad"]
            shifted = dict(est)
            shifted["position"] = np.asarray(est["position"]).ravel() \
                - (setpoint - p0)
            u = lqr.control({"quad": shifted})

            for name, val in u.items():
                cmd = float(np.clip(val, 0.0, 1.2))
                sim.command(name).set(cmd)
                ekf.command(name).set(cmd)

            sim.step(dt)                   # apply commands; publish GPS+gyro
            ekf.step(dt)                   # predict on commands; fold sensors

            if viz is not None:
                tp = np.asarray(sim.state["quad"]["position"]).ravel()
                tq = np.asarray(sim.state["quad"]["orientation"]).ravel()
                ep = np.asarray(est["position"]).ravel()
                eq = np.asarray(est["orientation"]).ravel()
                viz.t(t)
                viz.pose("world/quad", tp, tq)
                viz.pose("world/quad_est", ep, eq)
                viz.trail("world/trail", tp)   # world coords: not under the posed quad
                viz.point("world/setpoint", setpoint, color=(90, 230, 120),
                          radius=0.08)
                for r in ROTORS:                       # thrust arrows (body)
                    viz.arrow(f"world/quad/{r}", ARM[r], (0, 0, 0.4 * u[
                        f"quad.{r}.throttle"] / hover), color=(255, 120, 60),
                        radius=0.015)

            if (i + 1) % 100 == 0:
                tp = np.asarray(sim.state["quad"]["position"]).ravel()
                ep = np.asarray(ekf.state_dict()["quad"]["position"]).ravel()
                print(f"{t:>6.2f}  {np.round(tp, 2) + 0.0!s:>22}  "
                      f"{np.round(setpoint, 2) + 0.0!s:>18}  "
                      f"{np.linalg.norm(tp - ep):>8.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()

    final = np.asarray(sim.state["quad"]["position"]).ravel()
    print(f"\nfinal pos {np.round(final, 2) + 0.0}"
          f"  setpoint {np.round(setpoint, 2) + 0.0}"
          f"  (err {np.linalg.norm(final - setpoint):.3f} m)")


if __name__ == "__main__":
    main()
