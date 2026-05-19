"""Hover demo (M9 end-to-end showcase).

Runs a 3-second simulation of a 1.5 kg drone hovering against gravity
under commanded thrust, with noisy IMU gyro + GPS-style position
measurements feeding an Error-State EKF. Prints a per-second tracking
summary so you can watch the filter pull in from its initial offset.

Run::

    .venv/bin/python -m manta_next.examples.hover_demo
"""

import numpy as np

from manta_next import Craft, World
from manta_next.ekf import EKF, measurement_slot
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


def main() -> None:
    rng = np.random.default_rng(seed=7)
    g_world = (0.0, 0.0, -9.81)
    m = 1.5

    # Build the craft.
    c = Craft("drone")
    c.add(Mass("body", mass=m, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t"))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))

    # Sim — through the public World/CompiledWorld surface.
    w = World().set_gravity(g_world)
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    cw = w.compile()
    sim = cw.initial_state()

    # EKF — start with a deliberate position + velocity offset.
    ekf = EKF(c, gravity_anchor=g_world)
    init = c.initial_state(position=(0.0, 0.0, 4.0),
                           velocity=(0.5, 0.0, 0.0))
    ekf.reset(state=init,
              P=np.eye(ekf.spec.tangent_dim) * 1e-1)

    # Noise models.
    sigma_gyro = 0.01
    sigma_pos  = 0.05
    R_gyro = (sigma_gyro ** 2) * np.eye(3)
    R_pos  = (sigma_pos  ** 2) * np.eye(3)
    Q = np.diag([1e-6] * 3 + [1e-6] * 3 + [1e-4] * 3 + [1e-5] * 3)

    h_pos  = measurement_slot(ekf.spec, "position")
    h_gyro = measurement_slot(ekf.spec, "angular_velocity")

    dt = 0.005
    n  = 600
    thrust = m * 9.81

    print(f"{'t (s)':>6}  {'truth_z (m)':>11}  {'est_z (m)':>10}  "
          f"{'pos err (m)':>11}  {'vel err (m/s)':>13}  {'tr(P)':>8}")
    for i in range(n):
        sim["drone"]["t.thrust_cmd"] = thrust
        sim = cw.step(sim, dt=dt)

        gyro = np.array(sim["drone"]["g.gyro"]).ravel()
        pos  = np.array(sim["drone"]["gps.position"]).ravel()
        gyro_meas = gyro + rng.normal(0.0, sigma_gyro, 3)
        pos_meas  = pos  + rng.normal(0.0, sigma_pos,  3)

        ekf.predict(dt=dt, u={"t.thrust_cmd": thrust}, Q=Q)
        ekf.update(h_gyro, gyro_meas, R_gyro)
        if i % 4 == 0:
            ekf.update(h_pos, pos_meas, R_pos)

        if (i + 1) % 200 == 0:
            est = ekf.state_dict()
            truth_p = np.array(sim["drone"]["position"]).ravel()
            truth_v = np.array(sim["drone"]["velocity"]).ravel()
            pos_err = float(np.linalg.norm(truth_p - est["position"].ravel()))
            vel_err = float(np.linalg.norm(truth_v - est["velocity"].ravel()))
            print(f"{(i+1)*dt:>6.2f}  {truth_p[2]:>11.4f}  "
                  f"{est['position'][2]:>10.4f}  "
                  f"{pos_err:>11.4f}  {vel_err:>13.4f}  "
                  f"{np.trace(ekf.P):>8.4f}")


if __name__ == "__main__":
    main()
