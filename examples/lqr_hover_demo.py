"""LQR setpoint demo — the three transforms closing a loop.

A 3-axis-thrust free-flyer (2 kg) is driven from an offset start back to
a target setpoint, with the *full* model → estimate → control stack:

    Sim(world)   — truth, integrated forward (the "real" vehicle)
    EKF(world)   — state estimate from noisy position measurements
    LQR(world)   — feedback control computed on the *estimate*

This is the payoff of the sibling-transform architecture: `Sim`, `EKF`,
and `LQR` all consume the same model and the same `Linearization` seam,
and snap together in a few lines. The LQR regulates position + velocity
(the controllable subspace for thrust-only actuation); attitude is
uncontrolled and frozen at the operating point.

Run::

    .venv/bin/python -m examples.lqr_hover_demo
"""

import numpy as np

from manta import World, Craft, Sim, EKF, LQR, TargetNumpy
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster

M, G = 2.0, 9.81


def build_world():
    c = Craft("c")
    c.add(Mass("body", mass=M))
    c.add(Thruster("tx", force=(1, 0, 0)))
    c.add(Thruster("ty", force=(0, 1, 0)))
    c.add(Thruster("tz", force=(0, 0, 1)))
    c.add(PositionSensor("gps"))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10))
    return w, c


def main() -> None:
    rng = np.random.default_rng(7)
    target = np.array([0.0, 0.0, 10.0])
    dt = 0.02

    w, c = build_world()

    # Truth sim — start offset from the setpoint.
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()
    state["c"]["position"] = np.array([3.0, -2.0, 7.0])

    # Estimator — seed at the nominal; measurements pull it in.
    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"c": c.initial_state(position=(0, 0, 10))},
              P=np.eye(ekf.spec.tangent_dim) * 1.0)

    # Regulator about the setpoint; trim = weight on the z thruster.
    lqr = TargetNumpy(LQR(
        w,
        x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G},
        track=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))
    print(f"closed-loop |eig|max = "
          f"{np.max(np.abs(lqr._lqr.closed_loop_eigs)):.4f}  (stable < 1)\n")

    sigma_pos = 0.05
    print(f"{'t (s)':>6}  {'true pos':>22}  {'est err':>9}  {'norm(u)':>8}")
    u = {n: 0.0 for n in lqr.input_names}
    for i in range(500):
        # Control on the current estimate. control() returns full input
        # names ("c.tx.throttle"); split the owner to apply into the
        # craft's nested state, and reuse the dict as the EKF's u below.
        u = lqr.control(ekf.state_dict())
        for full, v in u.items():
            owner, rest = full.split(".", 1)
            state[owner][rest] = v

        # Advance truth; take a noisy position fix.
        state = sim.step(state, dt=dt)
        z = np.asarray(state["c"]["gps.position"]).ravel() \
            + rng.normal(0, sigma_pos, 3)

        # Estimator predict (known command) + measurement update.
        ekf.predict(dt=dt, u=u)
        ekf.update(c.parts[-1], position=z)

        if (i + 1) % 100 == 0:
            tp = np.asarray(state["c"]["position"]).ravel()
            ep = ekf.state_dict()["c"]["position"].ravel()
            print(f"{(i+1)*dt:>6.2f}  {np.array2string(tp, precision=3):>22}  "
                  f"{np.linalg.norm(tp - ep):>9.4f}  "
                  f"{np.linalg.norm(list(u.values())):>8.2f}")

    final = np.asarray(state["c"]["position"]).ravel()
    print(f"\nfinal position {np.round(final, 3)}  ->  target {target}  "
          f"(err {np.linalg.norm(final - target):.4f} m)")


if __name__ == "__main__":
    main()
