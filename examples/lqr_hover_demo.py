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

The three runtimes are connected by `wire(producer, consumer)` over
typed `Signal` ports, so the loop is just `compute → step → step`: the
sim's noisy GPS flows to the EKF, the LQR's command flows to both the
sim (truth) and the EKF (known input), and the EKF's estimate flows back
to the LQR. No manual dict plumbing, no manual noise injection.

Both noise covariances are model-derived from one σ each: the GPS
`position_noise` gives the measurement noise R, and the thruster
`force_noise` gives the process noise Q (it enters the wrench, so it
shows up in the state and the EKF reads it off `∂f/∂noise`). The same σ
also drives the truth — a `NoiseDriver` jitters the GPS reading and the
thrust — so the filter's assumptions and the world it estimates are
consistent by construction. No hand-tuned Q.

Run::

    .venv/bin/python -m examples.lqr_hover_demo
"""

import numpy as np

from manta import (
    World, Craft, Sim, EKF, LQR, NoiseDriver, TargetNumpy, wire,
)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster

M, G = 2.0, 9.81
SIGMA_POS    = 0.05   # GPS white-noise σ (m)   — measurement noise → EKF R.
SIGMA_THRUST = 0.5    # thruster force σ (N)     — process noise → EKF Q.


def build_world():
    c = Craft("c")
    c.add(Mass("body", mass=M))
    # The thruster force_noise is the actuator analogue of the GPS noise:
    # one σ on the part drives BOTH the truth jitter (via NoiseDriver) and
    # the EKF's process noise Q (auto-built, since it enters the wrench →
    # state), exactly as SIGMA_POS drives both truth noise and R.
    c.add(Thruster("tx", force=(1, 0, 0), force_noise_sigma=SIGMA_THRUST))
    c.add(Thruster("ty", force=(0, 1, 0), force_noise_sigma=SIGMA_THRUST))
    c.add(Thruster("tz", force=(0, 0, 1), force_noise_sigma=SIGMA_THRUST))
    c.add(PositionSensor("gps", position_noise_sigma=SIGMA_POS))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10))
    return w, c


def main() -> None:
    target = np.array([0.0, 0.0, 10.0])
    dt = 0.02

    w, c = build_world()

    # Truth sim — start offset from the setpoint. The NoiseDriver draws
    # the GPS white noise the part declares, so `gps.position` is a
    # genuinely noisy fix (no manual rng injection, and the EKF's R is
    # built from the *same* σ).
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=7))
    sim.state["c"]["position"] = np.array([3.0, -2.0, 7.0])

    # Estimator — seed at the nominal; measurements pull it in.
    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"c": c.initial_state(position=(0, 0, 10))},
              P=np.eye(ekf.spec.tangent_dim) * 1.0)
    # No hand-tuned Q: both R and Q are model-derived now. R comes from the
    # GPS noise; Q is auto-built from the thruster force_noise (it enters
    # the wrench, so it propagates into the state and the EKF picks it up
    # via ∂f/∂noise). A nonzero Q keeps the filter listening to the GPS
    # instead of collapsing its covariance and going deaf.

    # Regulator about the setpoint; trim = weight on the z thruster.
    lqr = TargetNumpy(LQR(
        w,
        x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G},
        track=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))
    print(f"closed-loop |eig|max = "
          f"{np.max(np.abs(lqr._lqr.closed_loop_eigs)):.4f}  (stable < 1)\n")

    # Wire the three runtimes. GPS → estimator; command → truth + the
    # estimator's predict; estimate → regulator.
    wire(sim.out("c.gps.position"), ekf.meas("c.gps.position"))
    for nm in lqr.input_names:
        wire(lqr.command(nm), sim.command(nm))
        wire(lqr.command(nm), ekf.command(nm))
    wire(ekf.estimate, lqr.estimate)

    print(f"{'t (s)':>6}  {'true pos':>22}  {'est err':>9}  {'norm(u)':>8}")
    for i in range(500):
        u = lqr.compute()  # estimate → command (both wired downstream)
        sim.step(dt)       # apply command to truth; publish noisy GPS
        ekf.step(dt)       # predict on command; fold the fresh GPS fix

        if (i + 1) % 100 == 0:
            tp = np.asarray(sim.state["c"]["position"]).ravel()
            ep = ekf.state_dict()["c"]["position"].ravel()
            print(f"{(i+1)*dt:>6.2f}  {np.array2string(tp, precision=3):>22}  "
                  f"{np.linalg.norm(tp - ep):>9.4f}  "
                  f"{np.linalg.norm(list(u.values())):>8.2f}")

    final = np.asarray(sim.state["c"]["position"]).ravel()
    print(f"\nfinal position {np.round(final, 3)}  ->  target {target}  "
          f"(err {np.linalg.norm(final - target):.4f} m)")


if __name__ == "__main__":
    main()
