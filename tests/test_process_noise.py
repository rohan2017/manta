"""Model-level process noise — a Noise channel on a force-producing part.

Process noise is declared exactly like measurement noise: a `Noise`
channel on a part. The difference is purely where the part uses it — a
sensor adds it into an Output (→ EKF builds R), an actuator adds it into
its wrench (→ it propagates through the dynamics into the next state, so
the EKF auto-builds Q from ∂f/∂noise and a NoiseDriver jitters the
truth). Same machinery, both covariances model-derived.
"""

import numpy as np

from manta import (
    Craft, EKF, LQR, NoiseDriver, Sim, TargetNumpy, World, wire,
)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster


def _free_flyer(force_sigma):
    c = Craft("c")
    c.add(Mass("body", mass=2.0))
    c.add(Thruster("tz", force=(0, 0, 1), force_noise_sigma=force_sigma))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))
    return w, c


def test_thruster_noise_inert_by_default():
    """σ=0 ⇒ no process noise: the EKF's auto-Q velocity block is zero."""
    w, _ = _free_flyer(0.0)
    rt = TargetNumpy(EKF(w))
    ekf = rt._ekf
    L = np.asarray(ekf._L_fn(rt.x, ekf._u_defaults, 0.02, 0.0))
    Q = L @ ekf._Sigma @ L.T
    assert np.allclose(Q, 0.0)


def test_thruster_noise_builds_velocity_Q():
    """Force noise enters the wrench → next velocity, so the EKF's auto-Q
    has a velocity block of (dt/m)²·σ² and nothing on position/attitude."""
    sigma, dt, m = 0.5, 0.02, 2.0
    w, _ = _free_flyer(sigma)
    rt = TargetNumpy(EKF(w))
    ekf = rt._ekf
    L = np.asarray(ekf._L_fn(rt.x, ekf._u_defaults, dt, 0.0))
    Q = L @ ekf._Sigma @ L.T
    # Q/L are tangent-indexed; use tangent_offset (not the ambient offset).
    vel = ekf.spec.slot("c.velocity")
    qv = np.diag(Q)[vel.tangent_offset:vel.tangent_offset + 3]
    np.testing.assert_allclose(qv, (dt / m) ** 2 * sigma ** 2, rtol=1e-6)
    # Position picks up only a second-order O(dt²) share (semi-implicit
    # integration steps velocity first, then position with the new
    # velocity) — orders of magnitude below the velocity block.
    pos = ekf.spec.slot("c.position")
    qp = np.diag(Q)[pos.tangent_offset:pos.tangent_offset + 3]
    assert np.all(qp < qv * dt)        # ~dt² vs dt⁰ scaling


def test_driver_jitters_truth_from_thruster_noise():
    """The same σ that builds Q also perturbs the truth thrust when a
    NoiseDriver is attached: velocity std after one step ≈ dt/m·σ."""
    sigma, dt, m = 0.5, 0.02, 2.0
    w, _ = _free_flyer(sigma)
    sim = TargetNumpy(Sim(w))
    ends = []
    for trial in range(400):
        sim.attach_driver(NoiseDriver(seed=trial))
        s = sim.initial_state()
        s = sim.step(s, dt=dt)
        ends.append(np.asarray(s["c"]["velocity"]).ravel())
    np.testing.assert_allclose(
        np.asarray(ends).std(axis=0), dt / m * sigma, rtol=0.12)


def test_model_Q_keeps_filter_alive_in_closed_loop():
    """End to end: with model process noise and NO hand-tuned Q, the
    closed loop regulates to the setpoint (the filter doesn't go deaf)."""
    M, G = 2.0, 9.81
    target = np.array([0.0, 0.0, 10.0])
    dt = 0.02

    def build():
        c = Craft("c")
        c.add(Mass("body", mass=M))
        for n, f in [("tx", (1, 0, 0)), ("ty", (0, 1, 0)), ("tz", (0, 0, 1))]:
            c.add(Thruster(n, force=f, force_noise_sigma=0.5))
        c.add(PositionSensor("gps", position_noise_sigma=0.05))
        w = World().add_field(GravityField(g=(0, 0, -G)))
        w.add_craft(c, position=(0, 0, 10))
        return w, c

    w, c = build()
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=7))
    sim.state["c"]["position"] = np.array([3.0, -2.0, 7.0])

    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"c": c.initial_state(position=(0, 0, 10))},
              P=np.eye(ekf.spec.tangent_dim))
    assert ekf.Q is None                      # no hand-tuned process noise

    lqr = TargetNumpy(LQR(
        w, x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, track=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))

    wire(sim.out("c.gps.position"), ekf.meas("c.gps.position"))
    for nm in lqr.input_names:
        wire(lqr.command(nm), sim.command(nm))
        wire(lqr.command(nm), ekf.command(nm))
    wire(ekf.estimate, lqr.estimate)

    for _ in range(500):
        lqr.compute()
        sim.step(dt)
        ekf.step(dt)

    final = np.asarray(sim.state["c"]["position"]).ravel()
    est = ekf.state_dict()["c"]["position"].ravel()
    assert np.linalg.norm(final - target) < 0.5   # regulated despite jitter
    assert np.linalg.norm(final - est) < 0.2       # filter stays healthy
