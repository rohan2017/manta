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
    EKF,
    LQR,
    Craft,
    NoiseDriver,
    Sim,
    TargetNumpy,
    World,
)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, ProcessNoise, Thruster


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
    ekf = EKF(w)
    rt = TargetNumpy(ekf)
    L = np.asarray(ekf.sys.L_fn(rt.x, ekf.sys.u_defaults, 0.02, 0.0))
    Q = L @ ekf.sys.Sigma @ L.T
    assert np.allclose(Q, 0.0)


def test_thruster_noise_builds_velocity_Q():
    """Force noise enters the wrench → next velocity, so the EKF's auto-Q
    has a velocity block of (dt/m)²·σ² and nothing on position/attitude."""
    sigma, dt, m = 0.5, 0.02, 2.0
    w, _ = _free_flyer(sigma)
    ekf = EKF(w)
    rt = TargetNumpy(ekf)
    L = np.asarray(ekf.sys.L_fn(rt.x, ekf.sys.u_defaults, dt, 0.0))
    Q = L @ ekf.sys.Sigma @ L.T
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
        sim.state = sim.initial_state()         # fresh start per trial
        sim.step(dt)
        ends.append(np.asarray(sim.state["c"]["velocity"]).ravel())
    np.testing.assert_allclose(
        np.asarray(ends).std(axis=0), dt / m * sigma, rtol=0.12)


def _drifter(force_sigma=0.0, torque_sigma=0.0):
    """A craft with NO actuator — the case the ProcessNoise part exists
    for (a thruster-less buoy/glider would otherwise declare Q = 0)."""
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.1, 0.1)))
    c.add(ProcessNoise("pn", force_noise_sigma=force_sigma,
                       torque_noise_sigma=torque_sigma))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))
    return w, c


def test_process_noise_part_inert_by_default():
    """Both σ=0 ⇒ the part contributes nothing: zero wrench, no auto-Q."""
    w, _ = _drifter()
    ekf = EKF(w)
    rt = TargetNumpy(ekf)
    if ekf.sys.L_fn is not None:
        L = np.asarray(ekf.sys.L_fn(rt.x, ekf.sys.u_defaults, 0.02, 0.0))
        assert np.allclose(L @ ekf.sys.Sigma @ L.T, 0.0)


def test_process_noise_part_builds_Q_without_actuators():
    """Force and torque channels build the velocity / angular-velocity Q
    blocks — same math as Thruster noise, on a craft with no actuator."""
    sigma_f, sigma_t, dt, m = 0.5, 0.05, 0.02, 2.0
    w, _ = _drifter(force_sigma=sigma_f, torque_sigma=sigma_t)
    ekf = EKF(w)
    rt = TargetNumpy(ekf)
    L = np.asarray(ekf.sys.L_fn(rt.x, ekf.sys.u_defaults, dt, 0.0))
    Q = L @ ekf.sys.Sigma @ L.T
    vel = ekf.spec.slot("c.velocity")
    qv = np.diag(Q)[vel.tangent_offset:vel.tangent_offset + 3]
    np.testing.assert_allclose(qv, (dt / m) ** 2 * sigma_f ** 2, rtol=1e-6)
    ang = ekf.spec.slot("c.angular_velocity")
    qw = np.diag(Q)[ang.tangent_offset:ang.tangent_offset + 3]
    np.testing.assert_allclose(qw, (dt / 0.1) ** 2 * sigma_t ** 2, rtol=1e-6)


def test_axis_resolved_wrench_noise_builds_anisotropic_auto_Q():
    from manta.parts import WrenchProcessNoise

    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.2, 0.4)))
    c.add(WrenchProcessNoise(
        "pn",
        force_noise_sigma=(0.2, 0.4, 0.6),
        torque_noise_sigma=(0.01, 0.04, 0.12),
    ))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, position=(0, 0, 0))
    ekf = EKF(w)
    runtime = TargetNumpy(ekf)
    dt = 0.02
    L = np.asarray(ekf.sys.L_fn(runtime.x, ekf.sys.u_defaults, dt, 0.0))
    Q = L @ ekf.sys.Sigma @ L.T

    velocity = ekf.spec.slot("c.velocity")
    velocity_q = np.diag(Q)[
        velocity.tangent_offset:velocity.tangent_offset + 3
    ]
    np.testing.assert_allclose(
        velocity_q,
        (dt / 2.0) ** 2 * np.square((0.2, 0.4, 0.6)),
        rtol=1e-6,
    )
    angular = ekf.spec.slot("c.angular_velocity")
    angular_q = np.diag(Q)[angular.tangent_offset:angular.tangent_offset + 3]
    np.testing.assert_allclose(
        angular_q,
        np.square(dt * np.asarray((0.01, 0.04, 0.12)) / (0.1, 0.2, 0.4)),
        rtol=1e-6,
    )


def test_correlated_wrench_covariance_survives_auto_q_assembly():
    from manta.parts import WrenchProcessNoise

    covariance = np.zeros((6, 6))
    covariance[:2, :2] = ((0.16, 0.06), (0.06, 0.09))
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.2, 0.4)))
    c.add(WrenchProcessNoise("pn", wrench_noise_covariance=covariance))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    world = World().add_field(GravityField(g=(0, 0, 0)))
    world.add_craft(c)
    ekf = EKF(world)
    runtime = TargetNumpy(ekf)
    dt = 0.02
    L = np.asarray(ekf.sys.L_fn(runtime.x, ekf.sys.u_defaults, dt, 0.0))
    Q = L @ ekf.sys.Sigma @ L.T
    velocity = ekf.spec.slot("c.velocity")
    sl = slice(velocity.tangent_offset, velocity.tangent_offset + 3)
    np.testing.assert_allclose(
        Q[sl, sl], (dt / 2.0) ** 2 * covariance[:3, :3], rtol=1e-6,
        atol=1e-14,
    )


def test_correlated_wrench_covariance_contracts():
    from manta.parts import WrenchProcessNoise

    covariance = np.eye(6)
    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        WrenchProcessNoise(
            "pn", force_noise_sigma=(1.0, 0.0, 0.0),
            wrench_noise_covariance=covariance,
        )
    asymmetric = covariance.copy()
    asymmetric[0, 1] = 0.5
    with np.testing.assert_raises_regex(ValueError, "symmetric"):
        WrenchProcessNoise("pn", wrench_noise_covariance=asymmetric)
    indefinite = covariance.copy()
    indefinite[0, 0] = -1.0
    with np.testing.assert_raises_regex(ValueError, "positive semidefinite"):
        WrenchProcessNoise("pn", wrench_noise_covariance=indefinite)


def test_process_noise_part_buffets_truth():
    """A NoiseDriver samples the channel: the drifting craft picks up
    motion a clean run doesn't have."""
    w, _ = _drifter(force_sigma=2.0)
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=7))
    for _ in range(50):
        sim.step(0.02)
    assert np.linalg.norm(np.asarray(sim.state["c"]["velocity"]).ravel()) > 0

    w0, _ = _drifter(force_sigma=0.0)
    clean = TargetNumpy(Sim(w0))
    clean.attach_driver(NoiseDriver(seed=7))
    for _ in range(50):
        clean.step(0.02)
    assert np.allclose(np.asarray(clean.state["c"]["velocity"]), 0.0)


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
    ekf.reset_from_model_record(
        {"c": c.initial_state(position=(0, 0, 10))},
        P=np.eye(ekf.spec.tangent_dim))
    assert ekf.Q is None                      # no hand-tuned process noise

    lqr = TargetNumpy(LQR(
        w, x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, regulate=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))

    for _ in range(500):
        u = lqr.control(ekf.state_dict())
        sim.step(dt, u=u)
        ekf.update("c.gps.position", sim.reading("c.gps.position"), u=u)
        ekf.predict(dt, u=u)

    final = np.asarray(sim.state["c"]["position"]).ravel()
    est = ekf.state_dict()["c"]["position"].ravel()
    assert np.linalg.norm(final - target) < 0.5   # regulated despite jitter
    assert np.linalg.norm(final - est) < 0.2       # filter stays healthy
