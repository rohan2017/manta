"""End-to-end EKF: predict + update on a manta Craft with synthetic
oracle sensors. Validates the autodiff Jacobian path and the standard
KF math.
"""

import math

import numpy as np

from manta import Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField
from manta.estimation import EKF, measurement_component, measurement_slot
from manta.parts import Mass
from manta import state_spec_from_craft


# ---------------------------------------------------------------------------
# StateSpec layout sanity
# ---------------------------------------------------------------------------

def test_state_spec_rigid_body_layout():
    c = Craft("body_only")
    c.add(Mass("body", mass=1.0))
    spec = state_spec_from_craft(c)

    assert spec.ambient_dim == 13   # 3 + 4 + 3 + 3
    assert spec.tangent_dim == 12   # SO(3) collapses 4→3
    names = [s.name for s in spec.slots]
    assert names == ["position", "orientation", "velocity", "angular_velocity"]
    assert spec.slot("orientation").manifold.kind == "quat"


def test_state_spec_with_parts():
    from manta.parts import RevoluteJoint

    c = Craft("with_state")
    c.add(Mass("body", mass=1.0))
    wheel = RevoluteJoint("wheel", mode="passive")
    wheel.add(Mass("wheel_disk", mass=0.05, moi=(0.001, 0.001, 0.005)))
    c.add(wheel)
    motor = RevoluteJoint("motor", mode="saturating", stall_torque=1e6)
    motor.add(Mass("motor_rotor", mass=0.05, moi=(0.001, 0.001, 0.005)))
    c.add(motor)

    spec = state_spec_from_craft(c)
    # Each RevoluteJoint contributes 2 state slots (angle, rate) → 13 + 2 + 2 = 17.
    assert spec.ambient_dim == 13 + 2 + 2
    names = [s.name for s in spec.slots]
    assert "wheel.angle" in names
    assert "wheel.rate"  in names
    assert "motor.angle" in names
    assert "motor.rate"  in names


def test_state_spec_pack_unpack_roundtrip():
    c = Craft("roundtrip")
    c.add(Mass("body", mass=1.0))
    spec = state_spec_from_craft(c)

    state = {
        "position":         np.array([1.0, 2.0, 3.0]),
        "orientation":      np.array([1.0, 0.0, 0.0, 0.0]),
        "velocity":         np.array([0.5, -0.3, 0.7]),
        "angular_velocity": np.array([0.1, 0.2, 0.3]),
    }
    flat = spec.pack(state)
    assert flat.shape == (13,)
    back = spec.unpack(flat)
    for k in state:
        assert np.allclose(back[k], state[k])


# ---------------------------------------------------------------------------
# EKF: noisy free-fall with oracle position sensor
# ---------------------------------------------------------------------------

def test_ekf_predict_alone_matches_tick():
    """With no measurements, EKF.predict() should give the same state as
    running the Craft's tick directly."""
    c = Craft("predict_only")
    c.add(Mass("body", mass=1.0))

    # Truth sim: an identical craft in its own World.
    tc = Craft("predict_only")
    tc.add(Mass("body", mass=1.0))
    tw = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    tw.add_craft(tc, position=(0.0, 0.0, 100.0))
    truth_sim = TargetNumpy(Sim(tw))

    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    _ekf_world.add_craft(c)
    ekf = TargetNumpy(EKF(_ekf_world))

    ekf.reset(state={"predict_only":
                     {"position": truth_sim.state["predict_only"]["position"]}})

    dt = 0.01
    for _ in range(100):
        # Direct tick
        truth_sim.step(dt)
        # EKF predict (with no process noise)
        ekf.predict(dt=dt)

    # Both should now be at z = 100 − 0.5·9.81·1 = 95.095.
    assert np.isclose(truth_sim.state["predict_only"]["position"][2],  95.095, atol=1e-5)
    assert np.isclose(ekf.state_dict()["predict_only"]["position"][2],
                      95.095, atol=1e-5)


def test_ekf_position_sensor_pulls_estimate_toward_truth():
    """Mismatched initial belief: truth at z=100, EKF prior at z=0.
    Noisy oracle z-position measurements should pull the estimate up."""
    rng = np.random.default_rng(7)
    c = Craft("oracle_demo")
    c.add(Mass("body", mass=1.0))

    tc = Craft("oracle_demo")
    tc.add(Mass("body", mass=1.0))
    tw = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    tw.add_craft(tc, position=(0.0, 0.0, 100.0))
    truth_sim = TargetNumpy(Sim(tw))

    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    _ekf_world.add_craft(c)
    ekf = TargetNumpy(EKF(_ekf_world))

    # Truth at z=100, EKF prior at z=0 — but covariance reflects the
    # uncertainty.
    P0 = np.eye(ekf.spec.tangent_dim) * 1.0   # broad prior, tangent-dim
    ekf.reset(state={"oracle_demo": {"position": np.zeros(3)}}, P=P0)

    h_z = measurement_component(ekf.spec, "position", component=2)
    R = np.array([[0.01]])          # 10 cm sensor stddev squared

    Q = np.eye(ekf.spec.tangent_dim) * 1e-6   # tiny process noise

    dt = 0.01
    for _ in range(500):  # 5 seconds
        # Truth integration.
        truth_sim.step(dt)
        # EKF predict + noisy measurement update.
        ekf.predict(dt=dt, Q=Q)
        z = truth_sim.state["oracle_demo"]["position"][2] + rng.normal(0.0, 0.1)
        ekf.update(h_z, z=np.array([z]), R=R)

    truth_z = truth_sim.state["oracle_demo"]["position"][2]
    est_z   = ekf.state_dict()["oracle_demo"]["position"][2]
    # Truth has fallen by ½·g·t² = 122.625 → z = -22.625.
    assert np.isclose(truth_z, 100.0 - 0.5 * 9.81 * 25.0, atol=1e-3)
    # Estimate should now track truth tightly.
    assert abs(est_z - truth_z) < 0.5, \
        f"EKF z={est_z:.3f} far from truth={truth_z:.3f}"


def test_ekf_full_position_observation_drives_convergence():
    """Three-component position measurement converges position estimate
    on all axes. Zero gravity here so the EKF's stationary-truth model
    is consistent — gravity-vs-stationary tension is the subject of a
    separate test below."""
    rng = np.random.default_rng(11)
    c = Craft("full_pos")
    c.add(Mass("body", mass=1.0))

    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    _ekf_world.add_craft(c)
    ekf = TargetNumpy(EKF(_ekf_world))   # no gravity
    ekf.reset(state={"full_pos": {"position": np.array([10.0, -5.0, 0.0])}},
              P=np.eye(ekf.spec.tangent_dim) * 4.0)

    truth_pos = np.array([0.5, 1.0, 80.0])
    h_p = measurement_slot(ekf.spec, "position")
    R   = np.eye(3) * 0.001                    # ~3 cm sensor

    Q = np.eye(ekf.spec.tangent_dim) * 1e-8
    for _ in range(200):
        ekf.predict(dt=0.01, Q=Q)
        z = truth_pos + rng.normal(0.0, 0.03, size=3)
        ekf.update(h_p, z=z, R=R)

    est_pos = ekf.state_dict()["full_pos"]["position"]
    assert np.allclose(est_pos, truth_pos, atol=0.05), \
        f"est={est_pos}, truth={truth_pos}"

    # Position covariance should have shrunk dramatically from the
    # broad prior.
    P_pos = ekf.P[:3, :3]
    assert np.all(np.diag(P_pos) < 0.01), \
        f"P_pos diagonal too large: {np.diag(P_pos)}"


def test_eskf_attitude_estimation_converges():
    """ESKF demonstrates manifold-aware orientation estimation.

    Setup: a spherical-inertia spinning body (zero gravity, no torque)
    spins about +z at a known rate. Truth orientation rolls out via
    boxplus. EKF has a slightly-wrong prior orientation and receives
    noisy orientation measurements; the manifold-aware update pulls
    the estimate onto the truth without quaternion-norm blowup.
    """
    rng = np.random.default_rng(33)
    c = Craft("attitude_demo")
    c.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))   # spherical I

    # Truth spins at ω = 1 rad/s about +z.
    omega_truth = np.array([0.0, 0.0, 1.0])
    tc = Craft("attitude_demo")
    tc.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    tw = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    tw.add_craft(tc, angular_velocity=tuple(omega_truth))
    truth_sim = TargetNumpy(Sim(tw))

    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    _ekf_world.add_craft(c)
    ekf = TargetNumpy(EKF(_ekf_world))

    # EKF prior: wrong orientation + uncertain.
    half = math.pi / 8.0
    wrong_q = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    n_tan = ekf.spec.tangent_dim
    P0 = np.eye(n_tan) * 1e-4
    # Bigger prior on the orientation tangent slot (offsets 3..6) — we
    # know our prior is wrong by tens of degrees.
    P0[3:6, 3:6] = np.eye(3) * 0.5
    ekf.reset(state={"attitude_demo": {"orientation":      wrong_q,
                                       "angular_velocity": omega_truth}},
              P=P0)

    # Noisy 4-D quaternion measurements of the orientation.
    def h_q(x):
        return x[3:7]
    R = np.eye(4) * 1e-3
    Q = np.eye(n_tan) * 1e-8

    for _ in range(500):
        truth_sim.step(0.01)
        ekf.predict(dt=0.01, Q=Q)
        z = truth_sim.state["attitude_demo"]["orientation"] + rng.normal(0.0, 0.03, size=4)
        ekf.update(h_q, z=z, R=R)

    # Estimate's orientation should now track truth tightly.
    est_q   = ekf.state_dict()["attitude_demo"]["orientation"]
    truth_q = truth_sim.state["attitude_demo"]["orientation"]
    # Inner product: |<est, truth>| ≈ 1 when they agree (modulo cover).
    inner = abs(float(np.dot(est_q, truth_q)))
    assert inner > 0.999, \
        f"orientations diverged: inner={inner}, est={est_q}, truth={truth_q}"

    # Quaternion remains unit-norm.
    assert np.isclose(np.linalg.norm(est_q), 1.0, atol=1e-10)


def test_eskf_state_spec_layout_with_tangent_offsets():
    """Tangent offsets are contiguous and respect SO3 collapse."""
    from manta.parts import RevoluteJoint
    c = Craft("with_state")
    c.add(Mass("body", mass=1.0))
    wheel = RevoluteJoint("wheel", mode="passive")
    wheel.add(Mass("wheel_disk", mass=0.01, moi=(0.0001, 0.0001, 0.0005)))
    c.add(wheel)
    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    _ekf_world.add_craft(c)
    ekf = TargetNumpy(EKF(_ekf_world))
    spec = ekf.spec
    # Total: rigid-body(13/12) + wheel.angle (1/1) + wheel.rate (1/1)
    assert spec.ambient_dim == 13 + 2
    assert spec.tangent_dim == 12 + 2
    # Tangent offsets are contiguous in slot order. World-tick slot
    # names are flat-prefixed with the craft name.
    expected_tan_offsets = {
        "with_state.position":          0,
        "with_state.orientation":       3,
        "with_state.velocity":          6,
        "with_state.angular_velocity":  9,
        "with_state.wheel.angle":      12,
        "with_state.wheel.rate":       13,
    }
    for slot in spec.slots:
        assert slot.tangent_offset == expected_tan_offsets[slot.name]


def test_ekf_jacobian_for_constant_velocity_model_is_identity_plus_dt():
    """For free-fall, ∂(position_new)/∂(velocity) = dt·I (in the tangent
    layout). Tests that the EKF actually computed a meaningful F."""
    c = Craft("J")
    c.add(Mass("body", mass=1.0))
    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    _ekf_world.add_craft(c)
    ekf_t = EKF(_ekf_world)
    ekf = TargetNumpy(ekf_t)

    dt = 0.05
    u_vec = np.zeros(len(ekf_t.sys.input_names))  # craft has no Inputs.
    F = np.asarray(ekf_t.sys.F_fn(ekf.x, u_vec, dt, 0.0))
    # Tangent layout: position[0:3], orientation[3:6], velocity[6:9],
    # angular_velocity[9:12]. F is 12×12.
    assert F.shape == (12, 12)
    pos_wrt_vel = F[0:3, 6:9]
    assert np.allclose(pos_wrt_vel, dt * np.eye(3), atol=1e-9)
    pos_wrt_pos = F[0:3, 0:3]
    assert np.allclose(pos_wrt_pos, np.eye(3), atol=1e-9)
