"""End-to-end UKF: the unscented twin of the EKF on a manta Craft.

The UKF carries the same `x`/`P` and the same additive `Q`/`R` as the EKF
and emits the same-shape Module — so `TargetNumpy(UKF(w))` is the identical
`predict`/`update` filter surface. These tests pin down three things:

  * on a linear/affine model the unscented transform is exact, so the UKF
    and the EKF agree to numerical noise (`test_ukf_*_matches_ekf*`);
  * the sigma-point predict and the baked Joseph-free update stay
    manifold-correct on SO(3) (quaternion unit-norm, P stays SPD);
  * the chosen sensor set carves the state exactly as the EKF's does.

Attitude convergence is driven through a *baked* magnetometer update (not
the numpy-only custom-`h(x)` escape hatch, which is EKF-linearized) so the
genuine unscented update kernel is exercised.
"""

import math

import numpy as np
import pytest

from manta import EKF, UKF, Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField, MagField
from manta.parts import IMU, Magnetometer, Mass, PositionSensor


# ---------------------------------------------------------------------------
# Module shape / view selection
# ---------------------------------------------------------------------------

def test_ukf_emits_filter_view():
    """A UKF lowers to the same HELD predict/update view as an EKF, with the
    same tangent-covariance `P`."""
    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name="shape").add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c)

    ukf = TargetNumpy(UKF(w))
    assert hasattr(ukf, "predict") and hasattr(ukf, "update")
    assert ukf.P.shape == (ukf.spec.tangent_dim, ukf.spec.tangent_dim)
    assert UKF(w).module().name == "shape_ukf"


# ---------------------------------------------------------------------------
# Linear-limit agreement with the EKF
# ---------------------------------------------------------------------------

def _free_fall_world(name):
    c = Craft("po")
    c.add(Mass("body", mass=1.0))
    w = World(name=name).add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 100.0))
    return w, c


def test_ukf_predict_alone_matches_tick():
    """Predict with no measurements reproduces the craft's tick: the
    unscented mean of affine free-fall dynamics is `f(x)` itself."""
    tw, _ = _free_fall_world("truth")
    truth_sim = TargetNumpy(Sim(tw))

    uw = World(name="ukf").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    uc = Craft("po")
    uc.add(Mass("body", mass=1.0))
    uw.add_craft(uc)
    ukf = TargetNumpy(UKF(uw))
    ukf.reset(state={"po": {"position": truth_sim.state["po"]["position"]}})

    for _ in range(100):
        truth_sim.step(0.01)
        ukf.predict(dt=0.01)

    assert np.isclose(truth_sim.state["po"]["position"][2], 95.095, atol=1e-5)
    assert np.isclose(ukf.state_dict()["po"]["position"][2], 95.095, atol=1e-4)


def test_ukf_matches_ekf_on_linear_problem():
    """Free fall + a 3-axis position sensor, with the manifold blocks
    quiesced: the excited position/velocity subspace is genuinely linear,
    so the unscented transform is exact there and the UKF and EKF run
    identical recursions — state and covariance agree to numerical noise
    at the default (finite) sigma spread.

    The manifold blocks (orientation tangent 3:6, angular velocity 9:12 —
    the tick couples them into the closure, so `track` cannot carve them
    off) get 1e-8 prior variance: at σ=1 those blocks are NOT linear —
    sigma points spread ~√3 rad across SO(3), where the UT's saturating
    covariance and the EKF's unbounded Jacobian push genuinely differ.
    Exact parity is only an honest claim while the curved directions stay
    quiescent."""
    rng = np.random.default_rng(3)

    def world(name):
        c = Craft("d")
        c.add(Mass("body", mass=1.0))
        c.add(PositionSensor("gps", position_noise_sigma=0.05))
        w = World(name=name).add_field(GravityField(g=(0.0, 0.0, -9.81)))
        w.add_craft(c, position=(0.0, 0.0, 100.0))
        return w

    ukf = TargetNumpy(UKF(world("uk")))
    ekf = TargetNumpy(EKF(world("ek")))
    for f in (ukf, ekf):
        P0 = np.eye(f.spec.tangent_dim)
        P0[3:6, 3:6] = np.eye(3) * 1e-8      # orientation tangent
        P0[9:12, 9:12] = np.eye(3) * 1e-8    # angular velocity
        f.reset(state={"d": {"position": np.zeros(3)}}, P=P0)

    truth = TargetNumpy(Sim(world("tr")))
    for _ in range(500):
        truth.step(0.01)
        z = truth.state["d"]["position"] + rng.normal(0.0, 0.1, 3)
        for f in (ukf, ekf):
            f.predict(0.01)
            f.update("gps.position", z)

    du = ukf.state_dict()["d"]["position"]
    de = ekf.state_dict()["d"]["position"]
    assert np.allclose(du, de, atol=1e-7)
    assert np.allclose(ukf.P, ekf.P, atol=1e-7)


def test_ukf_position_sensor_pulls_estimate_toward_truth():
    """Mismatched prior (truth z=100, belief z=0): noisy z-position updates
    drag the estimate up to track truth."""
    rng = np.random.default_rng(7)

    def world(name):
        c = Craft("oracle")
        c.add(Mass("body", mass=1.0))
        c.add(PositionSensor("gps", position_noise_sigma=0.1))
        w = World(name=name).add_field(GravityField(g=(0.0, 0.0, -9.81)))
        w.add_craft(c, position=(0.0, 0.0, 100.0))
        return w

    truth = TargetNumpy(Sim(world("tr")))
    ukf = TargetNumpy(UKF(world("uk")))
    ukf.reset(state={"oracle": {"position": np.zeros(3)}},
              P=np.eye(ukf.spec.tangent_dim) * 1.0)

    Q = np.eye(ukf.spec.tangent_dim) * 1e-6
    for _ in range(500):
        truth.step(0.01)
        ukf.predict(dt=0.01, Q=Q)
        z = truth.state["oracle"]["position"] + rng.normal(0.0, 0.1, 3)
        ukf.update("gps.position", z)

    truth_pos = truth.state["oracle"]["position"]
    est_pos = ukf.state_dict()["oracle"]["position"]
    assert np.isclose(truth_pos[2], 100.0 - 0.5 * 9.81 * 25.0, atol=1e-3)
    assert np.allclose(est_pos, truth_pos, atol=0.5), \
        f"UKF est={est_pos} far from truth={truth_pos}"


# ---------------------------------------------------------------------------
# Manifold correctness (SO(3)) through a baked unscented update
# ---------------------------------------------------------------------------

def _attitude_world(name):
    c = Craft("att")
    c.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))   # spherical I
    c.add(IMU("imu", gyro_noise_sigma=0.01, accel_noise_sigma=0.05))
    c.add(Magnetometer("mag", B_noise_sigma=1e-7))
    w = World(name=name).add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_field(MagField().add_uniform((2e-5, 0.0, -4e-5)))
    w.add_craft(c)
    return w, c


def test_ukf_attitude_converges_via_magnetometer():
    """A tilted prior is pulled onto truth by baked magnetometer updates —
    the genuine unscented update on the orientation manifold. The estimate
    converges, the quaternion stays unit-norm, and P stays SPD."""
    rng = np.random.default_rng(0)
    truth = TargetNumpy(Sim(_attitude_world("tr")[0]))
    ukf_t = UKF(_attitude_world("uk")[0])
    ukf = TargetNumpy(ukf_t)
    mag = next(s for s in ukf_t.sys.sensors if s.endswith("mag.B"))

    half = math.radians(25.0) / 2.0           # 25° pitch error in the prior
    wrong_q = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
    n = ukf.spec.tangent_dim
    P0 = np.eye(n) * 1e-4
    P0[3:6, 3:6] = np.eye(3) * 0.5
    ukf.reset(state={"att": {"orientation": wrong_q}}, P=P0)

    for _ in range(400):
        truth.step(0.01)
        ukf.predict(0.01)
        z = truth.reading(mag) + rng.normal(0.0, 1e-6, 3)
        ukf.update(mag, z)

    est_q = ukf.state_dict()["att"]["orientation"]
    truth_q = truth.state["att"]["orientation"]
    inner = abs(float(np.dot(est_q, truth_q)))
    assert inner > 0.999, f"orientations diverged: inner={inner}"
    assert np.isclose(np.linalg.norm(est_q), 1.0, atol=1e-10)
    assert np.linalg.eigvalsh(ukf.P).min() > 0.0, "P lost positive-definiteness"


def test_ukf_attitude_matches_ekf():
    """On the same attitude problem the UKF and EKF land on (essentially)
    the same estimate. Agreement here is close but not bitwise: at the
    default √3·σ sigma spread the unscented moments genuinely sample the
    manifold curvature the EKF linearizes away, so the two filters differ
    at the curvature scale — a few 1e-5 in the quaternion — not to
    machine epsilon (that regime only exists in the degenerate small-α
    limit where the UKF is a finite-difference EKF)."""
    def run(Filt, name):
        rng = np.random.default_rng(0)
        truth = TargetNumpy(Sim(_attitude_world(name + "_tr")[0]))
        ft = Filt(_attitude_world(name + "_f")[0])
        f = TargetNumpy(ft)
        mag = next(s for s in ft.sys.sensors if s.endswith("mag.B"))
        half = math.radians(25.0) / 2.0
        wrong_q = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
        n = f.spec.tangent_dim
        P0 = np.eye(n) * 1e-4
        P0[3:6, 3:6] = np.eye(3) * 0.5
        f.reset(state={"att": {"orientation": wrong_q}}, P=P0)
        for _ in range(400):
            truth.step(0.01)
            f.predict(0.01)
            f.update(mag, truth.reading(mag) + rng.normal(0.0, 1e-6, 3))
        return f.state_dict()["att"]["orientation"], f.P

    q_u, P_u = run(UKF, "u")
    q_e, P_e = run(EKF, "e")
    assert abs(abs(float(np.dot(q_u, q_e))) - 1.0) < 1e-4
    assert np.allclose(P_u, P_e, atol=1e-5)


# ---------------------------------------------------------------------------
# Numerical robustness — the jitter backstop and the weight guards
# ---------------------------------------------------------------------------

def test_ukf_survives_marginally_indefinite_P():
    """A P that roundoff has pushed marginally indefinite (one tiny
    negative eigenvalue) must not NaN the filter: the sigma factorization
    jitters and floors its pivots, so predict/update stay finite and P
    recovers PSD. This is the exact cliff the unguarded Cholesky fell off:
    sqrt(-1e-12) = NaN poisons every later column, then x and P forever."""
    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name="jit").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)

    ukf = TargetNumpy(UKF(w))
    n = ukf.spec.tangent_dim
    P_bad = np.eye(n) * 1e-2
    P_bad[0, 0] = -1e-12                     # marginally indefinite
    ukf.reset(P=P_bad)
    ukf.predict(0.01)
    ukf.update("gps.position", np.zeros(3))
    assert np.all(np.isfinite(ukf.P)), "jitter backstop failed: P has NaN"
    assert np.all(np.isfinite(ukf.state_dict()["d"]["position"]))
    assert np.linalg.eigvalsh(ukf.P).min() > 0.0


def test_ukf_degenerate_weights_raise():
    """`n + lam <= 0` makes the sigma spread imaginary — refused loudly at
    construction, not NaN'd at runtime."""
    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name="deg").add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c)
    with pytest.raises(ValueError, match="imaginary"):
        UKF(w, alpha=1e-3, kappa=-100.0)


def test_ukf_explicit_negative_weight_tuning_warns():
    """An explicit small-alpha tuning (the old degenerate default) still
    builds, but warns that the covariance sums lose their PSD guarantee."""
    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name="warn").add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c)
    with pytest.warns(RuntimeWarning, match="negative central covariance"):
        UKF(w, alpha=1e-3)


# ---------------------------------------------------------------------------
# State carving (track / sensors) — same plumbing as the EKF
# ---------------------------------------------------------------------------

def test_ukf_sensors_subset_restricts_update_kernels():
    """`sensors=[...]` keeps only the chosen measurement kernels, exactly as
    for the EKF."""
    c = Craft("d")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(IMU("g", gyro_noise_sigma=0.01))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name="subset").add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 5))

    ukf_t = UKF(w, sensors=["g.gyro", "gps.position"])
    assert set(ukf_t.sys.sensors) == {"d.g.gyro", "d.gps.position"}
    ukf = TargetNumpy(ukf_t)
    ukf.predict(0.01)
    ukf.update("gps.position", np.array([0.0, 0.0, 5.0]))
    with pytest.raises(Exception):
        ukf.update("g.accel", np.zeros(3))   # not in the chosen set


def test_ukf_block_count_matches_independent_crafts():
    """Independent crafts → block-diagonal predict, same as the EKF's."""
    def craft(n):
        c = Craft(n)
        c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        c.add(PositionSensor("gps", position_noise_sigma=0.05))
        return c
    w = World(name="two").add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(craft("a"), position=(0, 0, 5))
    w.add_craft(craft("b"), position=(3, 0, 5))
    assert UKF(w).n_blocks == 2
