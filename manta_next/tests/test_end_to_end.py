"""End-to-end demo + tests for M9.

Exercises the full design pipeline:

  1.  Build a Craft with Mass + Thruster + IMU + PositionSensor
  2.  Step the sim with a commanded thrust profile via CompiledWorld
  3.  Sample noisy sensor outputs (gyro, position)
  4.  Run an ESKF in parallel using:
        - the same per-tick commanded thrust (predict step),
        - the noisy gyro reading (update step),
        - the noisy position reading (update step).
  5.  Verify the EKF tracks ground truth within sensor-noise bounds.

The demo proves three things end-to-end:

  * Input declarations flow through World.step and EKF.predict identically.
  * Output declarations produce sensor-style readings that route into
    EKF.update via measurement_slot helpers.
  * The Joseph-form ESKF actually converges under realistic noise.
"""

import numpy as np
import pytest

from manta_next import Craft, World
from manta_next.ekf import EKF, measurement_slot
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


# ---------------------------------------------------------------------------
# EKF predict() honors per-tick inputs
# ---------------------------------------------------------------------------

def test_ekf_predict_with_per_tick_input():
    """The same thruster command applied to sim and EKF predict must drive
    them to the same state (no noise, no measurements)."""
    g = (0.0, 0.0, -9.81)
    m = 1.0
    c = Craft("hover_ekf")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster.linear("t", max_thrust=1.0))

    # Sim
    tick = c.compile_tick(gravity_anchor=g)
    sim_state = c.initial_state()
    # EKF
    ekf = EKF(c, gravity_anchor=g)

    # Thrust profile: zero for 0.2s, then m·g (hover) for 0.5s, then 2·m·g.
    dt = 0.005
    n = 140
    for i in range(n):
        t = i * dt
        if t < 0.2:
            thrust = 0.0
        elif t < 0.7:
            thrust = m * 9.81
        else:
            thrust = 2.0 * m * 9.81

        # Sim step.
        sim_state["t.throttle"] = thrust
        out = tick(dt=dt, **sim_state)
        sim_state = {**sim_state, **out}

        # EKF predict step.
        ekf.predict(dt=dt, u={"t.throttle": thrust})

    # No measurements, no noise — should agree to round-off.
    est = ekf.state_dict()
    np.testing.assert_allclose(est["position"].ravel(),
                               np.array(sim_state["position"]).ravel(),
                               atol=1e-9)
    np.testing.assert_allclose(est["velocity"].ravel(),
                               np.array(sim_state["velocity"]).ravel(),
                               atol=1e-9)


def test_ekf_predict_input_default_fallback():
    """`u` argument is optional; missing inputs use the part default
    captured at construction time."""
    g = (0.0, 0.0, -9.81)
    c = Craft("default_thrust")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    # Thruster default throttle=0 — craft should free-fall.
    c.add(Thruster.linear("t", max_thrust=1.0))
    ekf = EKF(c, gravity_anchor=g)
    for _ in range(200):
        ekf.predict(dt=0.005)

    est = ekf.state_dict()
    # 200 * 0.005 = 1.0s. z ≈ -½·g·t² = -4.905.
    assert np.isclose(est["position"][2], -4.905, atol=1e-2)


def test_ekf_predict_unknown_input_raises():
    c = Craft("any")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Thruster.linear("t", max_thrust=1.0))
    ekf = EKF(c, gravity_anchor=(0.0, 0.0, 0.0))
    with pytest.raises(KeyError, match="unknown input"):
        ekf.predict(dt=0.01, u={"nope.bad": 1.0})


# ---------------------------------------------------------------------------
# Full end-to-end: commanded hover + noisy gyro + noisy position → ESKF
# ---------------------------------------------------------------------------

def test_hover_with_eskf_tracks_ground_truth():
    """Drone-style craft hovering against gravity. Noisy IMU gyro and
    noisy position sensor feed an ESKF. After convergence the estimate
    must agree with sim ground truth within a few sigma of measurement
    noise."""
    rng = np.random.default_rng(seed=42)
    g_world = (0.0, 0.0, -9.81)
    m       = 1.5

    # Build the craft once; sim and EKF share the same instance.
    c = Craft("drone")
    c.add(Mass("body", mass=m, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster.linear("t", max_thrust=1.0))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))

    # Sim path: use World/CompiledWorld so the demo exercises the public
    # surface used by user code.
    w = World().add_uniform_gravity(g_world)
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    cw = w.compile()
    sim = cw.initial_state()
    sim["drone"]["t.throttle"] = m * 9.81   # hover

    # EKF path. Initial estimate offset from truth so we can watch it pull
    # in via measurement updates.
    ekf = EKF(c, gravity_anchor=g_world)
    init = c.initial_state(position=(0.0, 0.0, 4.0),
                           velocity=(0.5, 0.0, 0.0))
    P0   = np.eye(ekf.spec.tangent_dim) * 1e-1
    ekf.reset(state=init, P=P0)

    # Noise model.
    sigma_gyro = 0.01     # rad/s standard deviation
    sigma_pos  = 0.05     # m
    R_gyro = (sigma_gyro ** 2) * np.eye(3)
    R_pos  = (sigma_pos  ** 2) * np.eye(3)

    # Process noise tuned to give a consistent filter (NEES mean ≈ 11
    # over a 20-seed MC, ~84% inside the χ²₁₂ 95% band) for this
    # deterministic-model hover.
    Q = np.diag([1e-8] * 3 +    # position
                [1e-8] * 3 +    # orientation tangent
                [1e-6] * 3 +    # velocity
                [1e-7] * 3)     # angular velocity

    h_pos  = measurement_slot(ekf.spec, "position")
    h_gyro = measurement_slot(ekf.spec, "angular_velocity")

    dt = 0.005
    n_steps = 600                # 3 seconds simulated
    thrust = m * 9.81
    final_pos_err   = None
    final_vel_err   = None

    for i in range(n_steps):
        # Sim step.
        sim["drone"]["t.throttle"] = thrust
        sim = cw.step(sim, dt=dt)

        # Read noisy sensor outputs from the sim tick result.
        gyro_clean = np.array(sim["drone"]["g.gyro"]).ravel()
        pos_clean  = np.array(sim["drone"]["gps.position"]).ravel()
        gyro_meas = gyro_clean + rng.normal(0.0, sigma_gyro, 3)
        pos_meas  = pos_clean  + rng.normal(0.0, sigma_pos,  3)

        # EKF predict with the SAME commanded thrust as the sim.
        ekf.predict(dt=dt, u={"t.throttle": thrust}, Q=Q)

        # Gyro update every tick.
        ekf.update(h_gyro, gyro_meas, R_gyro)
        # Position update at 50 Hz (every 4 ticks @ 200 Hz tick rate).
        if i % 4 == 0:
            ekf.update(h_pos, pos_meas, R_pos)

    truth = sim["drone"]
    est   = ekf.state_dict()
    final_pos_err = np.linalg.norm(
        np.array(truth["position"]).ravel() - est["position"].ravel())
    final_vel_err = np.linalg.norm(
        np.array(truth["velocity"]).ravel() - est["velocity"].ravel())

    # Position must converge to within a couple sigma_pos.
    assert final_pos_err < 4.0 * sigma_pos, (
        f"position error {final_pos_err:.3f} > 4σ ({4.0 * sigma_pos:.3f})")
    # Velocity has no direct measurement; it's reconstructed by predict +
    # position-update innovation. A loose bound is enough to prove the
    # filter is healthy.
    assert final_vel_err < 0.25, f"velocity error {final_vel_err:.3f}"

    # Covariance must have shrunk meaningfully from the diffuse initial
    # P0 (1e-1·I on the 12-dim tangent → trace 1.2).
    P_final = ekf.P
    assert np.trace(P_final) < 0.5, (
        f"trace(P) didn't shrink: {np.trace(P_final):.3f}")


# ---------------------------------------------------------------------------
# Filter consistency: NEES (Normalized Estimation Error Squared)
# ---------------------------------------------------------------------------

def test_eskf_nees_consistency_over_seeds():
    """5-seed Monte Carlo: per-tick NEES = e^T·P^-1·e (tangent-space) must
    stay inside the χ²₁₂ 95% band [4.40, 23.34] most of the time. A filter
    that drifts outside is either overconfident (P too small → unsafe) or
    underconfident (P too large → throwing away information). Standard
    diagnostic per Bar-Shalom et al.

    This is faster (5 seeds, post-convergence samples only) than the
    rigorous 20-seed MC in the verification log but covers the same
    architectural ground."""
    import casadi as ca

    g_world = (0.0, 0.0, -9.81)
    m = 1.5
    sigma_gyro, sigma_pos = 0.01, 0.05
    R_gyro = (sigma_gyro**2) * np.eye(3)
    R_pos  = (sigma_pos **2) * np.eye(3)
    Q = np.diag([1e-8] * 3 + [1e-8] * 3 + [1e-6] * 3 + [1e-7] * 3)

    def make_craft():
        c = Craft("drone")
        c.add(Mass("body", mass=m, moi=(0.05, 0.05, 0.08)))
        c.add(Thruster.linear("t", max_thrust=1.0))
        c.add(IMU("g"))
        c.add(PositionSensor("gps"))
        return c

    dt = 0.005
    n  = 600
    thrust = m * 9.81
    nees_samples: list[float] = []

    bm_fn = None
    for seed in range(5):
        rng = np.random.default_rng(seed=seed)
        c = make_craft()
        w = World().add_uniform_gravity(g_world)
        w.add_craft(c, position=(0, 0, 5))
        cw = w.compile()
        sim = cw.initial_state()

        ekf = EKF(c, gravity_anchor=g_world)
        init = c.initial_state(position=(0, 0, 4), velocity=(0.5, 0, 0))
        ekf.reset(state=init, P=np.eye(ekf.spec.tangent_dim) * 1e-1)

        h_pos  = measurement_slot(ekf.spec, "position")
        h_gyro = measurement_slot(ekf.spec, "angular_velocity")

        if bm_fn is None:
            xa = ca.MX.sym("a", ekf.spec.ambient_dim, 1)
            xb = ca.MX.sym("b", ekf.spec.ambient_dim, 1)
            bm_fn = ca.Function("bm", [xa, xb],
                                 [ekf.spec.boxminus_sym(xa, xb)])

        for i in range(n):
            sim["drone"]["t.throttle"] = thrust
            sim = cw.step(sim, dt=dt)
            gyro_meas = (np.array(sim["drone"]["g.gyro"]).ravel()
                         + rng.normal(0.0, sigma_gyro, 3))
            pos_meas  = (np.array(sim["drone"]["gps.position"]).ravel()
                         + rng.normal(0.0, sigma_pos,  3))
            ekf.predict(dt=dt, u={"t.throttle": thrust}, Q=Q)
            ekf.update(h_gyro, gyro_meas, R_gyro)
            if i % 4 == 0:
                ekf.update(h_pos, pos_meas, R_pos)

            # Sample NEES post-convergence only (after t=0.5s).
            if i > 100 and i % 50 == 0:
                truth_dict = {
                    k: np.atleast_1d(np.asarray(v, dtype=float))
                    for k, v in sim["drone"].items() if k in ekf.spec}
                x_truth = ekf.spec.pack(truth_dict)
                err_tan = np.asarray(bm_fn(x_truth, ekf.x)).ravel()
                try:
                    nees = float(err_tan @ np.linalg.solve(ekf.P, err_tan))
                    nees_samples.append(nees)
                except np.linalg.LinAlgError:
                    pass

    nees_arr = np.array(nees_samples)
    assert nees_arr.size >= 25, (
        f"too few NEES samples: {nees_arr.size}")

    # χ²₁₂ 95% bound: [4.40, 23.34]. We allow a generous threshold here —
    # 5 seeds is a small sample and the Q value's the dominant tuning knob.
    # The point is to catch architectural regressions (NEES drifting to
    # 100+ or 0.1) not to nail textbook consistency.
    mean_nees   = nees_arr.mean()
    inside_band = ((nees_arr >= 4.40) & (nees_arr <= 23.34)).mean()
    assert 4.0 < mean_nees < 23.0, (
        f"NEES mean {mean_nees:.2f} outside healthy range [4, 23]")
    assert inside_band > 0.5, (
        f"only {inside_band*100:.0f}% of NEES samples inside χ²₁₂ "
        f"95% band — filter tuning needs work")
