"""Random-walk bias channels — Noise(kind="rw") end-to-end.

Validates:
  * `Noise(kind="rw")` declarations synthesize a bias state slot AND a
    driver noise input.
  * Inside `update()`, `self.<noise_name>` reads the bias state MX
    (not the per-tick driver).
  * `Craft.initial_state()` seeds the bias slot at zero AND the
    driver slot at zero.
  * `Craft.sample_noise()` returns RW driver samples keyed
    `<part>.<noise_name>_driver`.
  * Driver flows through `bias_next = bias + sqrt(dt) · driver`, so
    bias variance per tick scales as `dt · σ²` (continuous σ/√Hz
    semantics).
  * `StateSpec.from_craft` picks up RW biases as state slots (R3 for
    vec3, R1 for scalar).
  * EKF auto-Q gives Q[bias, bias] = dt · σ² · I (autodiff over the
    sqrt(dt) factor).
  * EKF estimates the bias from a sensor that reads ω + bias + noise.
"""

import numpy as np
import casadi as ca
import pytest

from manta import World, Craft, EKF, TargetNumpy
from manta.fields import GravityField
from manta.parts import Mass
from manta.parts.base import Noise, Output, Part, PartUpdate
from manta.ir.frames import CraftFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench


class BiasedGyro(Part):
    """A minimal gyro sensor with both white noise + RW bias."""
    gyro_noise = Noise(shape="vec3", kind="white", sigma=0.001)
    gyro_bias  = Noise(shape="vec3", kind="rw",    sigma=0.02)
    gyro = Output(shape="vec3")

    def update(self, ctx):
        zero = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={
                "gyro": ctx.angular_velocity + self.gyro_bias + self.gyro_noise,
            },
        )


def _build_world():
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
    c = Craft("drone")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(BiasedGyro("g"))
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    return w, c


# ---------------------------------------------------------------------------
# Declaration plumbing
# ---------------------------------------------------------------------------

def test_state_spec_picks_up_rw_bias_as_state_slot():
    """RW Noise channels add R3 state slots to the spec."""
    from manta.estimation.state_spec import StateSpec
    _, c = _build_world()
    spec = StateSpec.from_craft(c)
    names = [s.name for s in spec.slots]
    assert "g.gyro_bias" in names
    slot = spec.slot("g.gyro_bias")
    assert slot.dim         == 3
    assert slot.tangent_dim == 3
    assert slot.manifold    == "R3"


def test_initial_state_seeds_bias_and_driver_at_zero():
    """Both the bias state and its driver input slot start at zero."""
    _, c = _build_world()
    init = c.initial_state()
    assert "g.gyro_bias"        in init
    assert "g.gyro_bias_driver" in init
    np.testing.assert_allclose(init["g.gyro_bias"], np.zeros(3))
    np.testing.assert_allclose(init["g.gyro_bias_driver"], np.zeros(3))


def test_sample_noise_returns_rw_driver_under_underscore_driver_key():
    """`craft.sample_noise` keys RW drivers as `<part>.<name>_driver`."""
    _, c = _build_world()
    rng = np.random.default_rng(42)
    samples = c.sample_noise(rng)
    assert "g.gyro_noise"        in samples   # the white channel
    assert "g.gyro_bias_driver"  in samples   # the RW driver
    assert "g.gyro_bias"         not in samples  # bias is a STATE
    # Sample magnitude in the right neighborhood (σ=0.02).
    driver = samples["g.gyro_bias_driver"]
    assert driver.shape == (3,)
    # 3·σ would be 0.06 — be generous.
    assert np.max(np.abs(driver)) < 0.5


# ---------------------------------------------------------------------------
# Dynamics: bias evolution matches `bias + sqrt(dt) · driver`
# ---------------------------------------------------------------------------

def test_bias_advances_by_sqrt_dt_times_driver_in_sim():
    """With a deterministic driver, the bias state evolves exactly
    as the documented model `bias_next = bias + sqrt(dt) · driver`."""
    w, c = _build_world()
    cw = TargetNumpy(w.compile())
    dt = 0.01
    state = cw.initial_state()
    # Inject a constant driver each tick.
    drv = np.array([0.5, -0.25, 0.1])
    expected = np.zeros(3)
    for k in range(50):
        state["drone"]["g.gyro_bias_driver"] = drv.copy()
        state = cw.step(state, dt=dt)
        expected = expected + np.sqrt(dt) * drv
    np.testing.assert_allclose(state["drone"]["g.gyro_bias"], expected,
                               atol=1e-10)


# ---------------------------------------------------------------------------
# EKF picks up bias slot + estimates it from gyro readings
# ---------------------------------------------------------------------------

def test_ekf_estimates_rw_bias_from_gyro_readings():
    """A drone at rest (no torques, no thrusts) sees gyro = bias + noise.
    EKF with an explicit gyro update should pull the bias estimate
    toward the truth."""
    rng = np.random.default_rng(7)

    sim_w, sim_c = _build_world()
    est_w, est_c = _build_world()
    cw = TargetNumpy(sim_w.compile())
    ekf = TargetNumpy(EKF(est_w))

    # Truth: a fixed initial bias, no further drift (zero driver).
    initial_bias = np.array([0.05, -0.03, 0.02])
    state = cw.initial_state()
    state["drone"]["g.gyro_bias"] = initial_bias.copy()

    # Give the EKF a broad prior on the bias slot so it's free to
    # learn it from measurements.
    P = np.eye(ekf.spec.tangent_dim) * 1e-4
    # Bias slot is the last 3 tangent dims (R3).
    P[12:15, 12:15] = np.eye(3) * 0.5
    ekf.reset(P=P)

    imu_part = next(p for p in est_c.parts if p.name == "g")

    dt = 0.01
    t = 0.0
    for _ in range(500):  # 5 s
        # Sim noise samples — sim's bias gets driven by its own driver.
        noise = sim_c.sample_noise(rng)
        state["drone"].update(noise)
        state = cw.step(state, t=t, dt=dt)
        gyro_z = np.array(state["drone"]["g.gyro"]).ravel()
        ekf.predict(dt=dt, t=t)
        ekf.update(imu_part, gyro=gyro_z)
        t += dt

    truth_bias = np.array(state["drone"]["g.gyro_bias"])
    est_bias   = np.array(ekf.state_dict()["drone"]["g.gyro_bias"])
    err        = np.linalg.norm(est_bias - truth_bias)
    # The truth drifts a small amount due to σ=0.02 driver over 5s;
    # the estimate should track within a few hundredths of a rad/s.
    assert err < 0.05, (
        f"EKF failed to track RW bias: truth={truth_bias}, "
        f"est={est_bias}, err={err}")


# ---------------------------------------------------------------------------
# Auto-Q assembles `dt · σ²` on the bias slot
# ---------------------------------------------------------------------------

def test_auto_Q_for_rw_bias_scales_as_dt_sigma_squared():
    """Auto-Q from `L · Σ · Lᵀ`: with ∂bias_next/∂driver = sqrt(dt)·I
    and Σ_driver = σ²·I, Q[bias_block, bias_block] = dt · σ² · I."""
    _, c = _build_world()
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    ekf = TargetNumpy(EKF(w))
    # The bias slot is tangent index 12-14 (after pos, ori, vel, ω).
    dt = 0.01
    L = np.asarray(ekf.ekf._L_fn(ekf.x, np.zeros(0), dt, 0.0))
    Q = L @ ekf.ekf._Sigma @ L.T
    bias_block = Q[12:15, 12:15]
    expected = dt * (0.02 ** 2) * np.eye(3)
    np.testing.assert_allclose(bias_block, expected, atol=1e-12)
