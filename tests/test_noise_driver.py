"""NoiseDriver — stochastic source that makes a Sim's Noise channels live.

A model's `Noise` channels declare a distribution; the compiled tick
exposes each as a zero-seeded input. Without a driver the sim is a
noiseless oracle. `sim.attach_driver(NoiseDriver(seed=...))` draws an
independent `N(0, σ²)` sample per active channel each step and feeds it
in — so truth is genuinely noisy with the *same* σ the EKF reads to
size R/Q.
"""

import numpy as np

from manta import Craft, NoiseDriver, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor


def _gps_world(sigma):
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", position_noise_sigma=sigma))
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, position=(0.0, 0.0, 10.0))
    return w, c


def test_no_driver_is_noiseless_oracle():
    """With σ>0 on the part but no driver attached, the reading is still
    exactly the mean model — the channel stays at its zero seed."""
    w, _ = _gps_world(0.05)
    sim = TargetNumpy(Sim(w))
    assert sim.driver is None
    out = sim.step(dt=0.02)
    np.testing.assert_allclose(
        np.asarray(sim.outputs()["c"]["gps.position"]).ravel(),
        [0.0, 0.0, 10.0], atol=1e-12)


def test_driver_with_zero_sigma_stays_oracle():
    """An attached driver leaves σ=0 channels inert (they're omitted from
    the draw), so the reading is unchanged."""
    w, _ = _gps_world(0.0)
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=1))
    out = sim.step(dt=0.02)
    np.testing.assert_allclose(
        np.asarray(sim.outputs()["c"]["gps.position"]).ravel(),
        [0.0, 0.0, 10.0], atol=1e-12)


def test_white_noise_residual_matches_sigma():
    """The injected white noise is the GPS residual (reading − truth);
    its per-axis std must match the declared σ and its mean ~0."""
    sigma = 0.05
    w, _ = _gps_world(sigma)
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=7))
    res = []
    for _ in range(20000):
        state = sim.step(dt=0.02)
        gps = np.asarray(sim.outputs()["c"]["gps.position"]).ravel()
        truth = np.asarray(state["c"]["position"]).ravel()
        res.append(gps - truth)
    res = np.asarray(res)
    np.testing.assert_allclose(res.std(axis=0), sigma, rtol=0.05)
    np.testing.assert_allclose(res.mean(axis=0), 0.0, atol=3e-3)


def test_white_noise_never_leaks_into_state():
    """The draw lives only in the tick's inputs; the noise slot returns to
    its zero seed each step (independent draws, no accumulation)."""
    w, _ = _gps_world(0.05)
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=2))
    for _ in range(10):
        state = sim.step(dt=0.02)
        np.testing.assert_allclose(
            np.asarray(state["c"]["gps.position_noise"]).ravel(),
            0.0, atol=1e-12)


def test_seed_is_reproducible():
    w, _ = _gps_world(0.05)
    sim = TargetNumpy(Sim(w))

    def first_reading(seed):
        sim.attach_driver(NoiseDriver(seed=seed))
        sim.state = sim.initial_state()
        sim.step(dt=0.02)
        return np.asarray(sim.outputs()["c"]["gps.position"]).ravel()

    np.testing.assert_array_equal(first_reading(11), first_reading(11))
    assert not np.allclose(first_reading(11), first_reading(12))


def test_driver_reset_replays_stream():
    w, _ = _gps_world(0.05)
    sim = TargetNumpy(Sim(w))
    drv = NoiseDriver(seed=5)
    sim.attach_driver(drv)
    sim.step(dt=0.02)
    a = np.asarray(sim.outputs()["c"]["gps.position"])
    drv.reset()
    sim.state = sim.initial_state()
    sim.step(dt=0.02)
    b = np.asarray(sim.outputs()["c"]["gps.position"])
    np.testing.assert_array_equal(a.ravel(), b.ravel())


def test_random_walk_bias_walks_with_driver():
    """An RW channel's bias is frozen without a driver and random-walks
    with one — variance after N steps ≈ N·dt·σ² (kernel applies √dt)."""
    sigma, dt, N = 0.02, 0.01, 100
    c = Craft("c")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("imu", gyro_bias_sigma=sigma))
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 10.0))

    # No driver: bias stays put.
    sim = TargetNumpy(Sim(w))
    for _ in range(N):
        sim.step(dt=dt)
    np.testing.assert_allclose(
        np.asarray(sim.state["c"]["imu.gyro_bias"]).ravel(), 0.0, atol=1e-12)

    # With driver: ensemble std of the terminal bias matches √(N·dt)·σ.
    ends = []
    for trial in range(400):
        sim.attach_driver(NoiseDriver(seed=trial))
        sim.state = sim.initial_state()
        for _ in range(N):
            state = sim.step(dt=dt)
        ends.append(np.asarray(state["c"]["imu.gyro_bias"]).ravel())
    expected = np.sqrt(N * dt) * sigma
    np.testing.assert_allclose(
        np.asarray(ends).std(axis=0), expected, rtol=0.12)
