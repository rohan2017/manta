"""NoiseFit — EKF-innovation NLL fitting of noise σ values."""

import copy

import numpy as np
import pytest

from manta import NoiseDriver, NoiseFit, Prior, Sim, TargetNumpy, Window

from .test_fit import DT, _drone

TRUE_GYRO, TRUE_ACCEL = 0.004, 0.05


def _noisy_drone(gyro=TRUE_GYRO, accel=TRUE_ACCEL):
    w = _drone()
    imu = next(p for p in w.crafts[0].parts if p.name == "imu")
    imu.gyro_noise_sigma = gyro
    imu.accel_noise_sigma = accel
    return w


def _record(world, n_win=3, K=120, seed=5):
    rng = np.random.default_rng(seed)
    truth = TargetNumpy(Sim(world))
    truth.attach_driver(NoiseDriver(seed=seed))
    windows = []
    for _ in range(n_win):
        x0 = copy.deepcopy(truth.state)
        thr = {f"t{i}": np.clip(0.45 + 0.08 * rng.standard_normal(K),
                                0.05, 1.0) for i in range(1, 5)}
        Zg, Za = [], []
        for k in range(K):
            for nm, tr in thr.items():
                truth.state["drone"][f"{nm}.throttle"] = tr[k]
            truth.step(DT)
            out = truth.outputs()["drone"]
            Zg.append(out["imu.gyro"].copy())
            Za.append(out["imu.accel"].copy())
        windows.append(Window(
            x0=x0, u={f"{n}.throttle": tr for n, tr in thr.items()},
            z={"imu.gyro": np.array(Zg), "imu.accel": np.array(Za)},
            dt=DT))
    return windows


@pytest.fixture(scope="module")
def fitted():
    """One shared recovery solve: model starts 5x off, each way."""
    windows = _record(_noisy_drone(), n_win=2, K=100)
    model = _noisy_drone(gyro=0.02, accel=0.01)
    nf = NoiseFit(model, noise={
        "imu.gyro_noise": Prior(sigma=2.0),
        "imu.accel_noise": Prior(sigma=2.0),
    })
    return model, nf.solve(windows)


def test_noise_fit_recovers_sigmas(fitted):
    _, res = fitted
    g = res.values["drone.imu.gyro_noise"]
    a = res.values["drone.imu.accel_noise"]
    # ~200 samples/axis: expect σ to ~10%; assert 20%.
    assert abs(g - TRUE_GYRO) / TRUE_GYRO < 0.2, g
    assert abs(a - TRUE_ACCEL) / TRUE_ACCEL < 0.2, a
    # The data informed both: posterior ≪ prior.
    assert np.all(res.posterior_sigma < 0.1 * res.prior_sigma)


def test_noise_fit_apply_writes_sigma_attrs(fitted):
    model, res = fitted
    res.apply()
    imu = next(p for p in model.crafts[0].parts if p.name == "imu")
    assert imu.gyro_noise_sigma == res.values["drone.imu.gyro_noise"]
    assert imu.accel_noise_sigma == res.values["drone.imu.accel_noise"]


def test_noise_fit_summary_lists_channels(fitted):
    _, res = fitted
    s = res.summary()
    assert "drone.imu.gyro_noise" in s and "post/prior" in s


def test_noise_fit_zero_sigma_start_raises():
    # gyro σ declared 0 and no prior mean → no positive starting point.
    model = _noisy_drone(gyro=0.0)
    with pytest.raises(ValueError, match="positive"):
        NoiseFit(model, noise={"imu.gyro_noise": Prior(sigma=1.0)})


def test_noise_fit_zero_sigma_with_prior_mean_starts():
    model = _noisy_drone(gyro=0.0)
    nf = NoiseFit(model, noise={"imu.gyro_noise": Prior(sigma=1.0,
                                                        mean=0.01),
                                "imu.accel_noise": Prior(sigma=1.0)})
    assert any(c.alias == "drone.imu.gyro_noise" for c in nf.channels)


def test_noise_fit_singular_sensor_raises():
    # Both IMU channels at σ=0 and unfitted → first update singular.
    model = _noisy_drone(gyro=0.0, accel=0.0)
    # Fit only a process channel: thruster force noise must exist —
    # declare it active so the channel synthesizes.
    t1 = next(p for p in model.crafts[0].parts if p.name == "t1")
    t1.force_noise_sigma = 0.1
    with pytest.raises(ValueError, match="zero measurement noise"):
        NoiseFit(model, noise={"t1.force_noise": Prior(sigma=1.0)})


def test_noise_fit_unknown_channel_raises():
    with pytest.raises(KeyError):
        NoiseFit(_noisy_drone(), noise={"imu.bogus": None})


def test_noise_fit_missing_trace_raises():
    windows = _record(_noisy_drone(), n_win=1, K=20)
    for w in windows:
        del w.z["imu.accel"]
    nf = NoiseFit(_noisy_drone(), noise={"imu.gyro_noise": Prior(sigma=1.0)})
    with pytest.raises(ValueError, match="missing trace"):
        nf.solve(windows)
