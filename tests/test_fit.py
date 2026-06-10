"""Parameter promotion + system-ID fitting (`manta.fit`).

Covers the whole promotion pipeline — `Parameter(manifold=...)` →
trace-bound graph input → `Role.PARAMETER` port → numpy runtime — and
the MAP fit itself: synthetic recoverability, prior behavior on an
unobservable direction, and the posterior diagnostics.
"""

import copy

import numpy as np
import pytest

from manta import Craft, Fit, Prior, Sim, TargetNumpy, Window, World
from manta.fields import GravityField
from manta.ir.frames import CraftFrame, PartFrame
from manta.ir.types import Vec3
from manta.parts import IMU, Mass, Thruster

DT = 0.005


def _drone(mass=1.2, kf=10.0, arm=0.12):
    d = Craft("drone")
    d.add(Mass("body", mass=mass, moi=(0.02, 0.02, 0.04)))
    for nm, (x, y, s) in {"t1": (arm, arm, 1), "t2": (-arm, arm, -1),
                          "t3": (-arm, -arm, 1), "t4": (arm, -arm, -1)}.items():
        d.add(Thruster(nm, force_quad=(0, 0, kf),
                       torque_quad=(0, 0, 0.02 * s), transform=(x, y, 0)))
    d.add(IMU("imu", transform=(0.04, 0.0, 0.0)))
    w = World()
    w.add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(d, position=(0, 0, 10))
    return w


def _run(sim, throttles, n):
    for nm, v in throttles.items():
        sim.state["drone"][f"{nm}.throttle"] = v
    for _ in range(n):
        sim.step(DT)
    return sim


# ---------------------------------------------------------------------------
# Promotion mechanics
# ---------------------------------------------------------------------------

def test_promoted_defaults_reproduce_baked_model_exactly():
    base = _run(TargetNumpy(Sim(_drone())), {"t1": 0.6, "t3": 0.4}, 25)
    prom = _run(TargetNumpy(Sim(
        _drone(), parameters=["t1.force_quad", "body.mass", "t1.transform"])),
        {"t1": 0.6, "t3": 0.4}, 25)
    for slot in ("position", "orientation", "velocity", "angular_velocity"):
        assert np.array_equal(base.state["drone"][slot],
                              prom.state["drone"][slot]), slot


def test_param_port_layout_and_defaults():
    sim = TargetNumpy(Sim(_drone(mass=1.7, kf=8.5),
                          parameters=["body.mass", "t2.force_quad"]))
    port = sim.module.port("params")
    by_name = {f.name: f for f in port.fields}
    assert set(by_name) == {"drone.body.mass", "drone.t2.force_quad"}
    assert by_name["drone.body.mass"].dim == 1
    assert np.allclose(by_name["drone.body.mass"].default, [1.7])
    assert np.allclose(by_name["drone.t2.force_quad"].default, [0, 0, 8.5])
    assert port.size == 4


def test_set_parameters_changes_dynamics():
    # Exact hover: 4·kf·t² = m·g with overridden kf.
    sim = TargetNumpy(Sim(_drone(mass=1.2), parameters=["t1.force_quad",
                                                        "t2.force_quad",
                                                        "t3.force_quad",
                                                        "t4.force_quad"]))
    t = 0.55
    kf = 1.2 * 9.81 / (4 * t * t)
    sim.set_parameters({f"t{i}.force_quad": (0, 0, kf) for i in range(1, 5)})
    _run(sim, {f"t{i}": t for i in range(1, 5)}, 60)
    assert np.allclose(sim.state["drone"]["velocity"], 0.0, atol=1e-12)


def test_step_n_folds_params_identically():
    seq = _run(TargetNumpy(Sim(_drone(), parameters=["body.mass"])),
               {"t1": 0.7}, 30)
    fold = TargetNumpy(Sim(_drone(), parameters=["body.mass"]))
    fold.state["drone"]["t1.throttle"] = 0.7
    fold.step_n(DT, 30)
    assert np.allclose(seq.state["drone"]["position"],
                       fold.state["drone"]["position"])


def test_unknown_parameter_raises():
    with pytest.raises(KeyError, match="parameter"):
        Sim(_drone(), parameters=["t1.bogus"])


def test_non_promotable_parameter_raises():
    # `moi` is a plain Parameter (no manifold) — not promotable.
    with pytest.raises(KeyError):
        Sim(_drone(), parameters=["body.moi"])


def test_set_parameters_without_port_raises():
    sim = TargetNumpy(Sim(_drone()))
    with pytest.raises(ValueError, match="parameter port"):
        sim.set_parameters({"body.mass": 1.0})


def test_coerce_passthrough_and_frame_check():
    from manta.ir import Graph
    from manta.ir.frames import FrameError
    with Graph():
        v = Vec3[PartFrame].input("v")
        assert Vec3[PartFrame].coerce(v) is v
        c = Vec3[PartFrame].coerce((1.0, 2.0, 3.0))
        assert isinstance(c, Vec3)
        with pytest.raises(FrameError):
            Vec3[CraftFrame].coerce(v)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _record_windows(world, n_win=3, K=50, seed=0):
    """Noiseless truth rollouts with per-thruster throttle excitation."""
    rng = np.random.default_rng(seed)
    truth = TargetNumpy(Sim(world))
    windows = []
    for _ in range(n_win):
        x0 = copy.deepcopy(truth.state)
        thr = {f"t{i}": np.clip(0.4 + 0.1 * rng.standard_normal(K),
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
            x0=x0,
            u={f"{nm}.throttle": tr for nm, tr in thr.items()},
            z={"imu.gyro": np.array(Zg), "imu.accel": np.array(Za)},
            dt=DT))
    return windows


def test_fit_recovers_thrust_and_mass():
    windows = _record_windows(_drone(mass=1.32, kf=11.0))
    model = _drone(mass=1.5, kf=9.0)          # wrong starting guesses
    fit = Fit(model, parameters={
        "body.mass": Prior(sigma=0.2, log=True),
        **{f"t{i}.force_quad": Prior(sigma=4.0) for i in range(1, 5)},
    })
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    assert abs(res.values["drone.body.mass"] - 1.32) < 0.01
    for i in range(1, 5):
        kf = res.values[f"drone.t{i}.force_quad"]
        assert np.allclose(kf, [0, 0, 11.0], atol=0.05), (i, kf)
    # Data informed everything: posterior ≪ prior on every component.
    finite = np.isfinite(res.prior_sigma)
    assert np.all(res.posterior_sigma[finite] < 0.2 * res.prior_sigma[finite])


def test_fit_recovers_thruster_arm_from_gyro():
    # Truth has t1 mounted 3 cm off the modeled position; the torque arm
    # is observable through the gyro under differential thrust.
    windows = _record_windows(_drone(arm=0.12), n_win=3, K=50, seed=2)
    model = _drone()
    t1 = next(p for p in model.crafts[0].parts if p.name == "t1")
    t1.transform = (0.15, 0.12, 0.0)          # 3 cm wrong in x
    fit = Fit(model, parameters={"t1.transform": Prior(sigma=0.05)})
    res = fit.solve(windows)
    assert np.allclose(res.values["drone.t1.transform"],
                       [0.12, 0.12, 0.0], atol=0.003)


def test_prior_pins_unobservable_mass_accel_only():
    """Accel-only data observes thrust/mass, not mass: the fitted mass
    must come back ≈ the prior, with posterior σ ≈ prior σ — the
    diagnostic that says 'this number is your prior talking'."""
    windows = _record_windows(_drone(mass=1.2, kf=10.0), n_win=2, K=40,
                              seed=3)
    for w in windows:
        del w.z["imu.gyro"]                   # accel only
    model = _drone(mass=1.2, kf=8.0)          # mass right, thrust wrong
    fit = Fit(model, parameters={
        "body.mass": Prior(sigma=0.05, log=True),
        **{f"t{i}.force_quad": Prior(sigma=4.0) for i in range(1, 5)},
    })
    res = fit.solve(windows, weights={"imu.accel": 1e4})
    # Thrust/mass ratio is what accel observes — kf recovered given the
    # prior pinned mass at its (correct) prior mean.
    for i in range(1, 5):
        assert abs(res.values[f"drone.t{i}.force_quad"][2] - 10.0) < 0.3
    # Mass: data added ~nothing beyond the prior.
    i_mass = res.labels.index("drone.body.mass")
    assert res.posterior_sigma[i_mass] > 0.5 * res.prior_sigma[i_mass]


def test_fit_apply_bakes_fitted_values():
    windows = _record_windows(_drone(kf=11.0), n_win=2, K=40, seed=4)
    model = _drone(kf=9.0)
    fit = Fit(model, parameters={
        **{f"t{i}.force_quad": Prior(sigma=4.0) for i in range(1, 5)}})
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    res.apply()
    t1 = next(p for p in model.crafts[0].parts if p.name == "t1")
    assert abs(t1.force_quad[2] - 11.0) < 0.05
    # A fresh (un-promoted) Sim bakes the fitted values: its rollout now
    # matches truth.
    truth = _run(TargetNumpy(Sim(_drone(kf=11.0))), {"t2": 0.6}, 25)
    fitted = _run(TargetNumpy(Sim(model)), {"t2": 0.6}, 25)
    assert np.allclose(truth.state["drone"]["velocity"],
                       fitted.state["drone"]["velocity"], atol=1e-6)


def test_fit_summary_and_weak_directions():
    windows = _record_windows(_drone(), n_win=2, K=30, seed=5)
    fit = Fit(_drone(), parameters={"body.mass": Prior(sigma=0.1, log=True)})
    res = fit.solve(windows)
    s = res.summary()
    assert "drone.body.mass" in s and "post/prior" in s
    dirs = res.weak_directions(1)
    assert len(dirs) == 1 and isinstance(dirs[0][0], float)


def test_fit_validates_windows():
    fit = Fit(_drone(), parameters={"body.mass": Prior(sigma=0.1)})
    with pytest.raises(ValueError, match="at least one Window"):
        fit.solve([])
    truth = TargetNumpy(Sim(_drone()))
    with pytest.raises(ValueError, match="at least one sensor"):
        fit.solve([Window(x0=truth.state, z={}, dt=DT)])
