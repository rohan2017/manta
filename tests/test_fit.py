"""Parameter promotion + system-ID fitting (`manta.fit`).

Covers the whole promotion pipeline — `Parameter(manifold=...)` →
trace-bound graph input → `Role.PARAMETER` port → numpy runtime — and
the MAP fit itself: synthetic recoverability, prior behavior on an
unobservable direction, and the posterior diagnostics.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest

from manta import (
    Craft,
    Fit,
    Free,
    ModelArtifact,
    NoiseDriver,
    Prior,
    Sim,
    TargetNumpy,
    Tied,
    Window,
    World,
)
from manta.fields import GravityField
from manta.fit import FitEvidence, window_digest
from manta.ir.frames import CraftFrame, PartFrame
from manta.ir.types import Vec3
from manta.model import canonical_derivation_bytes
from manta.parts import IMU, Mass, Thruster

DT = 0.005


def _drone(mass=1.2, kf=10.0, arm=0.12):
    d = Craft("drone")
    d.add(Mass("body", mass=mass, moi=(0.02, 0.02, 0.04)))
    for nm, (x, y, s) in {"t1": (arm, arm, 1), "t2": (-arm, arm, -1),
                          "t3": (-arm, -arm, 1), "t4": (arm, -arm, -1)}.items():
        d.add(Thruster(nm, force_quad=(0, 0, kf),
                       torque_quad=(0, 0, 0.02 * s), mount_offset=(x, y, 0)))
    d.add(IMU("imu", mount_offset=(0.04, 0.0, 0.0)))
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
        _drone(), parameters=["t1.force_quad", "body.mass", "t1.mount_offset"])),
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
    # `force_tensors` is a plain Parameter (no manifold) — not promotable.
    from manta.fields import FluidField, UniformFluid
    from manta.parts import DragSurface
    world = _drone()
    world.add_field(FluidField().add(UniformFluid(density=1.225)))
    world.crafts[0].add(DragSurface.isotropic_quadratic(
        "hull", area=0.05, drag_coefficient=1.0))
    with pytest.raises(KeyError):
        Sim(world, parameters=["hull.force_tensors"])


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


def _held_out_windows(world, n_win=1, K=300, seed=11, accel_sigma=0.05):
    """Noisy truth rollouts from a *different* seed: the untouched
    acceptance set the derived artifact's evidence is computed on."""
    world = copy.deepcopy(world)
    imu = next(p for p in world.crafts[0].parts if p.name == "imu")
    imu.accel_noise_sigma = accel_sigma
    rng = np.random.default_rng(seed)
    truth = TargetNumpy(Sim(world))
    truth.attach_driver(NoiseDriver(seed=seed))
    windows = []
    for _ in range(n_win):
        x0 = copy.deepcopy(truth.state)
        thr = {f"t{i}": np.clip(0.4 + 0.1 * rng.standard_normal(K),
                                0.05, 1.0) for i in range(1, 5)}
        Za = []
        for k in range(K):
            for nm, tr in thr.items():
                truth.state["drone"][f"{nm}.throttle"] = tr[k]
            truth.step(DT)
            Za.append(truth.outputs()["drone"]["imu.accel"].copy())
        windows.append(Window(
            x0=x0, u={f"{nm}.throttle": tr for nm, tr in thr.items()},
            z={"imu.accel": np.array(Za)}, dt=DT))
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
    t1.mount_offset = (0.15, 0.12, 0.0)          # 3 cm wrong in x
    fit = Fit(model, parameters={"t1.mount_offset": Prior(sigma=0.05)})
    res = fit.solve(windows)
    assert np.allclose(res.values["drone.t1.mount_offset"],
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


def test_fit_recovers_moi_from_gyro():
    """`Mass.moi` is promotable: roll/pitch inertia is strongly observable
    through the gyro under differential thrust; yaw inertia only sees the
    (small) reaction torque, so its posterior must flag it as far less
    identified than the other two axes."""
    windows = _record_windows(_drone(), n_win=4, K=60, seed=6)
    model = _drone()
    body = next(p for p in model.crafts[0].parts if p.name == "body")
    body.moi = (0.034, 0.011, 0.06)           # truth is (0.02, 0.02, 0.04)
    fit = Fit(model, parameters={"body.moi": Prior(sigma=0.05)})
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    moi = res.values["drone.body.moi"]
    assert np.allclose(moi[:2], [0.02, 0.02], atol=1e-3), moi
    ix = res.labels.index("drone.body.moi[0]")
    iz = res.labels.index("drone.body.moi[2]")
    assert res.posterior_sigma[iz] > 5 * res.posterior_sigma[ix]
    res.apply()
    assert np.allclose(body.moi[:2], [0.02, 0.02], atol=1e-3)


def test_fit_apply_bakes_fitted_values():
    windows = _record_windows(_drone(kf=11.0), n_win=2, K=40, seed=4)
    model = _drone(kf=9.0)
    fit = Fit(model, parameters={
        **{f"t{i}.force_quad": Prior(sigma=4.0) for i in range(1, 5)}})
    base_model = Sim(model).model
    assert fit.sim.model.artifact_id == base_model.artifact_id
    assert "drone.t1.force_quad" in base_model.parameter_names
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})

    # Derivation produces a new executable revision and records the typed
    # held-out evidence without mutating the editable authoring World.
    held_out = _held_out_windows(_drone(kf=11.0))
    evidence = res.evidence(held_out, sensor="imu.accel")
    assert isinstance(evidence, FitEvidence)
    assert evidence.channel == "drone.imu.accel"
    assert evidence.held_out.sample_count == 300
    assert evidence.accepted, evidence.summary()
    assert evidence.binding is not None
    assert evidence.binding.fitted_artifact_id == res._candidate_artifact().artifact_id
    transferred = replace(
        evidence,
        binding=replace(evidence.binding, source_artifact_id="another-source"))
    with pytest.raises(ValueError, match="different fit/model scope"):
        res.derive(evidence=transferred)
    # White accelerometer noise on a well-fitted model: every axis falls
    # back to the white model, and says why.
    assert all(ax.white_fallback and ax.white_fallback_reason
               for ax in evidence.axes)
    assert all(abs(ax.white_sigma - 0.05) < 0.015 for ax in evidence.axes)
    # A training window can never pose as held-out evidence.
    with pytest.raises(ValueError, match="training window"):
        res.evidence(windows[:1], sensor="imu.accel")
    # Evidence is typed — there is no dict form.
    with pytest.raises(TypeError, match="FitEvidence"):
        res.derive(evidence={"accepted": True})
    # Without evidence the revision is visibly unaccepted.
    assert not res.derive().derivation["fit"].accepted
    assert res.derive().derivation["fit"].evidence is None

    derived = res.derive(evidence=evidence)
    assert isinstance(derived, ModelArtifact)
    assert derived.derivation["fit"].accepted
    assert derived.derivation["fit"].evidence is evidence
    assert derived.derivation["fit"].source_artifact_id == fit.sim.model.artifact_id
    assert derived.artifact_id != derived.model_id
    assert derived.artifact_id != res.derive().artifact_id
    derived_t1 = next(p for p in derived.world_copy().crafts[0].parts
                      if p.name == "t1")
    original_t1 = next(p for p in model.crafts[0].parts if p.name == "t1")
    assert abs(derived_t1.force_quad[2] - 11.0) < 0.05
    assert original_t1.force_quad[2] == 9.0
    assert Sim(derived).model.artifact_id == derived.artifact_id

    res.apply()
    t1 = next(p for p in model.crafts[0].parts if p.name == "t1")
    assert abs(t1.force_quad[2] - 11.0) < 0.05
    # A fresh (un-promoted) Sim bakes the fitted values: its rollout now
    # matches truth.
    truth = _run(TargetNumpy(Sim(_drone(kf=11.0))), {"t2": 0.6}, 25)
    fitted = _run(TargetNumpy(Sim(model)), {"t2": 0.6}, 25)
    assert np.allclose(truth.state["drone"]["velocity"],
                       fitted.state["drone"]["velocity"], atol=1e-6)


def test_partial_window_defaults_are_provenance_across_dataset_roles():
    def partial_window(seed: int, t0: float) -> Window:
        truth = TargetNumpy(Sim(_drone()))
        rng = np.random.default_rng(seed)
        accel = []
        for _ in range(120):
            truth.step(DT)
            mean = np.asarray(truth.outputs()["drone"]["imu.accel"])
            accel.append(mean + rng.normal(0.0, 0.02, 3))
        return Window(
            x0={},
            u={},
            z={"imu.accel": np.asarray(accel)},
            dt=DT,
            t0=t0,
        )

    training = partial_window(21, 0.0)
    selection = partial_window(22, 1.0)
    acceptance = partial_window(23, 2.0)
    result = Fit(
        _drone(), parameters={"body.mass": Prior(sigma=0.1)}
    ).solve([training], compute_posterior=False)

    exploratory = result.derive().derivation["fit"]
    assert exploratory.evidence is None
    assert exploratory.default_fills
    assert {fill.dataset_role for fill in exploratory.default_fills} == {
        "training"
    }

    evidence = result.evidence(
        [acceptance], sensor="imu.accel", lag_count=10,
        selection=[selection],
    )
    assert {fill.dataset_role for fill in evidence.default_fills} == {
        "training", "selection", "acceptance"
    }
    for role, window in (
        ("training", training),
        ("selection", selection),
        ("acceptance", acceptance),
    ):
        role_fills = tuple(
            fill for fill in evidence.default_fills
            if fill.dataset_role == role
        )
        assert role_fills
        assert {fill.window_digest for fill in role_fills} == {
            window_digest(window)
        }
        assert "drone.t1.throttle" in {fill.name for fill in role_fills}
        assert "drone.position" in {fill.name for fill in role_fills}

    repeated = result.evidence(
        [acceptance], sensor="imu.accel", lag_count=10,
        selection=[selection],
    )
    assert canonical_derivation_bytes({"fit": evidence}) == \
        canonical_derivation_bytes({"fit": repeated})
    accepted_report = result.derive(evidence=evidence).derivation["fit"]
    assert accepted_report.default_fills == evidence.default_fills


def test_fit_summary_and_weak_directions():
    windows = _record_windows(_drone(), n_win=2, K=30, seed=5)
    fit = Fit(_drone(), parameters={"body.mass": Prior(sigma=0.1, log=True)})
    res = fit.solve(windows)
    assert res.converged is True
    s = res.summary()
    assert "converged" in s
    assert "drone.body.mass" in s and "post/prior" in s
    dirs = res.weak_directions(1)
    assert len(dirs) == 1 and isinstance(dirs[0][0], float)


def test_fit_validates_windows():
    fit = Fit(_drone(), parameters={"body.mass": Prior(sigma=0.1)})
    with pytest.raises(ValueError, match="at least one Window"):
        fit.solve([])
    truth = TargetNumpy(Sim(_drone()))
    with pytest.raises(ValueError, match="at least one state or sensor"):
        fit.solve([Window(x0=truth.state, z={}, dt=DT)])


def test_fit_accepts_ground_truth_state_trajectories_without_sensors():
    """System identification must not require an IMU-shaped proxy when a
    simulator or motion-capture system provides the state trajectory."""
    truth = TargetNumpy(Sim(_drone(mass=1.32, kf=11.0)))
    x0 = copy.deepcopy(truth.state)
    K = 35
    traces = {name: [] for name in
              ("position", "orientation", "velocity", "angular_velocity")}
    throttle = np.linspace(0.2, 0.8, K)
    for value in throttle:
        truth.step(DT, u={f"t{i}.throttle": float(value)
                          for i in range(1, 5)})
        for name, trace in traces.items():
            trace.append(copy.deepcopy(truth.state["drone"][name]))
    window = Window(
        x0=x0,
        u={f"t{i}.throttle": throttle for i in range(1, 5)},
        x={"drone": {name: np.asarray(values)
                     for name, values in traces.items()}},
        dt=DT)
    result = Fit(_drone(mass=1.5, kf=11.0), parameters={
        "body.mass": Prior(sigma=0.5, log=True),
    }).solve([window], state_weights={
        "position": 1e5, "velocity": 1e5,
        "orientation": 1e4, "angular_velocity": 1e4,
    })
    assert result.values["drone.body.mass"] == pytest.approx(1.32, abs=0.02)


def test_state_scales_make_loss_a_normalized_trajectory_mean():
    """A scaled state slot contributes RMS², independent of trace length."""
    world = _drone(mass=1.5, kf=11.0)
    sim = TargetNumpy(Sim(world))
    x0 = copy.deepcopy(sim.state)
    K = 12
    truth = np.tile(np.asarray(x0["drone"]["position"])
                    + np.array([2.0, 0.0, 0.0]), (K, 1))
    window = Window(
        x0=x0, x={"drone": {"position": truth}},
        x_scale={"position": 2.0}, dt=0.0)
    # The model predicts zero position: ||error|| / scale = 1 at every
    # sample, hence the normalized trajectory objective is exactly one.
    result = Fit(world, parameters={
        "body.mass": Prior(sigma=0.5),
    }).solve([window], compute_posterior=False,
             ipopt_options={"ipopt.max_iter": 0})
    assert result.initial_objective == pytest.approx(1.0)


def test_state_robust_loss_bounds_a_bad_trajectory_influence():
    world = _drone(mass=1.5, kf=11.0)
    sim = TargetNumpy(Sim(world))
    x0 = copy.deepcopy(sim.state)
    K = 8
    window = Window(
        x0=x0,
        x={"drone": {"position": np.tile(
            np.asarray(x0["drone"]["position"])
            + np.array([20.0, 0.0, 0.0]), (K, 1))}},
        x_scale={"position": 2.0}, dt=0.0)
    result = Fit(world, parameters={
        "body.mass": Prior(sigma=0.5),
    }).solve([window], state_robust_delta=0.2,
             compute_posterior=False,
             ipopt_options={"ipopt.max_iter": 0})
    # Plain normalized MSE is 100. Pseudo-Huber at delta=.2 is ~3.92.
    assert result.initial_objective == pytest.approx(
        2 * 0.2**2 * (np.sqrt(1 + 100 / 0.2**2) - 1))


def test_state_scale_and_robust_delta_validation():
    world = _drone()
    x0 = TargetNumpy(Sim(world)).state
    window = Window(
        x0=x0, x={"drone": {"position": np.zeros((2, 3))}},
        x_scale={"position": 0.0}, dt=DT)
    fit = Fit(world, parameters={"body.mass": Prior(sigma=0.1)})
    with pytest.raises(ValueError, match="state scale"):
        fit.solve([window])
    with pytest.raises(ValueError, match="state_robust_delta"):
        fit.solve([Window(x0=x0, x={"drone": {
            "position": np.zeros((2, 3))}}, dt=DT)],
                  state_robust_delta=0.0)


def test_fit_accepts_an_ambient_parameter_warm_start():
    windows = _record_windows(_drone(mass=1.32), n_win=1, K=10)
    fit = Fit(_drone(mass=1.5), parameters={
        "body.mass": Prior(sigma=0.5, lower=1.0, upper=2.0),
    })
    with pytest.warns(RuntimeWarning, match="did NOT converge"):
        result = fit.solve(
            windows, initial_values={"body.mass": 1.32},
            compute_posterior=False, ipopt_options={"ipopt.max_iter": 0})
    assert result.initial_objective < 1.0
    with pytest.raises(ValueError, match="violates its bounds"):
        fit.solve(windows, initial_values={"body.mass": 3.0})


def test_limited_fit_returns_the_best_accepted_iterate():
    """An iteration cap must never return a worse model than its input."""
    windows = _record_windows(_drone(mass=1.32), n_win=1, K=15)
    fit = Fit(_drone(mass=1.5), parameters={
        "body.mass": Prior(sigma=0.5),
    })
    with pytest.warns(RuntimeWarning, match="did NOT converge"):
        result = fit.solve(
            windows, compute_posterior=False,
            ipopt_options={"ipopt.max_iter": 1})
    assert result.objective <= result.initial_objective
    assert result.objective_history[-1] == pytest.approx(result.objective)


def test_fit_validates_window_traces():
    """Wrong-width z, mismatched trace lengths, and a wrong-length u trace
    all raise before any solve."""
    fit = Fit(_drone(), parameters={"body.mass": Prior(sigma=0.1)})
    x0 = TargetNumpy(Sim(_drone())).state
    K = 10
    with pytest.raises(ValueError, match=r"expected \(K, 3\)"):
        fit.solve([Window(x0=x0, z={"imu.gyro": np.zeros((K, 2))}, dt=DT)])
    with pytest.raises(ValueError, match="trace length"):
        fit.solve([Window(x0=x0, z={"imu.gyro": np.zeros((K, 3)),
                                    "imu.accel": np.zeros((K + 2, 3))},
                          dt=DT)])
    with pytest.raises(ValueError, match=r"scalar or length-10"):
        fit.solve([Window(x0=x0, z={"imu.gyro": np.zeros((K, 3))},
                          u={"t1.throttle": np.zeros(K + 3)}, dt=DT)])


def test_fit_unconverged_sets_flag_and_warns():
    """A solve cut off by the iteration cap must say so: converged=False,
    a RuntimeWarning at solve, a loud summary, and apply must refuse it."""
    windows = _record_windows(_drone(kf=11.0), n_win=1, K=10, seed=7)
    fit = Fit(_drone(kf=9.0),
              parameters={"t1.force_quad": Prior(sigma=4.0)})
    with pytest.warns(RuntimeWarning, match="did NOT converge"):
        res = fit.solve(windows, ipopt_options={"ipopt.max_iter": 1})
    assert res.converged is False
    assert "NOT CONVERGED" in res.summary()
    with pytest.raises(RuntimeError, match="refuses"):
        res.apply()
    with pytest.raises(RuntimeError, match="refuses"):
        res.derive()


def test_log_prior_supports_vector_parameters():
    """log=True is elementwise: a vector-positive parameter (moi) rides as
    log(p_j) per component; a vector with a zero component is refused."""
    fit = Fit(_drone(), parameters={"body.moi": Prior(sigma=0.1, log=True)})
    b = next(blk for blk in fit._blocks if blk.full == "drone.body.moi")
    assert b.log and b.dim == 3
    assert np.allclose(b.init, np.log([0.02, 0.02, 0.04]))
    assert np.allclose(b.theta_of_v(b.init), [0.02, 0.02, 0.04])
    # t1.mount_offset has a zero z-component — log-reparam is impossible.
    with pytest.raises(ValueError, match="strictly positive"):
        Fit(_drone(),
            parameters={"t1.mount_offset": Prior(sigma=0.1, log=True)})


def test_tied_parameters_share_one_decision_variable():
    """Four identical thrusters fit ONE gain: t2..t4 are Tied to t1, so
    the decision space has 3 components (one vec3), every window informs
    the shared gain, and apply() writes the derived copies back too."""
    windows = _record_windows(_drone(kf=11.0), n_win=2, K=40, seed=10)
    model = _drone(kf=9.0)
    fit = Fit(model, parameters={
        "t1.force_quad": Prior(sigma=4.0),
        **{f"t{i}.force_quad": Tied("t1.force_quad") for i in (2, 3, 4)},
    })
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    assert len(res.labels) == 3               # one vec3, not four
    for i in range(1, 5):
        assert np.allclose(res.values[f"drone.t{i}.force_quad"],
                           [0, 0, 11.0], atol=0.05), i
    s = res.summary()
    assert "← drone.t1.force_quad" in s
    res.apply()
    t3 = next(p for p in model.crafts[0].parts if p.name == "t3")
    assert abs(t3.force_quad[2] - 11.0) < 0.05


def test_free_arm_length_with_matrix_ties():
    """The user's quadcopter symmetry: ONE scalar arm length (a `Free`
    decision variable) sources all four X-frame mount positions through
    fixed direction matrices — the fit can only slide the motors in and
    out together, so the recovered geometry stays a quadcopter."""
    windows = _record_windows(_drone(arm=0.15), n_win=3, K=50, seed=9)
    model = _drone(arm=0.12)                  # 3 cm wrong everywhere
    signs = {"t1": (1, 1), "t2": (-1, 1), "t3": (-1, -1), "t4": (1, -1)}
    fit = Fit(model, parameters={
        "arm": Free(0.12, prior=Prior(sigma=0.05, lower=0.0)),
        **{f"{nm}.mount_offset": Tied("arm", scale=[[sx], [sy], [0.0]])
           for nm, (sx, sy) in signs.items()},
    })
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    assert res.converged
    assert abs(res.values["arm"] - 0.15) < 0.002
    assert np.allclose(res.values["drone.t2.mount_offset"],
                       [-0.15, 0.15, 0.0], atol=0.002)
    res.apply()
    t4 = next(p for p in model.crafts[0].parts if p.name == "t4")
    assert np.allclose(t4.mount_offset, (0.15, -0.15, 0.0), atol=0.002)


def test_mirror_tie_elementwise_scale():
    """A per-component scale is a mirror map: t2's mount is t1's with the
    x sign flipped, so fitting t1's position moves both symmetrically."""
    windows = _record_windows(_drone(arm=0.12), n_win=3, K=50, seed=2)
    model = _drone()
    for nm in ("t1", "t2"):
        p = next(q for q in model.crafts[0].parts if q.name == nm)
        p.mount_offset = (0.15 if nm == "t1" else -0.15, 0.12, 0.0)
    fit = Fit(model, parameters={
        "t1.mount_offset": Prior(sigma=0.05),
        "t2.mount_offset": Tied("t1.mount_offset", scale=(-1, 1, 1)),
    })
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    assert np.allclose(res.values["drone.t1.mount_offset"],
                       [0.12, 0.12, 0.0], atol=0.003)
    assert np.allclose(res.values["drone.t2.mount_offset"],
                       [-0.12, 0.12, 0.0], atol=0.003)


def test_bounds_clamp_fitted_value():
    """Prior(upper=) is a hard wall: truth kf=11 but the gain is capped
    at 10 — the solve converges ON the bound instead of crossing it."""
    windows = _record_windows(_drone(kf=11.0), n_win=2, K=40, seed=11)
    model = _drone(kf=9.0)
    fit = Fit(model, parameters={
        "t1.force_quad": Prior(sigma=4.0, lower=-1.0,
                               upper=(1.0, 1.0, 10.0)),
        **{f"t{i}.force_quad": Tied("t1.force_quad") for i in (2, 3, 4)},
    })
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    kf = res.values["drone.t1.force_quad"][2]
    assert kf <= 10.0 + 1e-6
    assert kf > 9.9                           # pushed onto the wall


def test_tie_and_bound_validation():
    """Config errors raise at construction, before any solve: bad tie
    maps, chained ties, unknown sources, and self-violating bounds."""
    with pytest.raises(ValueError, match="matrix scale"):
        Fit(_drone(), parameters={
            "arm": Free(0.1),
            "t1.mount_offset": Tied("arm", scale=[[1.0], [1.0]])})
    with pytest.raises(ValueError, match="dim 1 != target dim 3"):
        Fit(_drone(), parameters={
            "body.mass": Prior(sigma=0.1),
            "t1.mount_offset": Tied("body.mass")})
    with pytest.raises(ValueError, match="itself tied"):
        Fit(_drone(), parameters={
            "t1.force_quad": Prior(sigma=1.0),
            "t2.force_quad": Tied("t1.force_quad"),
            "t3.force_quad": Tied("t2.force_quad")})
    with pytest.raises(KeyError, match="unknown source"):
        Fit(_drone(), parameters={
            "t1.force_quad": Prior(sigma=1.0),
            "t2.force_quad": Tied("bogus.param")})
    with pytest.raises(ValueError, match="violates its own bounds"):
        Fit(_drone(), parameters={"body.mass": Prior(sigma=0.1, lower=2.0)})
    with pytest.raises(ValueError, match="lower < upper"):
        Fit(_drone(), parameters={
            "body.mass": Prior(sigma=0.1, lower=1.0, upper=0.5)})
    with pytest.raises(ValueError, match="fit nothing"):
        Fit(_drone(), parameters={"arm": Free(0.1)})


def test_expand_off_matches_default():
    """The NLP is SX-expanded for speed by default. That is a graph
    representation, not a change of problem: the MX path must land on
    the same optimum."""
    windows = _record_windows(_drone(kf=11.0), n_win=2, K=40, seed=12)
    params = {"t1.force_quad": Prior(sigma=4.0),
              **{f"t{i}.force_quad": Tied("t1.force_quad")
                 for i in (2, 3, 4)}}
    w = {"imu.gyro": 1e6, "imu.accel": 1e4}
    fast = Fit(_drone(kf=9.0), parameters=params).solve(windows, weights=w)
    slow = Fit(_drone(kf=9.0), parameters=params).solve(
        windows, weights=w, ipopt_options={"expand": False})
    assert np.allclose(fast.values["drone.t1.force_quad"],
                       slow.values["drone.t1.force_quad"], atol=1e-6)


def test_fit_falls_back_when_nlp_cannot_expand():
    """A jointed craft's joint-space solve rides on the default `Linsol`,
    which has no `eval_sx` — SX expansion raises. The fitter must take
    the MX path instead of failing the solve, and it must SAY so: a
    RuntimeWarning at solve time and `result.expanded == False` (the MX
    path is an order of magnitude slower per IPOPT iteration — a user
    should never discover that by profiling)."""
    from manta.parts import RevoluteJoint

    def jointed(force_z):
        c = Craft("rover")
        c.add(Mass("body", mass=4.0, moi=(0.4, 0.5, 0.6)))
        c.add(Thruster("t", force=(0.0, 0.0, force_z),
                       mount_offset=(0.3, 0.0, 0.0)))
        wheel = RevoluteJoint("wheel", mode="passive", axis=(0, 0, 1.0))
        wheel.add(Mass("disk", mass=0.3, moi=(4e-3, 6e-3, 8e-3)))
        c.add(wheel)
        c.add(IMU("imu"))
        w = World()
        w.add_field(GravityField(g=(0, 0, -9.81)))
        w.add_craft(c, position=(0, 0, 10.0))
        return w

    truth = TargetNumpy(Sim(jointed(1.0)))
    x0 = copy.deepcopy(truth.state)
    Z = []
    for _ in range(25):
        truth.state["rover"]["t.throttle"] = 0.6
        truth.step(DT)
        Z.append(truth.outputs()["rover"]["imu.gyro"].copy())
    win = [Window(x0=x0, u={"t.throttle": 0.6},
                  z={"imu.gyro": np.array(Z)}, dt=DT)]
    with pytest.warns(RuntimeWarning, match="cannot SX-expand"):
        res = Fit(jointed(1.6),
                  parameters={"t.force": Prior(sigma=1.0)}).solve(win)
    assert res.expanded is False
    assert res.converged
    assert np.isfinite(res.values["rover.t.force"]).all()


def test_fit_records_expanded_flag():
    """A plain (Linsol-free) fit runs SX-expanded and says so."""
    windows = _record_windows(_drone(mass=1.32, kf=11.0))
    res = Fit(_drone(mass=1.5, kf=9.0), parameters={
        "body.mass": Prior(sigma=0.5),
        **{f"t{i}.force_quad": Prior(sigma=4.0) for i in range(1, 5)},
    }).solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    assert res.expanded is True


def test_fit_exposes_loss_history_and_can_skip_posterior_diagnostics():
    windows = _record_windows(_drone(mass=1.32), n_win=1, K=20)
    res = Fit(_drone(mass=1.5), parameters={
        "body.mass": Prior(sigma=0.5),
    }).solve(windows, compute_posterior=False,
             ipopt_options={"expand": False,
                            "ipopt.hessian_approximation": "limited-memory"})
    assert len(res.objective_history) >= 2
    assert res.objective_history[0] == pytest.approx(res.initial_objective)
    assert res.objective_history[-1] == pytest.approx(res.objective)
    assert res.objective_history[-1] < res.objective_history[0]
    assert not res.posterior_computed
    assert np.isnan(res.posterior_sigma).all()


def test_log_prior_with_bounds():
    """Ambient bounds compose with the log reparam: mass is fit in
    log-space but bounded in kg, and the recovered value respects both."""
    windows = _record_windows(_drone(mass=1.32, kf=11.0))
    model = _drone(mass=1.5, kf=9.0)
    fit = Fit(model, parameters={
        "body.mass": Prior(sigma=0.2, log=True, lower=0.5, upper=3.0),
        **{f"t{i}.force_quad": Prior(sigma=4.0) for i in range(1, 5)},
    })
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    assert abs(res.values["drone.body.mass"] - 1.32) < 0.01


def test_fit_recovers_moi_with_vector_log_prior():
    """End-to-end vector log=True: the observable roll/pitch inertias are
    recovered through the log-reparam (staying positive throughout)."""
    windows = _record_windows(_drone(), n_win=3, K=50, seed=8)
    model = _drone()
    body = next(p for p in model.crafts[0].parts if p.name == "body")
    body.moi = (0.034, 0.011, 0.06)           # truth is (0.02, 0.02, 0.04)
    fit = Fit(model, parameters={"body.moi": Prior(sigma=1.0, log=True)})
    res = fit.solve(windows, weights={"imu.gyro": 1e6, "imu.accel": 1e4})
    moi = res.values["drone.body.moi"]
    assert np.all(moi > 0)
    assert np.allclose(moi[:2], [0.02, 0.02], rtol=0.05), moi
