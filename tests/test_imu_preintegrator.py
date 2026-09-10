"""High-rate IMU preintegration and lower-rate INS packet propagation."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import (
    INS,
    Craft,
    IMUPreintegrator,
    NoiseFit,
    Prior,
    Sim,
    TargetCpp,
    TargetNumpy,
    Window,
    World,
    frame_preintegrated_packet,
)
from manta.estimation import nees, observability_trajectory
from manta.estimation._assembly import estimator_inputs
from manta.estimation.imu_preintegrator import _single_sample_packet
from manta.estimation.ins import PREINTEGRATION_DURATION_DOC
from manta.fields import GravityField
from manta.ir._rotation import quat_mul_np, so3_exp_np
from manta.parts import IMU, Mass, Thruster
from tests.test_ins import _ins, _world


def _sample(k, dt):
    t = k * dt
    return (
        np.array([0.4 + 0.1 * np.sin(3.0 * t),
                  -0.2 + 0.05 * np.cos(2.0 * t), 9.81 + 0.03 * t]),
        np.array([0.35 * np.cos(4.0 * t),
                  0.28 * np.sin(5.0 * t), 0.45 + 0.1 * np.sin(t)]),
    )


def _preintegrated_ins(world=None):
    world = _world() if world is None else world
    return INS(
        world, imu="craft.imu",
        sensors=["craft.model_force.specific_force"],
        propagation="preintegrated",
    )


def test_packet_contract_and_reset():
    runtime = TargetNumpy(IMUPreintegrator())
    packet = runtime.step(
        0.002, accel=(0, 0, 9.81), gyro=(0, 0, 0),
        accel_bias=(0.1, 0.2, 0.3), gyro_bias=(0.01, 0.02, 0.03))
    assert packet["duration"] == pytest.approx(0.002)
    assert packet["sample_count"] == 1
    assert packet["covariance"].shape == (81,)
    assert packet["delta_start_gyro_cross_covariance"].shape == (27,)
    assert packet["delta_end_gyro_cross_covariance"].shape == (27,)
    np.testing.assert_allclose(
        np.reshape(packet["start_end_gyro_correlation"], (3, 3), order="F"),
        np.eye(3),
    )
    assert packet["bias_jacobian"].shape == (54,)
    np.testing.assert_allclose(packet["gyro_bias_reference"], (.01, .02, .03))
    runtime.reset()
    assert runtime.state["duration"] == 0.0
    np.testing.assert_allclose(
        runtime.state["delta_orientation"], (1, 0, 0, 0))


def test_ordered_so3_products_preserve_coning():
    runtime = TargetNumpy(IMUPreintegrator())
    dt = 0.01
    gyros = [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
             (-2.0, 0.0, 0.0), (0.0, -2.0, 0.0)]
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    for gyro in gyros:
        packet = runtime.step(
            dt, accel=(0, 0, 0), gyro=gyro,
            accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
        expected = quat_mul_np(expected, so3_exp_np(np.asarray(gyro) * dt))
    np.testing.assert_allclose(packet["delta_orientation"], expected,
                               atol=1e-13)
    # Averaging these samples erases the non-commutative residual rotation.
    assert np.linalg.norm(packet["delta_orientation"][1:]) > 1e-4


def test_fresh_endpoint_framing_updates_joint_covariance_atomically():
    recurrence_packet = TargetNumpy(IMUPreintegrator(
        gyro_noise_density=3e-4)).step(
            0.002,
            accel=(0.0, 0.0, 9.81),
            gyro=(0.1, -0.2, 0.3),
            accel_bias=(0.0, 0.0, 0.0),
            gyro_bias=(0.0, 0.0, 0.0),
        )
    assert np.linalg.norm(
        recurrence_packet["delta_end_gyro_cross_covariance"]) > 0.0
    framed = frame_preintegrated_packet(
        recurrence_packet,
        end_accel=(0.4, 0.5, 9.7),
        end_gyro=(0.2, -0.1, 0.4),
        end_gyro_noise_sigma=(0.006, 0.007, 0.008),
    )
    np.testing.assert_array_equal(
        framed["delta_end_gyro_cross_covariance"], np.zeros(27))
    np.testing.assert_array_equal(
        framed["start_end_gyro_correlation"], np.zeros(9))
    np.testing.assert_allclose(
        framed["end_gyro_noise_sigma"], (0.006, 0.007, 0.008))
    # Framing owns a new packet and cannot corrupt a recurrence checkpoint.
    assert np.linalg.norm(
        recurrence_packet["delta_end_gyro_cross_covariance"]) > 0.0


def test_preintegrated_ins_matches_high_rate_raw_ins():
    dt = 0.002
    raw = TargetNumpy(_ins(_world()))
    preintegrator = TargetNumpy(IMUPreintegrator())
    packet_filter = TargetNumpy(_preintegrated_ins())
    for k in range(10):
        accel, gyro = _sample(k, dt)
        u = {"imu.accel": accel, "imu.gyro": gyro}
        raw.predict(dt, u=u)
        packet = preintegrator.step(
            dt, accel=accel, gyro=gyro,
            accel_bias=np.zeros(3), gyro_bias=np.zeros(3))
    packet_filter.predict_preintegrated(packet)
    a = raw.state_dict()["craft"]
    b = packet_filter.state_dict()["craft"]
    for field in ("position", "velocity", "orientation"):
        np.testing.assert_allclose(b[field], a[field], rtol=1e-11, atol=1e-12)


def test_preintegrated_ins_retains_nonzero_imu_lever_arm():
    world = _world(lever=(0.1, 0.0, 0.0),
                   angular_velocity=(0.0, 0.0, 3.0))
    packet_filter = TargetNumpy(_preintegrated_ins(world))
    preintegrator = TargetNumpy(IMUPreintegrator())
    for _ in range(100):
        packet = preintegrator.step(
            .001, accel=(-0.9, 0.0, 9.81), gyro=(0.0, 0.0, 3.0),
            accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
    packet_filter.predict_preintegrated(packet)
    state = packet_filter.state_dict()["craft"]
    # The remaining error is the high-rate left-endpoint quadrature error,
    # not an omitted 0.9 m/s^2 centripetal acceleration at the craft origin.
    assert np.linalg.norm(state["position"]) < 1e-5
    assert np.linalg.norm(state["velocity"]) < 2e-4


def test_preintegrated_ins_uses_true_endpoint_gyro_during_angular_acceleration():
    """Packet boundaries, not held samples, define lever-arm velocity."""
    craft = Craft("vehicle")
    craft.add(Mass("body", mass=20.0, moi=(1.0, 2.0, 3.0)))
    craft.add(Thruster(
        "turn", force=(0.0, 1.0, 0.0), mount_offset=(1.0, 0.0, 0.0)
    ))
    craft.add(IMU(
        "imu",
        mount_offset=(0.625, 0.0, 0.0),
        accel_noise_sigma=0.01,
        gyro_noise_sigma=0.001,
    ))
    world = World("preintegrated_dynamic_lever").add_field(
        GravityField(g=(0.0, 0.0, -9.81))
    )
    world.add_craft(craft)
    sim = TargetNumpy(Sim(world))
    runtime = TargetNumpy(INS(
        world,
        imu="vehicle.imu",
        sensors=[],
        propagation="preintegrated",
    ))
    assert runtime.module.metadata[
        "preintegration_boundary_covariance_qualified"
    ]
    assert runtime.module.metadata[
        "preintegration_boundary_covariance_model"
    ] == "dynamic_schmidt_joint_packet"
    preintegrator = TargetNumpy(IMUPreintegrator())
    dt = 0.004
    control = {"turn.throttle": 10.0}
    previous = None
    for index in range(51):
        truth_at_epoch = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in sim.state["vehicle"].items()
            if name in {"position", "orientation", "velocity"}
        }
        sim.step(dt, t=index * dt, u=control)
        current = {
            "accel": np.asarray(sim.reading("vehicle.imu.accel"), dtype=float),
            "gyro": np.asarray(sim.reading("vehicle.imu.gyro"), dtype=float),
        }
        if previous is not None:
            packet = dict(preintegrator.step(
                dt,
                t=(index - 1) * dt,
                accel=previous["accel"],
                gyro=previous["gyro"],
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
            ))
            packet["end_accel"] = current["accel"]
            packet["end_gyro"] = current["gyro"]
            runtime.predict_preintegrated(
                packet, t=(index - 1) * dt, u=control
            )
            estimate = runtime.state_dict()["vehicle"]
            np.testing.assert_allclose(
                estimate["orientation"], truth_at_epoch["orientation"],
                atol=2e-10,
            )
            # The plant's left-endpoint attitude step and continuous sensor
            # acceleration differ at O(dt), but the omitted endpoint lever
            # velocity (0.42 m/s in this fixture) must not survive.
            np.testing.assert_allclose(
                estimate["velocity"], truth_at_epoch["velocity"], atol=2e-3
            )
            np.testing.assert_allclose(
                estimate["position"], truth_at_epoch["position"], atol=2e-3
            )
            preintegrator.reset()
        previous = current


def test_packet_bias_jacobian_corrects_at_filter_bias():
    dt = 0.001
    accel_bias = np.array([0.025, -0.018, 0.012])
    gyro_bias = np.array([0.004, -0.003, 0.005])
    raw = TargetNumpy(_ins(_world()))
    packet_filter = TargetNumpy(_preintegrated_ins())
    for runtime in (raw, packet_filter):
        state = runtime.state_dict()
        state["craft"]["imu.accel_bias"] = accel_bias
        state["craft"]["imu.gyro_bias"] = gyro_bias
        runtime.set_state_keep_covariance(state)

    preintegrator = TargetNumpy(IMUPreintegrator())
    for k in range(20):
        true_accel, true_gyro = _sample(k, dt)
        measured_accel = true_accel + accel_bias
        measured_gyro = true_gyro + gyro_bias
        raw.predict(dt, u={"imu.accel": measured_accel,
                           "imu.gyro": measured_gyro})
        packet = preintegrator.step(
            dt, accel=measured_accel, gyro=measured_gyro,
            # Deliberately preintegrate at a stale zero-bias reference.
            accel_bias=np.zeros(3), gyro_bias=np.zeros(3))
    packet_filter.predict_preintegrated(packet)
    a = raw.state_dict()["craft"]
    b = packet_filter.state_dict()["craft"]
    np.testing.assert_allclose(b["orientation"], a["orientation"], atol=2e-9)
    np.testing.assert_allclose(b["velocity"], a["velocity"], atol=2e-8)
    np.testing.assert_allclose(b["position"], a["position"], atol=2e-10)


def test_packet_covariance_is_psd_and_enters_ins_covariance():
    preintegrator = TargetNumpy(IMUPreintegrator(
        accel_noise_density=2e-3, gyro_noise_density=3e-4))
    for _ in range(5):
        packet = preintegrator.step(
            .002, accel=(0, 0, 9.81), gyro=(0.1, -0.2, 0.3),
            accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
    C = np.reshape(packet["covariance"], (9, 9), order="F")
    np.testing.assert_allclose(C, C.T, atol=1e-18)
    assert np.linalg.eigvalsh(C).min() >= -1e-18
    assert np.trace(C) > 0.0

    runtime = TargetNumpy(_preintegrated_ins())
    n = runtime.P.shape[0]
    runtime.reset(P=np.zeros((n, n)))
    runtime.predict_preintegrated(packet, Q=np.zeros((n, n)))
    nav_names = ("craft.orientation", "craft.velocity", "craft.position")
    indices = []
    for name in nav_names:
        slot = runtime._spec.slot(name)
        indices.extend(range(slot.tangent_offset,
                             slot.tangent_offset + slot.tangent_dim))
    assert np.trace(runtime.P[np.ix_(indices, indices)]) > 0.0


def test_displaced_imu_packet_retains_boundary_gyro_covariance():
    """The packet/filter joint covariance must include both lever endpoints."""
    world = _world(lever=(0.625, 0.0, 0.0))
    ins = INS(world, imu="craft.imu", sensors=[],
              propagation="preintegrated")
    runtime = TargetNumpy(ins)
    n = runtime.P.shape[0]
    runtime.reset(P=np.zeros((n, n)))
    packet = _single_sample_packet(
        accel=(0.0, 0.0, 9.81),
        gyro=(0.0, 0.0, 0.0),
        end_accel=(0.0, 0.0, 9.81),
        end_gyro=(0.0, 0.0, 0.0),
        dt=0.005,
        accel_noise_sigma=0.01,
        gyro_noise_sigma=0.001,
    )
    runtime.predict_preintegrated(packet, Q=np.zeros((n, n)))
    velocity = runtime._spec.slot("craft.velocity")
    diagonal = np.diag(runtime.P)[
        velocity.tangent_offset:
        velocity.tangent_offset + velocity.tangent_dim
    ]
    accel_variance = (0.01 * 0.005) ** 2
    endpoint_lever_variance = 2.0 * (0.625 * 0.001) ** 2
    np.testing.assert_allclose(
        diagonal,
        (accel_variance,
         accel_variance + endpoint_lever_variance,
         accel_variance + endpoint_lever_variance),
        rtol=2e-5,
        atol=1e-15,
    )
    assert runtime.P_consider is not None
    # The newly carried boundary must remain correlated with lateral craft
    # velocity for the next packet and any same-epoch DVL update.
    assert np.linalg.norm(runtime.P_consider[
        velocity.tangent_offset:
        velocity.tangent_offset + velocity.tangent_dim, :
    ]) > 0.0


def test_preintegrated_runtime_rejects_incomplete_packet_and_collisions():
    runtime = TargetNumpy(_preintegrated_ins())
    with pytest.raises(KeyError, match="duration"):
        runtime.predict_preintegrated({})
    preintegrator = TargetNumpy(IMUPreintegrator())
    packet = preintegrator.step(
        .01, accel=(0, 0, 9.81), gyro=(0, 0, 0),
        accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
    with pytest.raises(ValueError, match="packet-owned"):
        runtime.predict_preintegrated(packet, u={"imu.accel": (0, 0, 9.81)})


def test_truth_backed_analysis_tools_adapt_raw_samples_to_packets():
    world = _world()
    ins = _preintegrated_ins(world)
    observable = observability_trajectory(
        world, dt=0.02, steps=2, estimator=ins)
    consistent = nees(
        world, dt=0.02, steps=2, runs=1, warmup=0, estimator=ins)
    assert observable.sensors == ["craft.model_force.specific_force"]
    assert consistent.samples == 2


def test_one_sample_truth_adapter_refuses_fake_preintegration_endpoint():
    ins = _preintegrated_ins()
    with pytest.raises(ValueError, match="cannot define both boundaries"):
        estimator_inputs(
            ins,
            {},
            reading=lambda _name: np.zeros(3),
            dt=0.02,
        )


def test_noisefit_accepts_recorded_preintegration_packet_traces():
    world = _world()
    ins = _preintegrated_ins(world)
    fit = NoiseFit(
        world,
        noise={"model_force.model_error_x": Prior(mean=0.5, sigma=1.0)},
        estimator=ins,
    )
    preintegrator = TargetNumpy(IMUPreintegrator())
    packet = preintegrator.step(
        .02, accel=(0, 0, 9.81), gyro=(0, 0, 0),
        accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
    K = 2
    traces = {
        full: np.tile(np.atleast_1d(packet[short]), (K, 1))
        for short, full in ins.preintegration_input_map.items()
    }
    _x0, U, Z, _mask, count = fit._window_arrays(
        Window(x0={}, z=traces, dt=.02))
    assert count == K
    assert U.shape == (sum(field.dim for field in ins.sys.input_fields), K)
    assert Z.shape == (3, K)


def test_preintegrator_cpp_roundtrip(tmp_path: Path):
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    eigen_inc = next((p for p in ("/usr/include/eigen3",
                                  "/usr/local/include/eigen3")
                      if Path(p, "Eigen", "Dense").exists()), None)
    if cxx is None or cc is None or eigen_inc is None:
        pytest.skip("C/C++ compiler or Eigen headers not available")

    block = IMUPreintegrator(
        accel_noise_density=2e-3, gyro_noise_density=3e-4)
    result = TargetCpp(block, tmp_path, class_name="Preintegrator")
    harness = tmp_path / "harness.cpp"
    harness.write_text(r'''
#include "preintegrator.hpp"
#include <cstdio>
int main() {
    manta_gen::Preintegrator p;
    manta_gen::Preintegrator::Inputs u;
    u.accel << 0.4, -0.2, 9.81;
    u.gyro << 0.3, -0.1, 0.5;
    u.accel_bias.setZero();
    u.gyro_bias.setZero();
    manta_gen::Preintegrator::Outputs o;
    for (int i = 0; i < 5; ++i) o = p.step(u, 0.002);
    std::printf("%.17g %.17g %.17g %.17g %.17g %.17g\n",
        o.duration, o.sample_count, o.delta_orientation[0],
        o.delta_orientation[1], o.delta_velocity[0], o.covariance[0]);
}
''')
    k_obj, w_obj = tmp_path / "k.o", tmp_path / "w.o"
    commands = (
        [cc, "-c", "-O2", str(result.kernels_c), "-o", str(k_obj)],
        [cxx, "-c", "-std=c++17", "-O2", f"-I{eigen_inc}",
         f"-I{tmp_path}", str(result.wrapper_cpp), "-o", str(w_obj)],
        [cxx, "-std=c++17", "-O2", f"-I{eigen_inc}", f"-I{tmp_path}",
         str(harness), str(w_obj), str(k_obj), "-o", str(tmp_path / "run")],
    )
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True,
                              check=False)
        assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([str(tmp_path / "run")], capture_output=True,
                          text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    cpp = np.fromstring(proc.stdout, sep=" ")

    runtime = TargetNumpy(block)
    for _ in range(5):
        packet = runtime.step(
            .002, accel=(.4, -.2, 9.81), gyro=(.3, -.1, .5),
            accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
    expected = np.array([
        packet["duration"], packet["sample_count"],
        packet["delta_orientation"][0], packet["delta_orientation"][1],
        packet["delta_velocity"][0], packet["covariance"][0],
    ])
    np.testing.assert_allclose(cpp, expected, rtol=1e-12, atol=1e-15)


def test_preintegrated_ins_emits_cpp(tmp_path: Path):
    result = TargetCpp(
        _preintegrated_ins(), tmp_path, class_name="PacketIns")
    assert result.wrapper_hpp.exists()
    header = result.wrapper_hpp.read_text()
    assert "craft_imu_preintegrated_delta_orientation" in header
    assert "craft_imu_preintegrated_covariance" in header
    # The packet span is a kernel input and its invariant is documented on
    # the generated Inputs field a direct C++ caller fills.
    assert "craft_imu_preintegrated_duration" in header
    assert f"// {PREINTEGRATION_DURATION_DOC}" in header


def test_packet_duration_is_a_kernel_input_checked_against_dt():
    ins = _preintegrated_ins()
    assert ins.preintegration_input_map["duration"] == \
        "craft.imu.preintegrated.duration"
    assert "craft.imu.preintegrated.duration" in ins.module().metadata[
        "prediction_inputs"]
    pre = TargetNumpy(IMUPreintegrator())
    for k in range(4):
        accel, gyro = _sample(k, .005)
        packet = pre.step(.005, accel=accel, gyro=gyro,
                          accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0))
    runtime = TargetNumpy(ins)
    u = ins.sys.resolve_u(runtime.preintegrated_inputs(packet))
    predict = ins.module().functions["predict"]
    x0, P0 = runtime.x.copy(), runtime.P.copy()
    C0 = runtime.P_consider
    assert C0 is not None
    consistent, _, _ = predict(
        x0, P0, C0, u, packet["duration"], 0.0)
    assert np.all(np.isfinite(np.asarray(consistent)))
    # The kernel itself (hence every generated backend) refuses a dt that is
    # not the packet span: the navigation state is poisoned, not mis-scaled.
    poisoned, _, _ = predict(
        x0, P0, C0, u, 0.5 * packet["duration"], 0.0)
    assert not np.any(np.isfinite(np.asarray(poisoned)[:10]))
    # The NumPy runtime names the mismatch before running the kernel.
    with pytest.raises(ValueError, match="differs from the preintegrated "
                                         "packet duration"):
        runtime.predict(0.5 * packet["duration"],
                        u=runtime.preintegrated_inputs(packet))
    np.testing.assert_array_equal(runtime.x, x0)
    runtime.predict_preintegrated(packet)
    assert np.all(np.isfinite(runtime.x))
    assert runtime.time == pytest.approx(packet["duration"])


def test_split_rate_desktop_example():
    from examples.vehicles.ins_preintegration import run
    result = run(duration=0.04, imu_rate=500.0, filter_rate=50.0)
    assert result["position_difference_m"] < 1e-10
    assert result["velocity_difference_mps"] < 1e-10
    assert result["quaternion_difference"] < 1e-10
