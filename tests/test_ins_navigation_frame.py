"""Independent Cartesian oracles for fixed planet-attached INS frames."""

from dataclasses import FrozenInstanceError, replace

import casadi as ca
import numpy as np
import pytest

from manta import INS, Craft, IMUPreintegrator, NavigationFrame, TargetNumpy, World
from manta.fields import GravityField
from manta.ir._rotation import (
    quat_mul_np,
    quat_to_rotmat,
    so3_exp_np,
)
from manta.parts import IMU, Mass, VelocitySensor

SPIN = 7.2921159e-5


def frame_at(latitude=37.78, longitude=-122.4, *, spin=SPIN, convention="effective"):
    # Independent ECEF-axis projection; the estimator accepts no latitude.
    phi, lam = np.radians([latitude, longitude])
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    north = np.cross(up, east)
    axes = np.column_stack([east, north, up])
    e2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)
    radius = 6378137 / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    anchor = radius * np.array(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), (1 - e2) * np.sin(phi)]
    )
    return NavigationFrame(
        frame_id="test/site",
        epoch="1",
        angular_velocity=axes.T @ np.array([0, 0, spin]),
        origin_from_rotation_center=axes.T @ anchor,
        gravity_convention=convention,
    )


def make_ins(
    frame=None,
    *,
    propagation="raw",
    q=(1, 0, 0, 0),
    lever=(0, 0, 0),
    mount=(1, 0, 0, 0),
    gravity=(0, 0, -9.81),
    noisy=False,
):
    craft = Craft("craft")
    craft.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    craft.add(
        IMU(
            "imu",
            mount_offset=lever,
            mount_orientation=mount,
            gyro_noise_sigma=1e-6 if noisy else 0,
            accel_noise_sigma=1e-4 if noisy else 0,
        )
    )
    craft.add(VelocitySensor("dvl", velocity_noise_sigma=0.001))
    world = World("rotating_ins").add_field(GravityField(g=gravity))
    world.add_craft(craft, orientation=q)
    return INS(
        world,
        imu="imu",
        sensors=["dvl.velocity"],
        navigation_frame=frame,
        propagation=propagation,
    )


def rotation(q):
    return np.asarray(quat_to_rotmat(ca.DM(q)))


def packet(samples, dt, *, noisy=False):
    pre = TargetNumpy(
        IMUPreintegrator(
            gyro_noise_density=1e-6 * np.sqrt(dt) if noisy else 0,
            accel_noise_density=1e-4 * np.sqrt(dt) if noisy else 0,
        )
    )
    for a, w in samples:
        result = pre.step(
            dt, accel=a, gyro=w, accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0)
        )
    return result


def test_frame_immutable_validated_and_identity_is_hashed():
    frame = frame_at()
    with pytest.raises(FrozenInstanceError):
        frame.epoch = "other"
    for change in (
        {"angular_velocity": (0, float("nan"), 0)},
        {"frame_id": ""},
        {"gravity_convention": "unknown"},
        {"epoch": 0},
    ):
        with pytest.raises(ValueError):
            replace(frame, **change)
    a = make_ins(frame).module()
    b = make_ins(replace(frame, epoch="2")).module()
    assert a.artifact_id != b.artifact_id
    assert (
        a.metadata["navigation_frame"]["frame_definition"]
        == "fixed_planet_attached_cartesian"
    )


@pytest.mark.parametrize("latitude", [0, 37.78, 60, 89.9, -37.78, -60])
@pytest.mark.parametrize("heading", [0, 90, 180, 233])
@pytest.mark.parametrize("propagation", ["raw", "preintegrated"])
def test_stationary_earth_fixed_attitude_and_velocity(latitude, heading, propagation):
    frame = frame_at(latitude)
    q = quat_mul_np(
        so3_exp_np(np.array([0, 0, np.radians(heading)])),
        so3_exp_np(np.array([0.12, -0.08, 0])),
    )
    mount = so3_exp_np(np.array([0.1, 0.2, -0.3]))
    R = rotation(q) @ rotation(mount)
    accel = R.T @ np.array([0, 0, 9.81])
    gyro = R.T @ np.array(frame.angular_velocity)
    ins = make_ins(
        frame, propagation=propagation, q=q, lever=(0.3, -0.2, 0.1), mount=mount
    )
    runtime = TargetNumpy(ins)
    if propagation == "raw":
        runtime.predict(0.01, u={"imu.accel": accel, "imu.gyro": gyro})
    else:
        runtime.predict_preintegrated(packet([(accel, gyro)] * 10, 0.01))
    state = runtime.state_dict()["craft"]
    np.testing.assert_allclose(state["orientation"], q, atol=2e-14)
    np.testing.assert_allclose(state["velocity"], 0, atol=2e-14)
    np.testing.assert_allclose(state["position"], 0, atol=2e-14)


def test_centrifugal_acceleration_included_exactly_once():
    frame = frame_at(convention="gravitation")
    w = np.array(frame.angular_velocity)
    r = np.array(frame.origin_from_rotation_center)
    grav = np.array([0, 0, -9.81])
    effective = grav - np.cross(w, np.cross(w, r))
    runtime = TargetNumpy(make_ins(frame, gravity=grav))
    runtime.predict(0.1, u={"imu.accel": -effective, "imu.gyro": w})
    np.testing.assert_allclose(runtime.state_dict()["craft"]["velocity"], 0, atol=1e-14)


@pytest.mark.parametrize("velocity", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_coriolis_sign_and_factor_two(velocity):
    frame = frame_at()
    w = np.array(frame.angular_velocity)
    runtime = TargetNumpy(make_ins(frame))
    runtime.reset(state={"craft": {"velocity": velocity}})
    dt = 0.01
    runtime.predict(dt, u={"imu.accel": (0, 0, 9.81), "imu.gyro": w})
    actual = runtime.state_dict()["craft"]["velocity"]
    expected = np.array(velocity) - 2 * np.cross(w, velocity) * dt
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    assert np.linalg.norm(actual) == pytest.approx(np.linalg.norm(velocity), abs=1e-14)


@pytest.mark.parametrize("propagation", ["raw", "preintegrated"])
def test_zero_rotation_recovers_original_equations(propagation):
    reference = TargetNumpy(make_ins(propagation=propagation))
    rotating = TargetNumpy(make_ins(frame_at(spin=0), propagation=propagation))
    for obj in [reference, rotating]:
        if propagation == "raw":
            obj.predict(
                0.02, u={"imu.accel": (0.2, 0.3, 9.8), "imu.gyro": (0.1, -0.2, 0.3)}
            )
        else:
            obj.predict_preintegrated(
                packet([((0.2, 0.3, 9.8), (0.1, -0.2, 0.3))] * 4, 0.005)
            )
    np.testing.assert_allclose(reference.x, rotating.x, atol=1e-14)
    np.testing.assert_allclose(reference.P, rotating.P, atol=1e-14)


def test_raw_one_packet_state_and_covariance():
    frame = frame_at()
    raw = TargetNumpy(make_ins(frame, noisy=True))
    pre = TargetNumpy(make_ins(frame, propagation="preintegrated", noisy=True))
    a = np.array([0.1, 0.2, 9.81])
    w = np.array(frame.angular_velocity) + (0.01, 0.02, 0.03)
    p = packet([(a, w)], 0.005, noisy=True)
    raw.predict(0.005, u={"imu.accel": a, "imu.gyro": w})
    pre.predict_preintegrated(p)
    np.testing.assert_allclose(raw.x, pre.x, atol=2e-14)
    np.testing.assert_allclose(raw.P, pre.P, atol=2e-12)


def test_variable_sample_timing_moments():
    runtime = TargetNumpy(IMUPreintegrator())
    durations = np.array([0.01, 0.02, 0.003, 0.006])
    starts = np.r_[0, np.cumsum(durations)[:-1]]
    T = durations.sum()
    for dt in durations:
        p = runtime.step(
            dt,
            accel=(0, 0, 0),
            gyro=(0, 0, 0),
            accel_bias=(0, 0, 0),
            gyro_bias=(0, 0, 0),
        )
    for j in range(4):
        assert p["velocity_time_moments"][j] == pytest.approx(
            np.sum(durations * starts**j)
        )
        assert p["position_time_moments"][j] == pytest.approx(
            np.sum(durations * (T - starts - durations / 2) * starts**j)
        )


def test_rotating_earth_ordinary_imu_is_an_independent_truth_source():
    from manta import Sim
    from manta.ir.frames import WorldFrame
    from manta.ir.types import Vec3
    from manta.parts import Thruster
    from manta.planets import Earth

    earth = Earth(
        include_j2=True,
        dipole_moment=0,
        water_density=0,
        air_density=0,
        surface_collision=False,
    )
    phi = np.radians(37.78)
    e2 = earth.FLATTENING * (2 - earth.FLATTENING)
    N = earth.R_EQ / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    anchor = np.array([N * np.cos(phi), 0, N * (1 - e2) * np.sin(phi)])
    scene = earth.scene_at(tuple(anchor), heading=-np.pi / 2)
    initial = scene.at_rest()
    R = rotation(initial["orientation"])
    # Resolved Earth field contains point-mass and J2 gravitation. Sample it
    # independently of the INS helper, then support the craft's orbital accel.
    probe = Craft("probe")
    probe.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    world = World("earth_probe").add_planet(earth)
    world.add_craft(probe, **initial)
    resolved = Sim(world).world
    field = next(f for f in resolved.fields if isinstance(f, GravityField))
    g_expr = field.value_at_sym(Vec3[WorldFrame].constant(tuple(anchor)), 0).mx
    gravity = np.asarray(
        ca.Function("gravity_at_anchor", [], [g_expr])()["o0"]
    ).reshape(3)
    spin = earth.omega_vec_world()
    specific = R.T @ (np.cross(spin, np.cross(spin, anchor)) - gravity)
    craft = Craft("craft")
    craft.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    craft.add(IMU("imu"))
    craft.add(Thruster("support", force=tuple(specific)))
    truth = World("earth_oracle").add_planet(
        Earth(
            include_j2=True,
            dipole_moment=0,
            water_density=0,
            air_density=0,
            surface_collision=False,
        )
    )
    truth.add_craft(craft, **initial)
    sim = TargetNumpy(Sim(truth))
    sim.step(0.001, u={"support.throttle": 1.0})
    gyro = np.asarray(sim.reading("craft.imu.gyro"))
    accel = np.asarray(sim.reading("craft.imu.accel"))
    np.testing.assert_allclose(gyro, R.T @ spin, atol=2e-15)
    np.testing.assert_allclose(accel, specific, atol=1e-10)
    frame = NavigationFrame(
        frame_id="earth/oracle",
        epoch="1",
        angular_velocity=R.T @ spin,
        origin_from_rotation_center=R.T @ anchor,
        gravity_convention="gravitation",
    )
    ins = TargetNumpy(make_ins(frame, gravity=R.T @ gravity))
    ins.predict(0.001, u={"imu.accel": accel, "imu.gyro": gyro})
    np.testing.assert_allclose(ins.state_dict()["craft"]["velocity"], 0, atol=1e-12)


@pytest.mark.parametrize("propagation", ["raw", "preintegrated"])
def test_generated_native_matches_interpreted(propagation, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    frame = frame_at(-60)
    ins = make_ins(frame, propagation=propagation, noisy=True)
    native = TargetNumpy(ins, compile=True, max_instructions=20000)
    reference = TargetNumpy(ins)
    a = np.array([0.1, 0.2, 9.81])
    w = np.array(frame.angular_velocity) + (0.01, 0.02, 0.03)
    p = packet([(a, w)] * 4, 0.005, noisy=True)
    for runtime in [native, reference]:
        for _ in range(10):
            if propagation == "raw":
                runtime.predict(0.005, u={"imu.accel": a, "imu.gyro": w})
            else:
                runtime.predict_preintegrated(p)
    np.testing.assert_allclose(native.x, reference.x, atol=2e-14)
    np.testing.assert_allclose(native.P, reference.P, atol=2e-12)


def test_bad_timing_moments_and_overlong_packets_fail():
    frame = frame_at()
    for corruption in ["moments", "duration"]:
        runtime = TargetNumpy(make_ins(frame, propagation="preintegrated"))
        dt = 100 if corruption == "duration" else 0.01
        p = packet([((0, 0, 9.81), frame.angular_velocity)], dt)
        if corruption == "moments":
            p["velocity_time_moments"][0] = 0
        with pytest.raises((ValueError, RuntimeError), match="finite|NaN|non-finite"):
            runtime.predict_preintegrated(p)


@pytest.mark.parametrize("propagation", ["raw", "preintegrated"])
def test_rotating_kernels_remain_sx_expandable(propagation):
    module = make_ins(frame_at(), propagation=propagation).module()
    for function in module.functions.values():
        assert function.expand().n_in() == function.n_in()


def test_split_rate_motion_and_covariance_refine_toward_raw():
    frame = frame_at()
    dt = 0.002

    def run(stride):
        raw = TargetNumpy(make_ins(frame, noisy=True))
        pre = TargetNumpy(make_ins(frame, propagation="preintegrated", noisy=True))
        accumulator = TargetNumpy(
            IMUPreintegrator(
                gyro_noise_density=1e-6 * np.sqrt(dt),
                accel_noise_density=1e-4 * np.sqrt(dt),
            )
        )
        for k in range(100):
            t = k * dt
            a = np.array([0.1 * np.sin(t * 2), 0.2 * np.cos(t * 3), 9.81])
            w = np.array(frame.angular_velocity) + (0.1, 0.2, 0.3)
            raw.predict(dt, u={"imu.accel": a, "imu.gyro": w})
            p = accumulator.step(
                dt, accel=a, gyro=w, accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0)
            )
            if (k + 1) % stride == 0:
                pre.predict_preintegrated(p)
                accumulator.reset()
        return np.linalg.norm(raw.x - pre.x), np.linalg.norm(raw.P - pre.P)

    fine = run(5)
    coarse = run(20)
    assert fine[0] < coarse[0]
    assert fine[1] < coarse[1]
    assert fine[0] < 1e-6
    assert fine[1] < 1e-5
