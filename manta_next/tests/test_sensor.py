"""IMU sensor part — gyro reading via Output declarations (M8)."""

import numpy as np
import pytest

from manta_next import World, Craft
from manta_next.fields import GravityField
from manta_next.parts import IMU, Joint, Mass, Output, Part, PartUpdate, Thruster, Wrench
from manta_next.ir.frames import CraftFrame
from manta_next.ir.types import Vec3


# ---------------------------------------------------------------------------
# Output declaration introspection
# ---------------------------------------------------------------------------

def test_output_decl_introspection():
    imu = IMU("g")
    decls = imu.output_declarations()
    assert set(decls.keys()) == {"gyro", "accel"}
    assert decls["gyro"].shape  == "vec3"
    assert decls["accel"].shape == "vec3"


def test_output_not_in_initial_state():
    """Outputs are per-tick reads, not state slots — they don't appear in
    initial_state(). (Noise slots DO appear, seeded at zero — they're
    user-supplied each tick.)"""
    c = Craft("with_imu")
    c.add(Mass("body", mass=1.0))
    c.add(IMU("g"))
    state = c.initial_state()
    g_keys = [k for k in state if k.startswith("g.")]
    # IMU declares two white Noise channels (gyro_noise, accel_noise)
    # plus two RW channels (gyro_bias, accel_bias) — but the RW
    # channels default to sigma=0 (inert) so neither bias state nor
    # driver shows up in initial_state. Only the two white drivers do.
    assert set(g_keys) == {"g.gyro_noise", "g.accel_noise"}
    # The actual Outputs are NOT in initial_state.
    assert "g.gyro"  not in state
    assert "g.accel" not in state


def test_output_decl_validation():
    with pytest.raises(ValueError, match="shape"):
        Output(shape="bogus")


# ---------------------------------------------------------------------------
# Stationary craft → zero gyro
# ---------------------------------------------------------------------------

def test_stationary_craft_reads_zero_gyro():
    c = Craft("stat")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("g"))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    out = tick(dt=0.001, **state)
    np.testing.assert_allclose(np.array(out["g.gyro"]).ravel(),
                               np.zeros(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Spinning craft → gyro = ω
# ---------------------------------------------------------------------------

def test_spinning_craft_reads_omega():
    """Seed the craft with a known body angular velocity; gyro must
    equal it on the first tick."""
    c = Craft("spin")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("g"))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    # Set body ω along three axes.
    state["angular_velocity"] = np.array([0.7, -0.2, 1.3])
    out = tick(dt=0.001, **state)
    np.testing.assert_allclose(np.array(out["g.gyro"]).ravel(),
                               np.array([0.7, -0.2, 1.3]), atol=1e-12)


def test_gyro_unaffected_by_linear_velocity():
    """An IMU at the body origin reads pure ω — linear velocity does
    not leak into the gyro channel."""
    c = Craft("translating")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("g"))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    state["velocity"] = np.array([5.0, 3.0, -2.0])
    out = tick(dt=0.001, **state)
    np.testing.assert_allclose(np.array(out["g.gyro"]).ravel(),
                               np.zeros(3), atol=1e-12)


# ---------------------------------------------------------------------------
# IMU is dynamically passive
# ---------------------------------------------------------------------------

def test_imu_adds_no_wrench():
    """Adding an IMU to a free-falling craft must not perturb the trajectory
    of an identical IMU-less craft."""
    def make(with_imu: bool):
        c = Craft("c")
        c.add(Mass("body", mass=2.0, moi=(0.5, 0.5, 0.5)))
        if with_imu:
            c.add(IMU("g"))
        return c

    g = (0.0, 0.0, -9.81)
    s1 = make(False); s2 = make(True)
    t1 = s1.compile_tick(gravity_field=GravityField(g=g))
    t2 = s2.compile_tick(gravity_field=GravityField(g=g))

    st1, st2 = s1.initial_state(), s2.initial_state()
    for _ in range(200):
        o1 = t1(dt=0.005, **st1); st1 = {**st1, **o1}
        o2 = t2(dt=0.005, **st2); st2 = {**st2, **o2}

    np.testing.assert_allclose(np.array(st1["position"]).ravel(),
                               np.array(st2["position"]).ravel(),
                               atol=1e-12)
    np.testing.assert_allclose(np.array(st1["velocity"]).ravel(),
                               np.array(st2["velocity"]).ravel(),
                               atol=1e-12)


# ---------------------------------------------------------------------------
# Gyro tracks ω as it changes (reaction-spin scenario)
# ---------------------------------------------------------------------------

def test_gyro_tracks_motor_reaction_spin():
    """Apply a joint torque; the body counter-rotates. The IMU gyro
    tracks the time-evolving ω."""
    c = Craft("reactive")
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    j = Joint("m", mode="saturating", stall_torque=1e9, axis=(0.0, 0.0, 1.0))
    j.add(Mass("m_rotor", mass=1e-6, moi=(0.0, 0.0, 0.01)))
    c.add(j)
    c.add(IMU("g"))

    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    state["m.torque_cmd"] = 0.1   # τ = 0.1 N·m about body z

    gyro_history = []
    for _ in range(500):
        out = tick(dt=0.001, **state)
        gyro_history.append(np.array(out["g.gyro"]).ravel().copy())
        state = {**state, **out}

    # First sample is near zero (ω was 0 at t=0, gyro reads ω at start).
    assert abs(gyro_history[0][2]) < 1e-9
    # x and y stay zero for a pure-z torque.
    np.testing.assert_allclose(gyro_history[-1][:2], np.zeros(2), atol=1e-9)
    # Body I_zz now includes the rotor's MOI (0.05 + 0.01 = 0.06).
    # α_body = -0.1 / 0.06 ≈ -1.667 → ω(0.5s) ≈ -0.833.
    assert np.isclose(gyro_history[-1][2], -0.1 / 0.06 * 0.5, atol=2e-3)


# ---------------------------------------------------------------------------
# World plumbing carries IMU outputs through step()
# ---------------------------------------------------------------------------

def test_imu_output_appears_in_world_step():
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    c = Craft("imu_craft")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("g"))
    w.add_craft(c)

    cw = w.compile()
    state = cw.initial_state()
    state["imu_craft"]["angular_velocity"] = np.array([0.0, 0.0, 2.5])

    state = cw.step(state, dt=0.001)
    np.testing.assert_allclose(np.array(state["imu_craft"]["g.gyro"]).ravel(),
                               np.array([0.0, 0.0, 2.5]), atol=1e-9)


# ---------------------------------------------------------------------------
# Validation: declaring an output and not writing it raises
# ---------------------------------------------------------------------------

def test_missing_output_write_raises():
    class BrokenSensor(Part):
        reading = Output(shape="vec3")

        def update(self, ctx):
            zero = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
            return PartUpdate(wrench=Wrench(force=zero, torque=zero))

    c = Craft("broken")
    c.add(Mass("body", mass=1.0))
    c.add(BrokenSensor("b"))
    with pytest.raises(KeyError, match="not written"):
        c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))


def test_unknown_output_write_raises():
    class UnknownOutput(Part):
        def update(self, ctx):
            zero = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
            return PartUpdate(
                wrench=Wrench(force=zero, torque=zero),
                outputs={"surprise": zero},
            )

    c = Craft("surprise")
    c.add(Mass("body", mass=1.0))
    c.add(UnknownOutput("u"))
    with pytest.raises(KeyError, match="unknown output"):
        c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))


# ---------------------------------------------------------------------------
# IMU accelerometer (post-Newton-Euler phase)
# ---------------------------------------------------------------------------

def test_accel_in_free_fall_reads_zero():
    """A free-falling body's accelerometer reads zero specific force —
    body is accelerating at g, so a_body - g_body = 0. Single tick is
    enough: `ctx.acceleration_body` is the current tick's N-E output
    (placeholder-substituted at compile time), no lag."""
    c = Craft("falling")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("g"))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, -9.81)))
    state = c.initial_state()
    out = tick(dt=0.001, **state)
    accel = np.array(out["g.accel"]).ravel()
    np.testing.assert_allclose(accel, np.zeros(3), atol=1e-6)


def test_accel_zero_gravity_zero_acceleration_reads_zero():
    """No gravity, no commanded force → IMU reads zero specific force."""
    c = Craft("drift")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("g"))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    state["velocity"] = np.array([5.0, 0.0, 0.0])   # constant velocity
    out = tick(dt=0.001, **state)
    np.testing.assert_allclose(np.array(out["g.accel"]).ravel(),
                               np.zeros(3), atol=1e-9)


def test_accel_picks_up_thruster_acceleration():
    """A thruster pushing the body produces non-zero specific force on
    the accelerometer: F/m in the thrust direction, in body frame.
    Single tick — the IMU reads the same-tick a, not a prev-tick echo."""
    c = Craft("powered")
    c.add(Mass("body", mass=2.0, moi=(0.1, 0.1, 0.1), apply_gravity=False))
    c.add(Thruster("t", force=(10.0, 0.0, 0.0)))   # 10 N along +x at throttle=1
    c.add(IMU("g"))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    state["t.throttle"] = 1.0
    out = tick(dt=0.001, **state)
    # F = 10 N, m = 2 kg → a = 5 m/s² along +x in body frame.
    accel = np.array(out["g.accel"]).ravel()
    np.testing.assert_allclose(accel, (5.0, 0.0, 0.0), atol=1e-6)


def test_accel_offset_imu_reads_centripetal_under_rotation():
    """Spinning body, IMU mounted off-axis: framework lifts the
    a_origin lever-arm contribution to the IMU's mount point, so
    the accelerometer reads ω × (ω × r) in body frame. With no
    other forces (no gravity, no thrust), `accel` IS the lever-arm
    term — proves the framework does the lift, not the IMU."""
    c = Craft("spinner")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1), apply_gravity=False))
    c.add(IMU("imu", transform=(1.0, 0.0, 0.0)))
    tick = c.compile_tick(gravity_field=GravityField(g=(0.0, 0.0, 0.0)))
    state = c.initial_state()
    omega_z = 2.0   # rad/s about +z
    state["angular_velocity"] = np.array([0.0, 0.0, omega_z])
    out = tick(dt=0.001, **state)
    # Centripetal acceleration: ω × (ω × r) with r=(1,0,0), ω=(0,0,2):
    #   ω × r = (0,0,2) × (1,0,0) = (0, 2, 0)
    #   ω × (ω × r) = (0,0,2) × (0,2,0) = (-4, 0, 0)
    # Specific force = a_body - g_body = (-4, 0, 0) - 0 = (-4, 0, 0).
    accel = np.array(out["imu.accel"]).ravel()
    np.testing.assert_allclose(accel, (-omega_z ** 2, 0.0, 0.0), atol=1e-6)


def test_imu_noise_sigmas_default_to_zero():
    """Default Noise sigma is 0 on every channel — clean signal,
    `sample_noise(rng)` returns zeros, `noise_R` returns the zero
    matrix."""
    imu = IMU("g")
    assert imu.gyro_noise_sigma  == 0.0
    assert imu.accel_noise_sigma == 0.0


def test_craft_sample_noise_zero_sigma_returns_zeros():
    """With zero sigmas, `craft.sample_noise(rng)` returns zero vectors
    for every declared Noise slot and doesn't consume RNG state."""
    c = Craft("clean")
    c.add(Mass("body", mass=1.0))
    c.add(IMU("g"))
    rng = np.random.default_rng(0)
    samp = c.sample_noise(rng)
    np.testing.assert_array_equal(samp["g.gyro_noise"],  np.zeros(3))
    np.testing.assert_array_equal(samp["g.accel_noise"], np.zeros(3))


def test_craft_sample_noise_nonzero_sigma_returns_scaled_samples():
    """With sigma > 0, sample_noise draws Gaussian samples at the
    requested 1-σ. Sigmas live on the part via `<name>_sigma`
    constructor kwargs."""
    c = Craft("noisy")
    c.add(Mass("body", mass=1.0))
    c.add(IMU("g", gyro_noise_sigma=0.05, accel_noise_sigma=0.2))
    rng = np.random.default_rng(0)
    samples_g, samples_a = [], []
    for _ in range(5000):
        s = c.sample_noise(rng)
        samples_g.append(s["g.gyro_noise"])
        samples_a.append(s["g.accel_noise"])
    sg = np.array(samples_g);  sa = np.array(samples_a)
    np.testing.assert_allclose(sg.std(axis=0), 0.05, rtol=0.05)
    np.testing.assert_allclose(sa.std(axis=0), 0.20, rtol=0.05)


def test_noise_R_is_sigma_squared_identity():
    imu = IMU("g", gyro_noise_sigma=0.01, accel_noise_sigma=0.1)
    np.testing.assert_allclose(imu.noise_R("gyro_noise"),
                                (0.01 ** 2) * np.eye(3))
    np.testing.assert_allclose(imu.noise_R("accel_noise"),
                                (0.1  ** 2) * np.eye(3))
