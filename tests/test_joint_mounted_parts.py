"""RevoluteJoint-mounted sensors and thrusters — the payoff of the unified
kinematic-acceleration pass.

A part riding a RevoluteJoint's rotor sees a moving frame: its thrust direction
rotates with the joint angle, and an accelerometer on the rim picks up the
rotor's centripetal/tangential acceleration that the rigid lever-arm
transfer alone misses. These tests pin both against closed-form values.
"""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField, MagField
from manta.parts import IMU, DragSurface, Magnetometer, Mass, RevoluteJoint, Thruster


def _free_world(craft, **overrides):
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    w.add_craft(craft, **overrides)
    return TargetNumpy(Sim(w))


# ---------------------------------------------------------------------------
# Gimballed thruster: body-frame thrust tracks the joint angle
# ---------------------------------------------------------------------------

def _accel_dir_for_angle(angle: float) -> np.ndarray:
    """One step of a craft whose only force is a thruster on a z-axis joint
    locked at `angle`. Returns the body/world linear acceleration direction."""
    F = 10.0
    c = Craft("gimbal_craft")
    c.add(Mass("bus", mass=1.0, moi=(1.0, 1.0, 1.0)))
    g = RevoluteJoint("gimbal", mode="passive", axis=(0.0, 0.0, 1.0))
    g.add(Mass("rotor", mass=0.001, moi=(1e-4, 1e-4, 1e-4)))  # axisymmetric
    # Thrust along the thruster's own +x. At joint angle θ about z it should
    # point along (cosθ, sinθ, 0) in body coords.
    g.add(Thruster("jet", force=(F, 0.0, 0.0)))
    c.add(g)

    sim = _free_world(c, **{"gimbal.angle": angle, "jet.throttle": 1.0})
    state = sim.step(dt=1e-3)
    v = np.asarray(state["gimbal_craft"]["velocity"]).ravel()
    return v / np.linalg.norm(v)


def test_gimballed_thruster_thrust_rotates_with_joint():
    # angle 0 → +x, angle π/2 → +y, angle π → -x.
    np.testing.assert_allclose(_accel_dir_for_angle(0.0),
                               (1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(_accel_dir_for_angle(np.pi / 2),
                               (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(_accel_dir_for_angle(np.pi),
                               (-1.0, 0.0, 0.0), atol=1e-6)


def test_gimballed_thruster_magnitude_is_angle_independent():
    """The thrust magnitude (hence |a|) is invariant under the gimbal angle —
    rotation preserves length."""
    F, m_total = 10.0, 1.001
    dt = 1e-3
    for angle in (0.3, 1.1, 2.7):
        c = Craft("g")
        c.add(Mass("bus", mass=1.0, moi=(1.0, 1.0, 1.0)))
        j = RevoluteJoint("gim", mode="passive", axis=(0.0, 0.0, 1.0))
        j.add(Mass("rotor", mass=0.001, moi=(1e-4, 1e-4, 1e-4)))
        j.add(Thruster("jet", force=(F, 0.0, 0.0)))
        c.add(j)
        sim = _free_world(c, **{"gim.angle": angle, "jet.throttle": 1.0})
        state = sim.step(dt=dt)
        speed = np.linalg.norm(np.asarray(state["g"]["velocity"]).ravel())
        np.testing.assert_allclose(speed, F / m_total * dt, rtol=1e-3)


# ---------------------------------------------------------------------------
# Rotor-mounted IMU: reads the rim's centripetal acceleration + spin rate
# ---------------------------------------------------------------------------

def test_rotor_imu_reads_centripetal_and_spin():
    """A spinning flywheel with an IMU on the rim (offset d from the z axis,
    rate ω₀). At angle 0 the sensor frame = body frame, so:
      * gyro.z   = ω₀                (rotor absolute spin about z)
      * accel.x ≈ −ω₀²·d             (centripetal, pointing to the axis)
    The centripetal term is exactly the joint-relative acceleration the old
    rigid lever-arm pass dropped; without the unified pass accel.x would read
    ~0."""
    omega0, d = 12.0, 0.5
    c = Craft("fly")
    # Heavy axisymmetric hub so the body barely recoils over one tick and the
    # joint Coriolis torque (∝ Ix−Iy) vanishes → no tangential term.
    c.add(Mass("hub", mass=1000.0, moi=(50.0, 50.0, 50.0)))
    wheel = RevoluteJoint("wheel", mode="passive", axis=(0.0, 0.0, 1.0))
    wheel.add(Mass("rim", mass=1.0, moi=(0.01, 0.01, 0.01)))  # on-axis inertia
    wheel.add(IMU("imu", mount_offset=(d, 0.0, 0.0)))            # offset on rim
    c.add(wheel)

    sim = _free_world(c, **{"wheel.rate": omega0})
    sim.step(dt=1e-4)

    gyro  = np.asarray(sim.outputs()["fly"]["imu.gyro"]).ravel()
    accel = np.asarray(sim.outputs()["fly"]["imu.accel"]).ravel()

    np.testing.assert_allclose(gyro, (0.0, 0.0, omega0), atol=1e-6)
    # Centripetal acceleration toward the axis; small body recoil tolerated.
    np.testing.assert_allclose(accel[0], -omega0**2 * d, rtol=2e-3)
    np.testing.assert_allclose(accel[1:], (0.0, 0.0), atol=1e-3)


def test_rotor_imu_centripetal_scales_with_rate_squared():
    """Doubling the spin rate quadruples the centripetal reading."""
    d = 0.5

    def accel_x(omega0):
        c = Craft("fly")
        c.add(Mass("hub", mass=1000.0, moi=(50.0, 50.0, 50.0)))
        wheel = RevoluteJoint("wheel", mode="passive", axis=(0.0, 0.0, 1.0))
        wheel.add(Mass("rim", mass=1.0, moi=(0.01, 0.01, 0.01)))
        wheel.add(IMU("imu", mount_offset=(d, 0.0, 0.0)))
        c.add(wheel)
        sim = _free_world(c, **{"wheel.rate": omega0})
        sim.step(dt=1e-4)
        return float(np.asarray(sim.outputs()["fly"]["imu.accel"]).ravel()[0])

    np.testing.assert_allclose(accel_x(20.0), 4.0 * accel_x(10.0), rtol=3e-3)


# ---------------------------------------------------------------------------
# Rotor-mounted magnetometer: reads the field in its own (spinning) frame
# ---------------------------------------------------------------------------

def test_rotor_magnetometer_reads_in_sensor_frame():
    """A magnetometer on a z-axis joint locked at θ reads the body-frame field
    rotated into its own frame: B_sensor = R_z(θ).T · B_body. With the craft
    unrotated and a uniform B=(1,0,0), at θ=π/2 the sensor reads (0,-1,0)."""
    c = Craft("compass")
    c.add(Mass("bus", mass=1.0, moi=(1.0, 1.0, 1.0)))
    j = RevoluteJoint("yaw", mode="passive", axis=(0.0, 0.0, 1.0))
    j.add(Mass("rotor", mass=0.01, moi=(1e-3, 1e-3, 1e-3)))
    j.add(Magnetometer("mag"))
    c.add(j)

    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    w.add_field(MagField().add_uniform((1.0, 0.0, 0.0)))
    w.add_craft(c, **{"yaw.angle": np.pi / 2})
    sim = TargetNumpy(Sim(w))
    sim.step(dt=1e-3)

    np.testing.assert_allclose(
        np.asarray(sim.outputs()["compass"]["mag.B"]).ravel(),
        (0.0, -1.0, 0.0), atol=1e-9)


# ---------------------------------------------------------------------------
# A previously-blocked force part (DragSurface) on a rotor: its drag tracks
# the rotor frame, with no part-side frame handling.
# ---------------------------------------------------------------------------

def test_rotor_dragsurface_drag_tracks_rotor_frame():
    """A linear-drag fin on a z-axis joint locked at θ=π/2, with the craft
    translating at +x through still fluid. The fin's drag tensor is in its
    own frame; at θ=π/2 the body-frame drag is the θ-rotated tensor acting on
    the body-frame relative wind. With A_1 = -c·I (isotropic) the drag is
    -ρ·c·v_rel regardless of rotor angle, so it opposes +x motion → a_x<0,
    and stays purely along x (no spurious lateral force)."""
    rho, c, V0, M = 1.0, 0.5, 3.0, 50.0
    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
         .add_field(FluidField().add_uniform(density=rho)))
    craft = Craft("sub")
    craft.add(Mass("hull", mass=M, moi=(5.0, 5.0, 5.0)))
    j = RevoluteJoint("yaw", mode="passive", axis=(0.0, 0.0, 1.0))
    j.add(Mass("rotor", mass=0.01, moi=(1e-3, 1e-3, 1e-3)))
    j.add(DragSurface("fin", force=(-c, -c, -c)))   # isotropic linear drag
    craft.add(j)
    w.add_craft(craft, **{"yaw.angle": np.pi / 2}, velocity=(V0, 0.0, 0.0))
    sim = TargetNumpy(Sim(w))
    state = sim.step(dt=1e-3)

    v = np.asarray(state["sub"]["velocity"]).ravel()
    # Isotropic drag opposes motion: decelerates +x, no lateral kick.
    assert v[0] < V0
    np.testing.assert_allclose(v[1:], (0.0, 0.0), atol=1e-9)


# ---------------------------------------------------------------------------
# Static mount orientation
# ---------------------------------------------------------------------------

def test_rotated_thruster_pushes_along_its_rotated_axis():
    """A thruster declaring +x thrust, mounted yawed 90°, pushes the
    craft along +y. Before static mount rotations existed, the only way
    to say this was to hand-rotate the thrust vector in the config —
    which duplicates the geometry and silently diverges from whatever
    the mixer believes."""
    import numpy as np

    from manta import Craft, Sim, TargetNumpy, World
    from manta.fields import GravityField
    from manta.parts import Mass, Thruster

    half = np.sqrt(0.5)
    c = Craft("boat")
    c.add(Mass("body", mass=10.0, moi=(1.0, 1.0, 1.0)))
    c.add(Thruster("t", force=(10.0, 0.0, 0.0),
                   mount_orientation=(half, 0.0, 0.0, half)))    # +90° yaw
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, **{"t.throttle": 1.0})
    sim = TargetNumpy(Sim(w))
    for _ in range(100):
        sim.step(0.001)

    # 10 N on 10 kg for 0.1 s.
    v = np.asarray(sim.state["boat"]["velocity"], dtype=float)
    assert v[1] == pytest.approx(0.1, abs=1e-4), "thrust should be along +y"
    assert abs(v[0]) < 1e-9, "nothing should remain along +x"


def test_rotated_sensor_reads_its_own_frame():
    """A gyro mounted on its side reports the craft's rate in the
    SENSOR's axes, not the hull's. This is the whole point of a static
    mount rotation for sensors: the measurement function should not
    have to undo a frame the framework knows about."""
    import numpy as np

    from manta import Craft, Sim, TargetNumpy, World
    from manta.fields import GravityField
    from manta.parts import IMU, Mass

    half = np.sqrt(0.5)
    c = Craft("d")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(IMU("upright", gyro_noise_sigma=0.0))
    # Rolled +90° about x, so the sensor's own +y points along the
    # hull's +z — which is the axis the craft is turning about.
    c.add(IMU("rolled", gyro_noise_sigma=0.0,
              mount_orientation=(half, half, 0.0, 0.0)))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, angular_velocity=(0.0, 0.0, 2.0))       # yaw rate
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)

    up = np.asarray(sim.reading("upright.gyro"), dtype=float)
    rolled = np.asarray(sim.reading("rolled.gyro"), dtype=float)
    np.testing.assert_allclose(up, [0.0, 0.0, 2.0], atol=1e-6)
    # The same physical rotation, read on the rolled sensor's own +y.
    np.testing.assert_allclose(rolled, [0.0, 2.0, 0.0], atol=1e-6)


def test_mount_orientation_composes_through_a_joint():
    """A rotated part on a rotating joint composes both — the joint
    angle and the static mount, in that order."""
    import numpy as np

    from manta import Craft, Sim, TargetNumpy, World
    from manta.fields import GravityField
    from manta.parts import IMU, Mass, RevoluteJoint

    half = np.sqrt(0.5)
    c = Craft("rig")
    c.add(Mass("body", mass=50.0, moi=(5.0, 5.0, 5.0)))
    j = c.add(RevoluteJoint("j", axis=(0.0, 0.0, 1.0), mode="passive"))
    j.add(Mass("rotor", mass=0.5, moi=(0.01, 0.01, 0.02)))
    j.add(IMU("on_rotor", gyro_noise_sigma=0.0,
              mount_orientation=(half, half, 0.0, 0.0)))          # rolled +90°
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, angular_velocity=(0.0, 0.0, 1.0))
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)

    # Body spins about +z; the rolled sensor reads it on its own +y,
    # exactly as it would on the hull — the joint sits at angle 0 here,
    # so the mount rotation is what does the work, composed on top of
    # the joint's (identity) rotation rather than instead of it.
    g = np.asarray(sim.reading("on_rotor.gyro"), dtype=float)
    np.testing.assert_allclose(g, [0.0, 1.0, 0.0], atol=1e-6)


def test_root_mount_pose_is_locked_to_identity():
    """A rotated root would silently redefine what "body frame" means
    for every state, sensor and wrench on the craft."""
    from manta.parts.base import RootPart

    r = RootPart("root")
    assert r.mounted_upright
    assert r.mount_offset == (0.0, 0.0, 0.0)


def test_orientation_is_validated_and_normalized():
    from manta.parts import Mass

    m = Mass("b", mass=1.0, mount_orientation=(2.0, 0.0, 0.0, 0.0))
    assert m.mount_orientation == pytest.approx((1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="non-zero quaternion"):
        Mass("b", mass=1.0, mount_orientation=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="wxyz quaternion"):
        Mass("b", mass=1.0, mount_orientation=(1.0, 0.0, 0.0))
