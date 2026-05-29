"""DVL + Magnetometer + MagField parity tests.

The point: with the standard Field + Disturbance machinery and the
established sensor-Output pattern, three new pieces (DVL, MagField,
Magnetometer) drop in with no special-case code. These tests verify
they actually work.
"""

import casadi as ca
import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import DipoleMag, MagField, UniformMag, GravityField
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import DVL, Magnetometer, Mass


# ---------------------------------------------------------------------------
# DVL
# ---------------------------------------------------------------------------

def test_dvl_stationary_craft_reads_zero():
    c = Craft("dvl_test")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DVL("d"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()
    state = sim.step(state, dt=0.001)
    np.testing.assert_allclose(np.array(state["dvl_test"]["d.velocity"]).ravel(),
                               np.zeros(3), atol=1e-12)


def test_dvl_moving_craft_reads_anchor_velocity_when_unrotated():
    """Identity orientation: body velocity = anchor velocity."""
    c = Craft("dvl_move")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DVL("d"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c, velocity=np.array([1.5, -0.3, 2.0]))
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()
    state = sim.step(state, dt=0.001)
    np.testing.assert_allclose(np.array(state["dvl_move"]["d.velocity"]).ravel(),
                               (1.5, -0.3, 2.0), atol=1e-12)


def test_dvl_rotated_craft_reads_rotated_velocity():
    """Craft rotated 90° about z: anchor +x velocity reads as body +y
    velocity (since body +x points to anchor -y under that rotation,
    inverse rotation maps anchor +x to body -y... actually let me work
    this out: q = (cos π/4, 0, 0, sin π/4) rotates anchor +x to anchor +y
    in body frame... reading body-frame velocity means R^T · v_anchor.
    R is the world-from-body rotation; R · body_x = anchor_y; R^T ·
    anchor_x = body... actually for +90° about z, R sends body +x → +y,
    so R^T sends anchor +x → body -y. Wait, let me re-check.

    With q = exp(z·π/4): R is +90° about z. R · body_x = anchor_+y; so
    R^T · anchor_+x = body_−y. Velocity along anchor +x reads as -1 in
    body +y direction.
    """
    c = Craft("dvl_rot")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DVL("d"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c,
                orientation=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                velocity=np.array([1.0, 0.0, 0.0]))
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()
    state = sim.step(state, dt=0.001)
    # body velocity = R^T · anchor velocity.
    # R^T · (1,0,0) for +90° about z: R = [[0,-1,0],[1,0,0],[0,0,1]]
    # so R^T = [[0,1,0],[-1,0,0],[0,0,1]] and R^T·(1,0,0) = (0, -1, 0).
    np.testing.assert_allclose(np.array(state["dvl_rot"]["d.velocity"]).ravel(),
                               (0.0, -1.0, 0.0), atol=1e-9)


# ---------------------------------------------------------------------------
# MagField
# ---------------------------------------------------------------------------

def _eval_mag_at(field, point_xyz):
    p = Vec3[WorldFrame].constant(point_xyz)
    v = field.state_at_sym(p, 0.0)
    return np.asarray(ca.evalf(v._mx)).ravel()


def test_uniform_mag_is_position_independent():
    mf = MagField()
    mf.add(UniformMag((3e-5, 0.0, -4.5e-5)))   # mid-lat Earth field
    for q in [(0, 0, 0), (10, -5, 100), (-1e6, 0, 0)]:
        np.testing.assert_allclose(_eval_mag_at(mf, q),
                                   (3e-5, 0.0, -4.5e-5), atol=1e-12)


def test_dipole_mag_falls_off_as_one_over_r_cubed():
    """Sample along the dipole axis at two distances; magnitudes scale
    as 1/r³. For a moment m along +z, on the +z axis, B = (2 μ₀/4π·m)/r³."""
    mf = MagField()
    moment = (0.0, 0.0, 8e22)        # Earth's dipole moment (A·m²)
    mf.add(DipoleMag(position=(0, 0, 0), moment=moment))

    # Two distances along +z.
    r1 = 1e7   # 10 000 km
    r2 = 2e7   # 20 000 km
    B1 = _eval_mag_at(mf, (0, 0, r1))
    B2 = _eval_mag_at(mf, (0, 0, r2))
    # Magnitudes: 8× smaller at 2× distance (1/r³ scaling).
    assert np.isclose(np.linalg.norm(B1) / np.linalg.norm(B2), 8.0, rtol=1e-9)


def test_dipole_mag_axial_value_matches_formula():
    """On the dipole's axis, B_axial = μ₀/(4π) · 2m/r³ (vector along
    the moment direction)."""
    mf = MagField()
    m = 1.0e23
    mf.add(DipoleMag(position=(0, 0, 0), moment=(0.0, 0.0, m), eps=0.0))
    r = 1e7
    B = _eval_mag_at(mf, (0, 0, r))
    expected = 1e-7 * 2 * m / r**3
    np.testing.assert_allclose(B, (0.0, 0.0, expected), rtol=1e-9)


# ---------------------------------------------------------------------------
# Magnetometer
# ---------------------------------------------------------------------------

def test_magnetometer_reads_uniform_field_when_unrotated():
    c = Craft("mag_craft")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Magnetometer("m"))
    w = World().add_field(MagField().add_uniform((1.0, 2.0, 3.0)))
    w.add_craft(c)
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    state = cw.step(state, dt=0.001)
    np.testing.assert_allclose(
        np.array(state["mag_craft"]["m.B"]).ravel(),
        (1.0, 2.0, 3.0), atol=1e-12)


def test_magnetometer_reads_rotated_field_under_craft_rotation():
    """Same anchor field, craft rotated 90° about z. Body-frame reading
    is R^T · B_world."""
    c = Craft("mag_rot")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Magnetometer("m"))
    w = World().add_field(MagField().add_uniform((1.0, 0.0, 0.0)))
    w.add_craft(c, orientation=(np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    out = cw.step(state, dt=0.001)
    # R^T · (1,0,0) with R = +90° about z → (0, -1, 0) in body frame.
    np.testing.assert_allclose(
        np.array(out["mag_rot"]["m.B"]).ravel(),
        (0.0, -1.0, 0.0), atol=1e-9)


def test_magnetometer_with_no_field_reads_zero():
    c = Craft("nomag")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Magnetometer("m"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()
    state = sim.step(state, dt=0.001)
    np.testing.assert_allclose(np.array(state["nomag"]["m.B"]).ravel(),
                               np.zeros(3), atol=1e-12)


def test_magnetometer_picks_up_dipole_at_local_position():
    """Place a strong dipole at the world origin, drop a craft 1e7 m
    above it on +z. The magnetometer reads the +z component of the
    on-axis dipole field, rotated into body frame (identity here)."""
    mf = MagField()
    m = 1.0e23
    mf.add(DipoleMag(position=(0, 0, 0), moment=(0.0, 0.0, m), eps=0.0))

    w = World().add_field(mf)
    c = Craft("orbiter")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Magnetometer("m"))
    w.add_craft(c, position=(0, 0, 1e7))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    out   = cw.step(state, dt=0.001)
    expected_z = 1e-7 * 2 * m / (1e7) ** 3
    np.testing.assert_allclose(
        np.array(out["orbiter"]["m.B"]).ravel(),
        (0.0, 0.0, expected_z), rtol=1e-9, atol=1e-15)


def test_magnetometer_samples_at_single_mount_offset():
    """Regression: ctx.position already includes the part transform, so the
    magnetometer must NOT re-add it. A sensor at transform=(0,0,h) on a craft
    at z=Z must read the same field as a transform=0 sensor at z=Z+h — in a
    position-varying (dipole) field. The old double-offset bug sampled z=Z+2h."""
    m, Z, h = 1.0e23, 1.0e7, 2.0e6

    def reading(craft_z, offset):
        mf = MagField()
        mf.add(DipoleMag(position=(0, 0, 0), moment=(0.0, 0.0, m), eps=0.0))
        c = Craft("c")
        c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        c.add(Magnetometer("mag", transform=(0.0, 0.0, offset)))
        w = World().add_field(mf)
        w.add_craft(c, position=(0, 0, craft_z))
        sim = TargetNumpy(Sim(w))
        out = sim.step(sim.initial_state(), dt=1e-3)
        return np.array(out["c"]["mag.B"]).ravel()

    offset_sensor = reading(Z, h)         # mounts at Z, offset h → samples Z+h
    flush_sensor  = reading(Z + h, 0.0)   # mounts at Z+h, no offset
    np.testing.assert_allclose(offset_sensor, flush_sensor, rtol=1e-9)
    # And it differs from the buggy double-offset sample at Z+2h.
    double = reading(Z + 2 * h, 0.0)
    assert not np.allclose(offset_sensor, double, rtol=1e-6)
