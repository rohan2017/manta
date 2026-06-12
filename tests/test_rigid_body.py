"""6-DOF rigid-body dynamics: orientation, MOI aggregation, angular
momentum, and parallel-axis lifts."""

import math

import numpy as np

from manta import Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField
from manta.parts import Mass


# ---------------------------------------------------------------------------
# Inertial aggregation
# ---------------------------------------------------------------------------

def test_aggregate_mass_and_com():
    """Two point masses at ±r along x: COM should be at 0, MOI about origin
    should be 2·m·r² about y and z, 0 about x."""
    r = 0.5
    m = 1.0
    c = Craft("dumbbell")
    c.add(Mass("a", mass=m, transform=(+r, 0.0, 0.0)))
    c.add(Mass("b", mass=m, transform=(-r, 0.0, 0.0)))

    inertials = c.aggregate_inertials()
    assert inertials["m_total"] == 2 * m
    assert np.allclose(inertials["com"], [0.0, 0.0, 0.0])
    # I about origin:
    #   point mass at (r, 0, 0): contribution m·(|r|²·I − r·r^T) = m·diag(0, r², r²)
    # Two of them stacked → diag(0, 2·m·r², 2·m·r²).
    expected = np.diag([0.0, 2 * m * r * r, 2 * m * r * r])
    assert np.allclose(inertials["I_origin"], expected)
    # COM is at origin → I_com == I_origin.
    assert np.allclose(inertials["I_com"], expected)


def test_aggregate_offset_com():
    """Mass at (r, 0, 0): COM = (r, 0, 0). I_com (about COM) = 0 for a
    point mass. I_origin = m·r² · diag(0, 1, 1)."""
    r = 1.0
    m = 2.0
    c = Craft("single_offset")
    c.add(Mass("body", mass=m, transform=(r, 0.0, 0.0)))

    inertials = c.aggregate_inertials()
    assert np.allclose(inertials["com"], [r, 0.0, 0.0])
    assert np.allclose(inertials["I_com"], np.zeros((3, 3)), atol=1e-12)
    assert np.allclose(inertials["I_origin"], np.diag([0.0, m * r * r, m * r * r]))


def test_aggregate_with_moi_diagonal():
    """A box-like part: MOI specified as a diagonal triple."""
    c = Craft("box")
    c.add(Mass("body", mass=10.0, moi=(0.5, 1.0, 1.0)))
    inertials = c.aggregate_inertials()
    assert np.allclose(inertials["com"], [0.0, 0.0, 0.0])
    assert np.allclose(inertials["I_com"], np.diag([0.5, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Pure-rotation dynamics
# ---------------------------------------------------------------------------

def test_free_spinning_body_conserves_angular_momentum_z():
    """A spherical-inertia body with no torques should preserve ω_z."""
    c = Craft("spinner")
    # Spherical inertia so ω×(I·ω) = 0 even off-axis; clean check.
    c.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))

    omega_0 = 2.0  # rad/s about +z
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))  # no gravity
    w.add_craft(c, angular_velocity=(0.0, 0.0, omega_0))
    sim = TargetNumpy(Sim(w))

    for _ in range(1000):
        sim.step(0.001)

    # ω should still be ~(0, 0, 2.0) after 1 s.
    assert np.allclose(sim.state["spinner"]["angular_velocity"], [0.0, 0.0, omega_0], atol=1e-6)
    # Body should have rotated by 2 rad. The quaternion at angle θ about +z
    # is (cos(θ/2), 0, 0, sin(θ/2)).
    expected_w = math.cos(1.0)
    expected_z = math.sin(1.0)
    assert np.isclose(sim.state["spinner"]["orientation"][0], expected_w, atol=1e-3)
    assert np.isclose(sim.state["spinner"]["orientation"][3], expected_z, atol=1e-3)
    assert np.allclose(sim.state["spinner"]["orientation"][1:3], [0.0, 0.0], atol=1e-9)


def test_free_spinning_anisotropic_intermediate_axis():
    """Spinning about the principal axis of largest inertia is stable —
    the body should keep ω ≈ ω_0 throughout. Sanity check on Newton-Euler."""
    c = Craft("ellipsoid")
    c.add(Mass("body", mass=1.0, moi=(0.5, 1.0, 2.0)))   # Izz > Iyy > Ixx

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, angular_velocity=(0.0, 0.0, 1.0))
    sim = TargetNumpy(Sim(w))

    for _ in range(500):
        sim.step(0.001)

    # Rotation about the largest-MOI axis is stable: ω stays at (0,0,1).
    assert np.allclose(sim.state["ellipsoid"]["angular_velocity"], [0.0, 0.0, 1.0], atol=1e-6)


def test_no_position_drift_from_pure_rotation():
    """A rotating craft with no linear forces shouldn't translate.
    COM = (r, 0, 0) but body origin should also stay fixed because
    no external force means a_com = 0; the origin orbits the COM but
    in inertial frame is bounded."""
    r = 0.3
    c = Craft("rotating_offset")
    # Two equal masses symmetrically placed → COM at origin → body origin
    # tracks a closed orbit (it's the same as the COM, which is fixed).
    c.add(Mass("a", mass=1.0, transform=(+r, 0.0, 0.0)))
    c.add(Mass("b", mass=1.0, transform=(-r, 0.0, 0.0)))

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, angular_velocity=(0.0, 0.0, 5.0))
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(0.001)

    # No external forces, COM at origin → origin shouldn't move.
    assert np.allclose(sim.state["rotating_offset"]["position"], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(sim.state["rotating_offset"]["velocity"], [0.0, 0.0, 0.0], atol=1e-6)


def test_gravity_creates_no_torque_at_balanced_com():
    """Uniform gravity on a body whose COM coincides with the origin
    produces no torque about the origin. Body in pure free-fall: no spin
    induced, no orientation change."""
    c = Craft("balanced")
    c.add(Mass("a", mass=1.0, transform=(+0.5, 0.0, 0.0)))
    c.add(Mass("b", mass=1.0, transform=(-0.5, 0.0, 0.0)))
    # COM = (0,0,0) by symmetry.
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))

    for _ in range(100):
        sim.step(0.01)

    # Free-fall: vz = -9.81 after 1 s.
    assert np.isclose(sim.state["balanced"]["velocity"][2], -9.81, atol=1e-5)
    # No angular dynamics imparted by symmetric gravity.
    assert np.allclose(sim.state["balanced"]["angular_velocity"], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(sim.state["balanced"]["orientation"], [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_quaternion_normalization_holds_over_long_run():
    """After thousands of integration steps the quaternion stays unit-norm
    thanks to the .normalize() call inside the tick."""
    c = Craft("long_spin")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.2, 0.3)))
    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, angular_velocity=(0.3, 0.5, 0.7))
    sim = TargetNumpy(Sim(w))

    for _ in range(10_000):
        sim.step(0.001)

    q = sim.state["long_spin"]["orientation"]
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_aggregate_composes_nested_offsets():
    """A Mass riding a joint: its craft-frame position is the joint's
    transform composed with its own — not its bare `transform`."""
    from manta.parts import RevoluteJoint

    c = Craft("gimballed")
    c.add(Mass("hub", mass=1.0))
    j = c.add(RevoluteJoint("pan", axis=(0.0, 0.0, 1.0),
                            transform=(1.0, 0.0, 0.0)))
    j.add(Mass("rotor", mass=1.0, transform=(0.5, 0.0, 0.0)))

    inertials = c.aggregate_inertials()
    assert inertials["m_total"] == 2.0
    # rotor sits at (1.5, 0, 0) in craft frame → COM at (0.75, 0, 0).
    np.testing.assert_allclose(inertials["com"], (0.75, 0.0, 0.0))
    np.testing.assert_allclose(
        inertials["I_origin"], np.diag([0.0, 2.25, 2.25]))
    np.testing.assert_allclose(
        inertials["I_com"], np.diag([0.0, 1.125, 1.125]))


def test_aggregate_composes_declared_joint_angle():
    """The numpy snapshot honours the joint's configured rest angle: a
    90° pan about z swings the rotor's offset from +x to +y."""
    from manta.parts import RevoluteJoint

    c = Craft("gimballed")
    c.add(Mass("hub", mass=1.0))
    j = c.add(RevoluteJoint("pan", axis=(0.0, 0.0, 1.0),
                            transform=(1.0, 0.0, 0.0),
                            angle=math.pi / 2))
    j.add(Mass("rotor", mass=1.0, transform=(0.5, 0.0, 0.0)))

    inertials = c.aggregate_inertials()
    # rotor at (1.0, 0.5, 0.0) → COM at the mass-weighted midpoint.
    np.testing.assert_allclose(inertials["com"], (0.5, 0.25, 0.0),
                               atol=1e-12)
