"""Analytical geometry, action/reaction and mounted-load contracts."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.contact import ContactMaterial, Kinematics, RevolvedSolid, sphere_contact
from manta.fields import GravityField
from manta.parts import Mass, PrismaticJoint
from manta.parts.disturbance.external_wrench import ExternalWrench


@pytest.mark.parametrize(
    "point,distance,normal",
    [
        ((2.0, 0.0, 0.0), 1.0, (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 2.0), 1.0, (0.0, 0.0, 1.0)),
        ((2.0, 0.0, 2.0), np.sqrt(2), (1 / np.sqrt(2), 0.0, 1 / np.sqrt(2))),
        ((0.5, 0.0, 0.0), -0.5, (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.5), -0.5, (0.0, 0.0, 1.0)),
    ],
)
def test_cylinder_signed_distance(point, distance, normal):
    d, n = RevolvedSolid.cylinder(1.0, 2.0).distance_normal(point)
    assert d == pytest.approx(distance)
    np.testing.assert_allclose(n, normal, atol=1e-12)


def test_contact_has_no_force_outside_and_never_attracts():
    cylinder = RevolvedSolid.cylinder(1.0, 2.0)
    frame = Kinematics((0.0, 0.0, 0.0))
    material = ContactMaterial(stiffness=1000.0, damping=100.0)
    assert (
        sphere_contact(Kinematics((1.2, 0.0, 0.0)), 0.1, cylinder, frame, material)
        is None
    )
    hit = sphere_contact(
        Kinematics((1.05, 0.0, 0.0), velocity=(10.0, 0.0, 0.0)),
        0.1,
        cylinder,
        frame,
        material,
    )
    np.testing.assert_allclose(hit.force_on_sphere_world, (0.0, 0.0, 0.0))


def test_pair_preserves_total_force_and_moment_under_rotation():
    q = (np.cos(0.3), 0.0, np.sin(0.3), 0.0)
    frame = Kinematics((3.0, 4.0, 5.0), q, angular_velocity=(0.2, 0.3, 0.4))
    sphere = Kinematics(tuple(frame.point((1.05, 0.0, 0.3))), velocity=(1.0, 2.0, 3.0))
    hit = sphere_contact(
        sphere, 0.1, RevolvedSolid.cylinder(1.0, 2.0), frame, ContactMaterial()
    )
    a, b = hit.wrenches(np.array(sphere.position), np.array(frame.position))
    np.testing.assert_allclose(a[0] + b[0], 0.0, atol=1e-12)
    total_moment = (
        a[1] + np.cross(sphere.position, a[0]) + b[1] + np.cross(frame.position, b[0])
    )
    np.testing.assert_allclose(total_moment, 0.0, atol=1e-10)


def test_moving_surface_uses_relative_contact_velocity():
    solid = RevolvedSolid.cylinder(1.0, 2.0)
    material = ContactMaterial(stiffness=1000.0, damping=100.0)
    sphere = Kinematics((1.05, 0.0, 0.0), velocity=(0.0, 3.0, 0.0))
    frame = Kinematics((0.0, 0.0, 0.0), velocity=(0.0, 3.0, 0.0))
    hit = sphere_contact(sphere, 0.1, solid, frame, material)
    np.testing.assert_allclose(hit.force_on_sphere_world, (50.0, 0.0, 0.0), atol=1e-10)


def test_open_bore_passes_head_and_closed_bore_retains_it():
    frame = Kinematics((0.0, 0.0, 0.0))
    head = Kinematics((0.0, 0.0, 0.04))
    open_ring = RevolvedSolid(((0.1, -0.01), (0.2, -0.01), (0.2, 0.01), (0.1, 0.01)))
    closed_ring = RevolvedSolid(
        ((0.018, -0.01), (0.2, -0.01), (0.2, 0.01), (0.018, 0.01))
    )
    assert sphere_contact(head, 0.04, open_ring, frame, ContactMaterial()) is None
    assert sphere_contact(head, 0.04, closed_ring, frame, ContactMaterial()) is not None


@pytest.mark.parametrize(
    "profile",
    [
        ((0, 0), (0, 0), (1, 1)),
        ((0, 0), (-1, 0), (1, 1)),
        ((0, 0), (1, 0), (0.2, 0.2), (1, 1), (0, 1)),
    ],
)
def test_invalid_meridian_is_rejected(profile):
    with pytest.raises(ValueError):
        RevolvedSolid(profile)


def test_external_load_reaches_slider_and_conserves_linear_momentum():
    craft = Craft("rig")
    craft.add(Mass("base", mass=10.0, moi=(5.0, 5.0, 5.0)))
    slide = PrismaticJoint("slide", axis=(1.0, 0.0, 0.0))
    slide.add(Mass("slider", mass=2.0, moi=(1.0, 1.0, 1.0)))
    slide.add(ExternalWrench("load"))
    craft.add(slide)
    world = World().add_field(GravityField.none())
    world.add_craft(craft)
    runtime = TargetNumpy(Sim(world))
    runtime.state["rig"]["load.fx"] = 4.0
    runtime.step(0.001)
    state = runtime.state["rig"]
    assert float(state["slide.rate"]) == pytest.approx(0.002, rel=1e-6)
    velocity = np.array(state["velocity"])
    momentum = 10.0 * velocity + 2.0 * (velocity + (state["slide.rate"], 0.0, 0.0))
    np.testing.assert_allclose(momentum, (0.004, 0.0, 0.0), atol=1e-10)


def test_off_center_world_load_applies_lever_arm_torque():
    craft = Craft("body")
    craft.add(Mass("mass", mass=2.0, moi=(1.0, 1.0, 1.0)))
    craft.add(ExternalWrench("load", mount_offset=(0.0, 1.0, 0.0)))
    world = World().add_field(GravityField.none())
    world.add_craft(craft)
    runtime = TargetNumpy(Sim(world))
    runtime.state["body"]["load.fx"] = 2.0
    runtime.step(0.001)
    np.testing.assert_allclose(
        runtime.state["body"]["angular_velocity"], (0.0, 0.0, -0.002), atol=1e-10
    )


def test_centered_head_on_annular_gate_has_no_arbitrary_side_force():
    ring = RevolvedSolid(((0.018, -0.01), (0.2, -0.01), (0.2, 0.01), (0.018, 0.01)))
    hit = sphere_contact(
        Kinematics((0.0, 0.0, 0.035)),
        0.04,
        ring,
        Kinematics((0.0, 0.0, 0.0)),
        ContactMaterial(),
    )
    assert hit is not None
    assert hit.force_on_sphere_world[2] > 0
    np.testing.assert_allclose(hit.force_on_sphere_world[:2], 0.0, atol=1e-12)


def test_cylinder_body_contacts_are_reactive_and_separated_bodies_are_free():
    from manta.contact import cylinder_contacts

    a = Kinematics((0.0, 0.0, 0.0))
    far = Kinematics((0.5, 0.0, 0.0))
    near = Kinematics((0.18, 0.0, 0.0))
    material = ContactMaterial()
    assert cylinder_contacts(a, 0.1, 1.0, far, 0.1, 1.0, material) == ()
    hits = cylinder_contacts(a, 0.1, 1.0, near, 0.1, 1.0, material)
    assert hits
    assert sum(h.force_on_sphere_world[0] for h in hits) < 0
    for hit in hits:
        wa, wb = hit.wrenches(a.position, near.position)
        np.testing.assert_allclose(wa[0] + wb[0], 0.0)
