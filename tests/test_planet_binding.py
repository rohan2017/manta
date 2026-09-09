"""Exact craft-to-planet binding and Cartesian planet sensor contracts."""

import numpy as np
import pytest

from manta import Craft, Planet, Sim, TargetNumpy, World
from manta.fields import GravityField, PlanetBindingField
from manta.parts import BottomVelocitySensor, Mass, PlanetPositionSensor


def _craft(name: str, sensor) -> Craft:
    craft = Craft(name)
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    craft.add(sensor)
    return craft


def _zero_g_world() -> World:
    return World().add_field(GravityField.none())


def test_planet_binding_is_craft_scoped_with_two_same_type_planets() -> None:
    a = Planet("a", position=(100.0, 0.0, 0.0), omega=0.1)
    b = Planet("b", position=(-200.0, 30.0, 0.0), omega=-0.2)
    ca = _craft("ca", PlanetPositionSensor("position"))
    cb = _craft("cb", PlanetPositionSensor("position"))
    world = _zero_g_world()
    world.add_planet(a)
    world.add_planet(b)
    world.add_craft(ca, planet=a, position=a.position(3.0, 4.0, 5.0))
    world.add_craft(cb, planet=b, position=b.position(-7.0, 8.0, 9.0))

    sim = TargetNumpy(Sim(world))
    sim.step(0.01, t=0.0)
    np.testing.assert_allclose(
        sim.outputs()["ca"]["position.position"], (3.0, 4.0, 5.0)
    )
    np.testing.assert_allclose(
        sim.outputs()["cb"]["position.position"], (-7.0, 8.0, 9.0)
    )


def test_planet_dependent_part_rejects_unbound_craft() -> None:
    craft = _craft("probe", PlanetPositionSensor("position"))
    world = _zero_g_world()
    world.add_planet(Planet("earth"))
    world.add_craft(craft)
    with pytest.raises(ValueError, match="PlanetBindingField"):
        Sim(world)


def test_planet_binding_cannot_be_registered_as_world_field() -> None:
    world = _zero_g_world()
    with pytest.raises(ValueError, match="craft-scoped"):
        world.add_field(PlanetBindingField(Planet("earth")))


def test_bound_planet_must_be_the_registered_instance() -> None:
    planet = Planet("earth")
    craft = _craft("probe", PlanetPositionSensor("position"))
    world = _zero_g_world()
    world.add_craft(craft, planet=planet)
    with pytest.raises(ValueError, match="bound to unregistered planet 'earth'"):
        Sim(world)


def test_initial_planet_state_must_match_binding() -> None:
    a = Planet("a")
    b = Planet("b")
    craft = _craft("probe", PlanetPositionSensor("position"))
    world = _zero_g_world()
    world.add_planet(a)
    world.add_planet(b)
    world.add_craft(craft, planet=a, position=b.position(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="bound to planet 'a'.*references planet 'b'"):
        Sim(world)


def test_bottom_velocity_removes_bound_planet_corotation() -> None:
    planet = Planet("earth", omega=0.25)
    craft = _craft("probe", BottomVelocitySensor("dvl"))
    p_planet = (10.0, 0.0, 0.0)
    # A craft fixed in the planet frame has nonzero inertial velocity but zero
    # bottom-relative velocity.
    world = _zero_g_world()
    world.add_planet(planet)
    world.add_craft(
        craft,
        planet=planet,
        position=planet.position(*p_planet),
        velocity=planet.velocity(0.0, 0.0, 0.0),
    )
    sim = TargetNumpy(Sim(world))
    sim.step(0.01, t=0.0)
    np.testing.assert_allclose(
        sim.outputs()["probe"]["dvl.velocity"], (0.0, 0.0, 0.0), atol=1e-12
    )


def test_bottom_velocity_reports_planet_relative_motion_in_sensor_frame() -> None:
    planet = Planet("earth", omega=0.25)
    sensor = BottomVelocitySensor(
        "dvl", mount_orientation=(0.0, 1.0, 0.0, 0.0)
    )
    craft = _craft("probe", sensor)
    world = _zero_g_world()
    world.add_planet(planet)
    world.add_craft(
        craft,
        planet=planet,
        position=planet.position(10.0, 0.0, 0.0),
        velocity=planet.velocity(1.0, 2.0, 3.0),
    )
    sim = TargetNumpy(Sim(world))
    sim.step(0.01, t=0.0)
    # The mount is rotated 180 degrees about x, so y/z reverse in its frame.
    np.testing.assert_allclose(
        sim.outputs()["probe"]["dvl.velocity"], (1.0, -2.0, -3.0), atol=1e-12
    )
