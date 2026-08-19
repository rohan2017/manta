import math

import numpy as np

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.parts import HeadingSensor, Mass


def _world(heading_rad: float, *, noise_sigma: float = 0.0) -> World:
    world = World()
    craft = Craft("craft")
    craft.add(Mass("body", mass=1.0))
    craft.add(HeadingSensor("phase_heading", heading_vector_noise_sigma=noise_sigma))
    world.add_craft(
        craft,
        orientation=(
            math.cos(heading_rad / 2.0),
            0.0,
            0.0,
            math.sin(heading_rad / 2.0),
        ),
    )
    return world


def test_heading_sensor_emits_continuous_horizontal_unit_vector() -> None:
    for angle in (-math.pi + 1e-6, -0.7, 0.0, 1.2, math.pi - 1e-6):
        sim = TargetNumpy(Sim(_world(angle)))
        sim.step(0.01)
        vector = np.asarray(
            sim.outputs()["craft"]["phase_heading.heading_vector"]
        ).ravel()
        np.testing.assert_allclose(
            vector,
            (math.cos(angle), math.sin(angle), 0.0),
            atol=1e-10,
        )
        np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-10)


def test_heading_sensor_output_is_usable_as_filter_measurement() -> None:
    world = _world(0.4, noise_sigma=0.01)
    ekf = TargetNumpy(EKF(world, sensors=["craft.phase_heading.heading_vector"]))
    assert ekf.module.port("craft.phase_heading.heading_vector").shape == (3,)
