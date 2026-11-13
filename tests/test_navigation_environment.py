"""Independent inverse-square and centrifugal oracles for a custom planet."""

import casadi as ca
import numpy as np
import pytest

from manta import Craft, INSStream, NavigationEnvironment, World
from manta.fields import GravityField, PointMassGravity
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import ConstantBiasIMU, Mass
from manta.planets import Planet


class SmallPlanet(Planet):
    def register_disturbances(self, world):
        world.get_or_create_field(GravityField).add(
            PointMassGravity(position=tuple(self.center), GM=4e12, eps=0)
        )


@pytest.mark.parametrize("spin", [0, 2e-4, -1e-4])
def test_custom_planet_gravity_and_spin_share_axes_and_origin(spin):
    planet = SmallPlanet(position=(1e3, 2e3, -3e3), rotation_axis=(1, 2, 3), omega=spin)
    scene = planet.scene_at((2e6, 1e6, 3e6), heading=0.4)
    env = NavigationEnvironment.from_scene(scene, frame_id="site", epoch="0")
    omega = scene.R_world_from_scene(0).T @ planet.omega_vec_world()
    np.testing.assert_allclose(env.frame.angular_velocity, omega)
    for t in (0, 1000):
        for local in (np.zeros(3), np.array([5000, -2000, 1000])):
            R = scene.R_world_from_scene(t)
            radius = scene.origin_world(t) + R @ local - planet.center
            expected = R.T @ (-4e12 * radius / np.linalg.norm(radius) ** 3)
            expected -= np.cross(omega, np.cross(omega, R.T @ radius))
            actual = env.gravity.value_at_sym(
                Vec3[WorldFrame].constant(local), ca.DM(t)
            )
            np.testing.assert_allclose(
                np.asarray(ca.evalf(actual.mx)).ravel(), expected, atol=2e-15
            )

    # Authoring edits cannot silently change an already captured environment.
    old = env.frame.angular_velocity
    planet.omega *= 2
    assert env.frame.angular_velocity == old


def test_stationary_custom_planet_stream_has_no_artificial_motion():
    planet = SmallPlanet(omega=2e-4)
    env = NavigationEnvironment.from_scene(
        planet.scene_at((2e6, 0, 2e6)), frame_id="site", epoch="0"
    )
    c = Craft("robot")
    c.add(Mass("body", mass=1, moi=(1, 1, 1)))
    c.add(ConstantBiasIMU("imu", gyro_noise_sigma=1e-5, accel_noise_sigma=0.001))
    w = World().add_field(env.gravity)
    w.add_craft(c)
    ins = INSStream(
        w,
        imu="imu",
        sample_rate_hz=100,
        predict_every=2,
        navigation_frame=env.frame,
        sensors=[],
    )
    g = env.gravity.value_at_sym(Vec3[WorldFrame].constant((0, 0, 0)), ca.DM(0))
    accel = -np.asarray(ca.evalf(g.mx)).ravel()
    for k in range(101):
        ins.update_imu(accel, env.frame.angular_velocity, t=k * 0.01)
    state = ins.state_dict()["robot"]
    np.testing.assert_allclose(state["position"], 0, atol=1e-12)
    np.testing.assert_allclose(state["velocity"], 0, atol=1e-12)
    np.testing.assert_allclose(state["orientation"], [1, 0, 0, 0], atol=1e-12)


def test_stateful_gravity_is_not_silently_frozen():
    from manta.fields.base import Disturbance
    from manta.parts import State

    class StatefulGravity(Disturbance):
        field_value_shape = GravityField.value_shape
        strength = State(1.0)

        def contribute_at_sym(self, point, t):
            return self.field_value_shape.constant((0, 0, -1)) * self.strength

    class DynamicPlanet(Planet):
        def register_disturbances(self, world):
            world.get_or_create_field(GravityField).add(StatefulGravity())

    with pytest.raises(ValueError, match="stateless"):
        NavigationEnvironment.from_scene(
            DynamicPlanet().scene_at((1, 0, 0)), frame_id="test", epoch="0"
        )
