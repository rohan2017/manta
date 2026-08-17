import shutil

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.parts import Antenna, Mass, RevoluteJoint


def _quat_z(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def _quat_x(angle):
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def _quat_product(a, b):
    aw, av = a[0], np.asarray(a[1:])
    bw, bv = b[0], np.asarray(b[1:])
    return np.concatenate(
        (
            [aw * bw - np.dot(av, bv)],
            aw * bv + bw * av + np.cross(av, bv),
        )
    )


def _reading(sim, name):
    return np.asarray(sim.outputs()["boat"][name], dtype=float).ravel()


def test_antenna_reports_kinematic_world_position():
    craft = Craft("boat")
    craft.add(Mass("hull", mass=1.0))
    craft.add(Antenna("wifi", mount_offset=(1.0, 0.0, 0.5)))
    world = World()
    world.add_craft(craft, position=(4.0, 2.0, 1.0))
    sim = TargetNumpy(Sim(world))
    sim.step(0.01)
    np.testing.assert_allclose(sim.outputs()["boat"]["wifi.position"], [5.0, 2.0, 1.5])


def test_antenna_reports_static_mount_world_orientation():
    """World attitude composes craft attitude then the antenna mount."""
    craft_q = _quat_z(np.pi / 2)
    mount_q = _quat_x(np.pi / 2)
    craft = Craft("boat")
    craft.add(Mass("hull", mass=1.0))
    craft.add(Antenna("wifi", mount_orientation=mount_q))
    world = World()
    world.add_craft(craft, orientation=craft_q)

    sim = TargetNumpy(Sim(world))
    sim.step(0.01)

    expected = _quat_product(craft_q, mount_q)
    actual = _reading(sim, "wifi.orientation")
    # q and -q encode the same attitude.  Align signs before checking the
    # convention so the assertion remains geometrically meaningful.
    actual *= np.sign(np.dot(actual, expected)) or 1.0
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_antenna_reports_rotating_craft_rate_in_world_frame():
    """Body rates are rotated to world axes; mount rotation changes no rate."""
    craft_q = _quat_z(np.pi / 2)
    craft = Craft("boat")
    craft.add(Mass("hull", mass=1.0, moi=(1.0, 1.0, 1.0)))
    craft.add(Antenna("wifi", mount_orientation=_quat_x(np.pi / 3)))
    world = World()
    # +x body rate points along +y in world after the craft yaw.
    world.add_craft(craft, orientation=craft_q, angular_velocity=(2.0, 0.0, 0.0))

    sim = TargetNumpy(Sim(world))
    sim.step(0.001)

    np.testing.assert_allclose(
        _reading(sim, "wifi.angular_velocity"),
        (0.0, 2.0, 0.0),
        atol=1e-12,
    )


def _articulated_world():
    craft = Craft("boat")
    craft.add(Mass("hull", mass=20.0, moi=(5.0, 5.0, 5.0)))
    mast = RevoluteJoint("mast", axis=(0.0, 0.0, 1.0), mode="passive")
    mast.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.01)))
    mast.add(
        Antenna(
            "wifi", mount_offset=(1.0, 0.0, 0.0), mount_orientation=_quat_x(np.pi / 2)
        )
    )
    craft.add(mast)
    world = World()
    world.add_craft(
        craft,
        **{
            "mast.angle": np.pi / 2,
            "mast.rate": 3.0,
        },
    )
    return world


def test_antenna_reports_articulated_world_pose_and_rate():
    """Antenna outputs include the angle and rate of every joint above it."""
    sim = TargetNumpy(Sim(_articulated_world()))
    sim.step(0.001)

    np.testing.assert_allclose(
        _reading(sim, "wifi.position"), (0.0, 1.0, 0.0), atol=1e-12
    )
    expected_q = _quat_product(_quat_z(np.pi / 2), _quat_x(np.pi / 2))
    actual_q = _reading(sim, "wifi.orientation")
    actual_q *= np.sign(np.dot(actual_q, expected_q)) or 1.0
    np.testing.assert_allclose(actual_q, expected_q, atol=1e-12)
    np.testing.assert_allclose(
        _reading(sim, "wifi.angular_velocity"),
        (0.0, 0.0, 3.0),
        atol=1e-12,
    )


def test_antenna_numpy_generated_kernel_parity():
    """The generated C kernel preserves all antenna output conventions."""
    if shutil.which("cc") is None:
        pytest.skip("no C compiler on PATH")
    interpreted = TargetNumpy(Sim(_articulated_world()))
    generated = TargetNumpy(Sim(_articulated_world()), compile=True)

    interpreted.step(0.001)
    generated.step(0.001)
    for field in ("position", "orientation", "angular_velocity"):
        np.testing.assert_allclose(
            _reading(generated, f"wifi.{field}"),
            _reading(interpreted, f"wifi.{field}"),
            atol=1e-12,
        )
