import numpy as np
from manta import Craft, Sim, TargetNumpy, World
from manta.parts import Antenna, Mass


def test_antenna_reports_kinematic_world_position():
    craft = Craft("boat")
    craft.add(Mass("hull", mass=1.0))
    craft.add(Antenna("wifi", mount_offset=(1.0, 0.0, 0.5)))
    world = World()
    world.add_craft(craft, position=(4.0, 2.0, 1.0))
    sim = TargetNumpy(Sim(world))
    sim.step(0.01)
    np.testing.assert_allclose(sim.outputs()["boat"]["wifi.position"],
                               [5.0, 2.0, 1.5])
