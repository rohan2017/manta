"""Independent physical oracles: these do not replay an estimator's equations."""

import numpy as np


def test_truth_orientation_agrees_with_constant_angular_acceleration():
    """Point-sampled gyros and saved attitude must describe the same motion."""
    from manta import Craft, Sim, TargetNumpy, World
    from manta.fields import GravityField
    from manta.parts import ConstantWrench, IMU, Mass

    craft = Craft("craft")
    craft.add(Mass("mass", mass=1.0, moi=(1.0, 1.0, 1.0)))
    craft.add(ConstantWrench("torque", torque=(0.0, 0.0, 0.02)))
    craft.add(IMU("imu"))
    world = World("constant_acceleration").add_field(GravityField.none())
    world.add_craft(craft)
    sim = TargetNumpy(Sim(world))
    h, duration = 0.005, 10.0
    for i in range(round(duration / h)):
        sim.step(h, t=i * h)
    q = np.asarray(sim.state["craft"]["orientation"])
    yaw = 2 * np.arctan2(q[3], q[0])
    np.testing.assert_allclose(yaw, 0.5 * 0.02 * duration**2, atol=1e-11, rtol=0)
    np.testing.assert_allclose(
        sim.state["craft"]["angular_velocity"],
        [0, 0, 0.02 * duration],
        atol=1e-12,
        rtol=0,
    )
