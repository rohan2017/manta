"""PositionSensor — emits world-frame craft position as an Output."""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor


def test_position_sensor_at_origin_reads_craft_position():
    c = Craft("p")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps"))

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c, position=(3.0, -1.0, 2.5))
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()

    state = sim.step(state, dt=0.001)
    np.testing.assert_allclose(np.array(state["p"]["gps.position"]).ravel(),
                               np.array([3.0, -1.0, 2.5]), atol=1e-9)


def test_position_sensor_tracks_position_under_freefall():
    """gps.position is the START-of-tick position (it's an observation
    of the input state). It must equal the position fed in *that* tick."""
    g = (0.0, 0.0, -9.81)
    c = Craft("p")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps"))

    w = World().add_field(GravityField(g=g))
    w.add_craft(c, position=(0.0, 0.0, 10.0))
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()

    for _ in range(500):
        pos_in = np.array(state["p"]["position"]).ravel().copy()
        state  = sim.step(state, dt=0.001)
        np.testing.assert_allclose(np.array(state["p"]["gps.position"]).ravel(),
                                   pos_in, atol=1e-12)


def test_position_sensor_with_offset_adds_R_offset():
    """Sensor mounted at (1, 0, 0) on a craft rotated 90° about z reads
    craft.position + (0, 1, 0) in world frame."""
    c = Craft("p")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", transform=(1.0, 0.0, 0.0)))

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    # Craft at origin, rotated 90° about world-frame z (so body +x → anchor +y).
    # Quaternion (w, x, y, z) for 90° about z: (cos(π/4), 0, 0, sin(π/4)).
    w.add_craft(c, position=(0.0, 0.0, 0.0),
                orientation=(np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)))
    sim = TargetNumpy(Sim(w))
    state = sim.initial_state()

    state = sim.step(state, dt=0.001)
    np.testing.assert_allclose(np.array(state["p"]["gps.position"]).ravel(),
                               np.array([0.0, 1.0, 0.0]), atol=1e-9)
