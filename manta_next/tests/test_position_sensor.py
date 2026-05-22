"""PositionSensor — emits world-frame craft position as an Output."""

import numpy as np

from manta_next import Craft, World
from manta_next.parts import Mass, PositionSensor


def test_position_sensor_at_origin_reads_craft_position():
    c = Craft("p")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps"))

    tick = c.compile_tick(gravity_world=(0.0, 0.0, 0.0))
    state = c.initial_state()
    state["position"] = np.array([3.0, -1.0, 2.5])

    out = tick(dt=0.001, **state)
    np.testing.assert_allclose(np.array(out["gps.position"]).ravel(),
                               np.array([3.0, -1.0, 2.5]), atol=1e-9)


def test_position_sensor_tracks_position_under_freefall():
    """gps.position is the START-of-tick position (it's an observation
    of the input state). It must equal the position fed in *that* tick."""
    g = (0.0, 0.0, -9.81)
    c = Craft("p")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps"))

    tick = c.compile_tick(gravity_world=g)
    state = c.initial_state()
    state["position"] = np.array([0.0, 0.0, 10.0])

    for _ in range(500):
        pos_in = state["position"].copy()
        out    = tick(dt=0.001, **state)
        np.testing.assert_allclose(np.array(out["gps.position"]).ravel(),
                                   pos_in, atol=1e-12)
        state = {**state, **out}


def test_position_sensor_with_offset_adds_R_offset():
    """Sensor mounted at (1, 0, 0) on a craft rotated 90° about z reads
    craft.position + (0, 1, 0) in world frame."""
    c = Craft("p")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", transform=(1.0, 0.0, 0.0)))

    tick = c.compile_tick(gravity_world=(0.0, 0.0, 0.0))
    state = c.initial_state()
    # Craft at origin, rotated 90° about world-frame z (so body +x → anchor +y).
    state["position"] = np.array([0.0, 0.0, 0.0])
    # Quaternion (w, x, y, z) for 90° about z: (cos(π/4), 0, 0, sin(π/4)).
    state["orientation"] = np.array([np.cos(np.pi / 4), 0.0, 0.0,
                                      np.sin(np.pi / 4)])

    out = tick(dt=0.001, **state)
    np.testing.assert_allclose(np.array(out["gps.position"]).ravel(),
                               np.array([0.0, 1.0, 0.0]), atol=1e-9)
