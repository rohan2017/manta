"""Tests for manta_next.codegen.extract — per-function CasADi extraction."""

import numpy as np
import pytest

from manta_next import Craft, World
from manta_next.codegen.extract import extract
from manta_next.parts import IMU, Mass, PositionSensor, Thruster


# ---------------------------------------------------------------------------
# A minimal craft for extraction
# ---------------------------------------------------------------------------

def _hover_craft():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t"))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    return c


# ---------------------------------------------------------------------------
# Function set is complete and well-sized
# ---------------------------------------------------------------------------

def test_extract_returns_complete_function_set():
    cf = extract(_hover_craft())

    assert cf.craft_name == "drone"
    assert cf.ambient_dim == 13   # pos(3) + ori(4) + vel(3) + omega(3)
    assert cf.tangent_dim == 12

    assert cf.input_names == ["t.thrust_cmd"]
    assert cf.n_inputs == 1

    # predict: in (x, u, dt), out x_new
    assert cf.predict_fn.size_in("x")  == (13, 1)
    assert cf.predict_fn.size_in("u")  == (1, 1)
    assert cf.predict_fn.size_in("dt") == (1, 1)
    assert cf.predict_fn.size_out("x_new") == (13, 1)

    # predict_jacobian: F is tangent×tangent
    assert cf.predict_jacobian_fn.size_out("F") == (12, 12)

    # Outputs: gps.position (3) and g.gyro (3)
    by_name = {o.full_name: o for o in cf.outputs}
    assert set(by_name) == {"gps.position", "g.gyro"}
    assert by_name["gps.position"].out_dim == 3
    assert by_name["g.gyro"].out_dim       == 3

    # Each Output's H has shape (out_dim, tangent_dim).
    for o in cf.outputs:
        assert o.H_fn.size_out("H") == (o.out_dim, cf.tangent_dim)


# ---------------------------------------------------------------------------
# predict_fn agrees with the existing compile_tick output
# ---------------------------------------------------------------------------

def test_predict_fn_matches_compile_tick():
    c   = _hover_craft()
    cf  = extract(c)

    tick = c.compile_tick(gravity_anchor=(0.0, 0.0, -9.81))
    state = c.initial_state()
    state["position"] = np.array([0.0, 0.0, 5.0])
    state["t.thrust_cmd"] = 1.5 * 9.81

    # Pack into flat ambient + flat u.
    x_flat = cf.spec.pack(state)
    u_flat = np.array([state["t.thrust_cmd"]])

    # Through extracted predict.
    x_new_extract = np.asarray(cf.predict_fn(x_flat, u_flat, 0.005)).ravel()
    # Through native compile_tick.
    out = tick(dt=0.005, **state)
    state_new = {**state, **out}
    x_new_tick = cf.spec.pack(state_new)

    np.testing.assert_allclose(x_new_extract, x_new_tick, atol=1e-12)


# ---------------------------------------------------------------------------
# H Jacobians look right at canonical state
# ---------------------------------------------------------------------------

def test_position_sensor_H_is_identity_on_position_block():
    """For a sensor at the craft origin, h(x) = position, so
    H = ∂position/∂δ = [I3 | 0 …] in the tangent layout."""
    c  = _hover_craft()
    cf = extract(c)

    by_name = {o.full_name: o for o in cf.outputs}
    H_pos = by_name["gps.position"].H_fn

    x = cf.spec.pack(c.initial_state())
    u = np.zeros(cf.n_inputs)
    H = np.asarray(H_pos(x, u, 0.005))
    # Tangent layout: position[0:3], orientation[3:6], velocity[6:9],
    # angular_velocity[9:12].
    np.testing.assert_allclose(H[:, 0:3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(H[:, 3:], 0.0,       atol=1e-12)


def test_gyro_H_is_identity_on_angular_velocity_block():
    """gyro = ω → H has I3 in the ω columns and zeros elsewhere."""
    c  = _hover_craft()
    cf = extract(c)

    by_name = {o.full_name: o for o in cf.outputs}
    H_gyro = by_name["g.gyro"].H_fn

    x = cf.spec.pack(c.initial_state())
    u = np.zeros(cf.n_inputs)
    H = np.asarray(H_gyro(x, u, 0.005))
    np.testing.assert_allclose(H[:, 9:12], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(H[:, 0:9],  0.0,       atol=1e-12)


# ---------------------------------------------------------------------------
# Craft without Inputs still works
# ---------------------------------------------------------------------------

def test_extract_craft_with_no_inputs():
    c = Craft("free_fall")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    cf = extract(c)
    assert cf.n_inputs == 0
    # u vector is empty; predict still callable.
    x = cf.spec.pack(c.initial_state())
    u = np.zeros(0)
    x_new = np.asarray(cf.predict_fn(x, u, 0.005)).ravel()
    # Under (0,0,-9.81) gravity, after dt=0.005: v_z ≈ -0.04905, z ≈ -1.226e-4.
    assert np.isclose(x_new[9], -0.04905, atol=1e-6)   # velocity[2]
    assert np.isclose(x_new[2], -1.22625e-4, atol=1e-8)  # position[2]
