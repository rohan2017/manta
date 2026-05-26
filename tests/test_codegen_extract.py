"""Tests for `manta.codegen.extract` — world-tick CasADi extraction."""

import numpy as np

from manta import Craft, World
from manta.fields import GravityField
from manta.codegen.extract import extract
from manta.parts import IMU, Mass, PositionSensor, Thruster


# ---------------------------------------------------------------------------
# A minimal world for extraction (single drone + gravity)
# ---------------------------------------------------------------------------

def _hover_craft():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    return c


def _hover_world(name: str = "hover_world") -> "CompiledWorld":
    w = World(name=name)
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(_hover_craft())
    return w.compile()


# ---------------------------------------------------------------------------
# Function set is complete and well-sized
# ---------------------------------------------------------------------------

def test_extract_returns_complete_function_set():
    cf = extract(_hover_world())

    assert cf.world_name  == "hover_world"
    assert cf.ambient_dim == 13   # pos(3) + ori(4) + vel(3) + omega(3)
    assert cf.tangent_dim == 12

    # The Thruster's throttle input is the only u.
    assert cf.input_names == ["drone.t.throttle"]
    assert cf.n_inputs == 1

    # predict: in (x, u, dt, t), out x_new
    assert cf.predict_fn.size_in("x")  == (13, 1)
    assert cf.predict_fn.size_in("u")  == (1, 1)
    assert cf.predict_fn.size_in("dt") == (1, 1)
    assert cf.predict_fn.size_in("t")  == (1, 1)
    assert cf.predict_fn.size_out("x_new") == (13, 1)

    # predict_jacobian: F is tangent²
    assert cf.predict_jacobian_fn.size_out("F") == (12, 12)

    # Outputs use full `<craft>.<part>.<output>` names.
    by_name = {o.full_name: o for o in cf.outputs}
    assert set(by_name) == {
        "drone.gps.position", "drone.g.gyro", "drone.g.accel",
    }
    for name in by_name:
        assert by_name[name].out_dim == 3

    # Each Output's H has shape (out_dim, tangent_dim).
    for o in cf.outputs:
        assert o.H_fn.size_out("H") == (o.out_dim, cf.tangent_dim)


# ---------------------------------------------------------------------------
# predict_fn agrees with the world tick
# ---------------------------------------------------------------------------

def test_predict_fn_matches_world_tick():
    from manta import TargetNumpy
    cw  = _hover_world()
    cf  = extract(cw)
    sim = TargetNumpy(cw)

    state = sim.initial_state()
    state["drone"]["position"]   = np.array([0.0, 0.0, 5.0])
    state["drone"]["t.throttle"] = 1.5 * 9.81

    # Pack flat state for the extracted predict.
    flat = {f"drone.{k}": v for k, v in state["drone"].items()
            if f"drone.{k}" in cf.spec}
    x_flat = cf.spec.pack(flat)
    u_flat = np.array([state["drone"]["t.throttle"]])

    x_new_extract = np.asarray(
        cf.predict_fn(x_flat, u_flat, 0.005, 0.0)).ravel()
    # Through the world tick.
    state_new = sim.step(state, dt=0.005)
    flat_new = {f"drone.{k}": v for k, v in state_new["drone"].items()
                if f"drone.{k}" in cf.spec}
    x_new_tick = cf.spec.pack(flat_new)

    np.testing.assert_allclose(x_new_extract, x_new_tick, atol=1e-12)


# ---------------------------------------------------------------------------
# H Jacobians look right at canonical state
# ---------------------------------------------------------------------------

def test_position_sensor_H_is_identity_on_position_block():
    """For a sensor at the craft origin, h(x) = position, so
    H = ∂position/∂δ = [I3 | 0 …] in the tangent layout."""
    cf = extract(_hover_world())

    by_name = {o.full_name: o for o in cf.outputs}
    H_pos = by_name["drone.gps.position"].H_fn

    x = cf.spec.pack({s.name: np.zeros(s.dim) if s.dim > 1 else 0.0
                       for s in cf.spec.slots})
    # Set the rigid-body orientation to identity quaternion.
    x[cf.spec.slot("drone.orientation").offset] = 1.0
    u = np.zeros(cf.n_inputs)
    H = np.asarray(H_pos(x, u, 0.005, 0.0))
    np.testing.assert_allclose(H[:, 0:3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(H[:, 3:],  0.0,       atol=1e-12)


def test_gyro_H_is_identity_on_angular_velocity_block():
    """gyro = ω → H has I3 in the ω columns and zeros elsewhere."""
    cf = extract(_hover_world())

    by_name = {o.full_name: o for o in cf.outputs}
    H_gyro = by_name["drone.g.gyro"].H_fn

    x = cf.spec.pack({s.name: np.zeros(s.dim) if s.dim > 1 else 0.0
                       for s in cf.spec.slots})
    x[cf.spec.slot("drone.orientation").offset] = 1.0
    u = np.zeros(cf.n_inputs)
    H = np.asarray(H_gyro(x, u, 0.005, 0.0))
    np.testing.assert_allclose(H[:, 9:12], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(H[:, 0:9],  0.0,       atol=1e-12)


# ---------------------------------------------------------------------------
# A world without per-craft Inputs still works
# ---------------------------------------------------------------------------

def test_extract_world_with_no_inputs():
    c = Craft("free_fall")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    w = World(name="ff")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    cf = extract(w.compile())
    assert cf.n_inputs == 0
    x = cf.spec.pack({s.name: np.zeros(s.dim) if s.dim > 1 else 0.0
                       for s in cf.spec.slots})
    x[cf.spec.slot("free_fall.orientation").offset] = 1.0
    u = np.zeros(0)
    x_new = np.asarray(cf.predict_fn(x, u, 0.005, 0.0)).ravel()
    # Free-fall under -9.81: v_z = -0.04905, z = -1.226e-4 after one step.
    vel_off = cf.spec.slot("free_fall.velocity").offset
    pos_off = cf.spec.slot("free_fall.position").offset
    assert np.isclose(x_new[vel_off + 2], -0.04905, atol=1e-6)
    assert np.isclose(x_new[pos_off + 2], -1.22625e-4, atol=1e-8)
