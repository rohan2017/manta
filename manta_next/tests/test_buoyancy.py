"""FluidField + PointBuoy — Archimedes' principle in manta_next."""

import casadi as ca
import numpy as np

from manta_next import Craft, World
from manta_next.fields import (
    CurrentFlow, FluidField, FluidState, UniformFluid,
)
from manta_next.ir.frames import WorldFrame
from manta_next.ir.types import Vec3
from manta_next.parts import Mass, PointBuoy


def _eval_fluid_at(field, point_xyz):
    """Evaluate a FluidField at a concrete point, returning numpy
    (density, velocity)."""
    p = Vec3[WorldFrame].constant(point_xyz)
    s = field.state_at_sym(p)
    rho = float(ca.evalf(s.density))
    vel = np.asarray(ca.evalf(s.velocity._mx)).ravel()
    return rho, vel


# ---------------------------------------------------------------------------
# FluidField + FluidState
# ---------------------------------------------------------------------------

def test_empty_fluid_field_is_zero():
    ff = FluidField()
    rho, vel = _eval_fluid_at(ff, (1.0, 2.0, 3.0))
    assert rho == 0.0
    np.testing.assert_allclose(vel, np.zeros(3))


def test_uniform_fluid_density_position_independent():
    ff = FluidField()
    ff.add(UniformFluid(density=1025.0))
    for q in [(0, 0, 0), (1, -2, 3), (1e6, 0, -1e6)]:
        rho, vel = _eval_fluid_at(ff, q)
        assert rho == 1025.0
        np.testing.assert_allclose(vel, np.zeros(3))


def test_uniform_fluid_with_flow():
    ff = FluidField()
    ff.add(UniformFluid(density=1000.0, velocity=(0.0, 1.5, 0.0)))
    rho, vel = _eval_fluid_at(ff, (0, 0, 0))
    assert rho == 1000.0
    np.testing.assert_allclose(vel, (0.0, 1.5, 0.0))


def test_fluid_superposition_density_and_velocity():
    """UniformFluid + CurrentFlow → density from the uniform background,
    velocity from both summed."""
    ff = FluidField()
    ff.add(UniformFluid(density=1000.0, velocity=(0.0, 0.0, -0.2)))
    ff.add(CurrentFlow(velocity=(1.0, 0.0, 0.0)))
    rho, vel = _eval_fluid_at(ff, (5.0, 0.0, 0.0))
    assert rho == 1000.0       # only uniform contributes density
    np.testing.assert_allclose(vel, (1.0, 0.0, -0.2))


# ---------------------------------------------------------------------------
# PointBuoy mechanics
# ---------------------------------------------------------------------------

def test_buoy_neutral_when_displaced_weight_equals_craft_weight():
    """ρ·V·g exactly cancels m·g → craft hovers stationary."""
    # 1 kg craft displacing 1 kg of fluid: V = m/ρ
    rho = 1000.0
    m   = 1.0
    V   = m / rho     # 1e-3 m³
    g   = (0.0, 0.0, -9.81)

    w = (World()
         .add_uniform_gravity(g)
         .add_uniform_fluid(density=rho))
    c = Craft("float")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(PointBuoy("buoy", volume=V))
    w.add_craft(c, position=(0, 0, 10))
    cw = w.compile()

    state = cw.initial_state()
    for _ in range(500):
        state = cw.step(state, dt=0.001)

    np.testing.assert_allclose(state["float"]["position"],
                               np.array([0.0, 0.0, 10.0]), atol=1e-6)
    np.testing.assert_allclose(state["float"]["velocity"],
                               np.zeros(3), atol=1e-6)


def test_buoy_sinks_when_displaced_weight_less_than_craft_weight():
    """Heavier than the displaced fluid → sinks."""
    rho = 1000.0
    m   = 2.0
    V   = 1e-3        # displaces 1 kg of water → net 1 kg sinking
    g   = (0.0, 0.0, -9.81)

    w = (World()
         .add_uniform_gravity(g)
         .add_uniform_fluid(density=rho))
    c = Craft("sinker")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(PointBuoy("buoy", volume=V))
    w.add_craft(c, position=(0, 0, 10))
    cw = w.compile()

    state = cw.initial_state()
    # Net upward force per kg of craft: F = -ρVg + mg → on z, F = ρV·9.81 − m·9.81.
    # F = 1·9.81 − 2·9.81 = -9.81 N. a_z = F/m = -4.905 m/s². After 0.5s:
    # z_drop = 0.5·4.905·0.25 = 0.613 m.
    for _ in range(500):
        state = cw.step(state, dt=0.001)
    expected_a = (rho * V - m) / m * 9.81   # < 0, sinking
    expected_drop = 0.5 * abs(expected_a) * (0.5 ** 2)
    assert np.isclose(state["sinker"]["position"][2], 10.0 - expected_drop, atol=1e-4)


def test_buoy_rises_when_displaced_weight_greater_than_craft_weight():
    """Lighter than displaced fluid → rises."""
    rho = 1000.0
    m   = 0.5
    V   = 1e-3        # displaces 1 kg, but craft only weighs 0.5 → rises
    g   = (0.0, 0.0, -9.81)

    w = (World()
         .add_uniform_gravity(g)
         .add_uniform_fluid(density=rho))
    c = Craft("balloon")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(PointBuoy("buoy", volume=V))
    w.add_craft(c, position=(0, 0, 10))
    cw = w.compile()

    state = cw.initial_state()
    for _ in range(500):
        state = cw.step(state, dt=0.001)
    assert state["balloon"]["position"][2] > 10.0   # rose


def test_no_fluid_field_means_no_buoyancy():
    """Without a FluidField registered the buoy contributes nothing —
    the craft free-falls as if there were no fluid at all."""
    g = (0.0, 0.0, -9.81)
    w = World().add_uniform_gravity(g)    # no fluid added
    c = Craft("freefall")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PointBuoy("buoy", volume=1.0))   # would normally float a lot
    w.add_craft(c, position=(0, 0, 10))
    cw = w.compile()

    state = cw.initial_state()
    for _ in range(500):
        state = cw.step(state, dt=0.001)
    # Free-fall: z = 10 - 0.5·9.81·0.25 = 10 - 1.22625 = 8.77375.
    assert np.isclose(state["freefall"]["position"][2], 8.77375, atol=1e-4)


def test_buoy_offset_produces_righting_torque():
    """An offset buoyancy point on a rigid body creates a torque about the
    body origin — the classic 'righting moment' of a tilted boat. Here
    we verify the torque appears immediately at t=0 by simulating one
    tick of a tilted (45° about y) craft with a high-up buoy and zero net
    vertical force."""
    rho = 1000.0
    m   = 1.0
    V   = m / rho       # neutral buoyancy at the COM
    g   = (0.0, 0.0, -9.81)

    w = (World()
         .add_uniform_gravity(g)
         .add_uniform_fluid(density=rho))
    c = Craft("hull")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    # Buoy mounted 0.5 m above body origin in craft +z.
    c.add(PointBuoy("top", volume=V, transform=(0.0, 0.0, 0.5)))
    # Tilt 45° about body y → craft +z points anchor +x somewhat.
    phi = np.pi / 4
    q = (np.cos(phi/2), 0, np.sin(phi/2), 0)
    w.add_craft(c, position=(0, 0, 0), orientation=q)
    cw = w.compile()

    state = cw.initial_state()
    state_after = cw.step(state, dt=0.001)
    # The buoy applies F = -ρVg = (0, 0, +m·g) at offset (0,0,0.5) in body.
    # In body frame, F appears at (0,0,0.5) after rotating g into body frame.
    # The lever-arm × force produces a body-frame torque about y. The
    # angular velocity after 1ms is small but nonzero in the y component.
    omega = np.array(state_after["hull"]["angular_velocity"]).ravel()
    assert abs(omega[1]) > 1e-6     # torque about y appeared
    # x and z angular velocity stay zero by symmetry.
    assert abs(omega[0]) < 1e-9
    assert abs(omega[2]) < 1e-9
