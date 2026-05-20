"""DragSurface — quadratic drag tests."""

import numpy as np

from manta_next import Craft, World
from manta_next.parts import DragSurface, Mass, Thruster


def test_terminal_velocity_from_drag_balances_gravity():
    """Sphere of mass m falling in a fluid of density ρ with drag area A
    and Cd reaches terminal velocity where ½ρACd·v² = m·|g|, so
    v_terminal = sqrt(2·m·g / (ρ·A·Cd))."""
    m   = 1.0
    rho = 1.225        # air
    A   = 0.01
    Cd  = 0.5          # smooth sphere
    g   = 9.81
    v_terminal = np.sqrt(2 * m * g / (rho * A * Cd))

    w = (World()
         .add_uniform_gravity((0.0, 0.0, -g))
         .add_uniform_fluid(density=rho))
    c = Craft("sphere")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface("hull", area=A, drag_coefficient=Cd))
    w.add_craft(c, position=(0, 0, 0))
    cw = w.compile()

    state = cw.initial_state()
    # Run long enough to settle (well past terminal).
    for _ in range(20000):
        state = cw.step(state, dt=0.005)

    vz = float(np.array(state["sphere"]["velocity"]).ravel()[2])
    # vz approaches -v_terminal from above. Compare magnitude.
    assert np.isclose(abs(vz), v_terminal, rtol=1e-3), (
        f"vz={vz:.4f}, expected ≈ {-v_terminal:.4f}")


def test_drag_opposes_motion_through_still_fluid():
    """Push a craft along +x through still fluid; drag decelerates it."""
    w = (World()
         .add_uniform_gravity((0, 0, 0))     # no gravity, isolate drag
         .add_uniform_fluid(density=1000.0))
    c = Craft("slug")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface("hull", area=0.01, drag_coefficient=1.0))
    w.add_craft(c, velocity=(2.0, 0.0, 0.0))
    cw = w.compile()
    state = cw.initial_state()

    for _ in range(1000):
        state = cw.step(state, dt=0.001)

    vx = float(np.array(state["slug"]["velocity"]).ravel()[0])
    # vx should have decreased monotonically from 2.0 and stayed positive.
    assert 0.0 < vx < 2.0


def test_drag_with_following_current_reduces_force():
    """A current moving the same direction as the craft reduces v_rel,
    therefore reduces drag. Two identical setups, one with following
    current — the one with the current decelerates LESS."""
    w_still = (World()
               .add_uniform_gravity((0, 0, 0))
               .add_uniform_fluid(density=1000.0))
    c_still = Craft("still")
    c_still.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c_still.add(DragSurface("hull", area=0.01, drag_coefficient=1.0))
    w_still.add_craft(c_still, velocity=(2.0, 0.0, 0.0))

    w_curr = (World()
              .add_uniform_gravity((0, 0, 0))
              .add_uniform_fluid(density=1000.0, velocity=(1.0, 0.0, 0.0)))
    c_curr = Craft("curr")
    c_curr.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c_curr.add(DragSurface("hull", area=0.01, drag_coefficient=1.0))
    w_curr.add_craft(c_curr, velocity=(2.0, 0.0, 0.0))

    cw_s = w_still.compile(); s = cw_s.initial_state()
    cw_c = w_curr.compile();  c = cw_c.initial_state()
    for _ in range(500):
        s = cw_s.step(s, dt=0.001)
        c = cw_c.step(c, dt=0.001)

    vx_still = float(np.array(s["still"]["velocity"]).ravel()[0])
    vx_curr  = float(np.array(c["curr"]["velocity"]).ravel()[0])
    # Same starting velocity, opposing drag, but the still-fluid case sees
    # full v_rel = 2 and the current case sees v_rel = 1. The current
    # case decelerates LESS, so vx_curr > vx_still.
    assert vx_curr > vx_still


def test_offset_drag_surface_produces_torque():
    """A drag surface mounted off-axis on a translating craft induces a
    yaw torque (the streamer-banner effect)."""
    w = (World()
         .add_uniform_gravity((0, 0, 0))
         .add_uniform_fluid(density=1000.0))
    c = Craft("offset")
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    c.add(DragSurface("drogue",
                      area=0.1, drag_coefficient=1.0,
                      transform=(0.5, 0.0, 0.0)))   # +x offset
    # Craft moving in +y → drag at +x offset → moment about +z (yaw).
    w.add_craft(c, velocity=(0.0, 1.0, 0.0))
    cw = w.compile()
    state = cw.initial_state()
    state = cw.step(state, dt=0.001)

    omega = np.array(state["offset"]["angular_velocity"]).ravel()
    # x and y angular velocity should stay zero; z develops.
    assert abs(omega[2]) > 1e-6
    assert abs(omega[0]) < 1e-9
    assert abs(omega[1]) < 1e-9


def test_no_fluid_field_means_no_drag():
    """Without a FluidField registered the drag surface contributes
    nothing — craft cruises forever."""
    w = World().add_uniform_gravity((0, 0, 0))    # no fluid
    c = Craft("cruise")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface("hull", area=1.0, drag_coefficient=10.0))   # massive A·Cd
    w.add_craft(c, velocity=(1.0, 0.0, 0.0))
    cw = w.compile()
    state = cw.initial_state()
    for _ in range(1000):
        state = cw.step(state, dt=0.001)
    vx = float(np.array(state["cruise"]["velocity"]).ravel()[0])
    assert np.isclose(vx, 1.0, atol=1e-9)
