"""DragSurface — quadratic drag tests."""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import DragSurface, Mass, Thruster


def test_offset_drag_uses_mount_velocity_not_double_lever_arm():
    """Regression: ctx.velocity is already the surface point's velocity
    (kinematic pass includes the rotational lever arm), so DragSurface must
    NOT re-add ω×offset. A linear-drag fin offset d along +x on a craft
    spinning at ω about +z sees relative wind ω·d along +y, giving drag
    force −ρ·c·ω·d along y → a_y = −ρ·c·ω·d / M. The old double-lever bug
    doubled the wind (and the force)."""
    rho, c, d, omega, M = 1.0, 0.5, 1.0, 2.0, 100.0
    dt = 1e-3

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))   # free
         .add_field(FluidField().add_uniform(density=rho)))        # still
    craft = Craft("spinner")
    craft.add(Mass("core", mass=M, moi=(10.0, 10.0, 10.0)))        # COM at origin
    craft.add(DragSurface("fin", force=(-c, -c, -c), transform=(d, 0.0, 0.0)))
    w.add_craft(craft, angular_velocity=(0.0, 0.0, omega))
    sim = TargetNumpy(Sim(w))
    state = sim.step(sim.initial_state(), dt=dt)

    vy = float(np.asarray(state["spinner"]["velocity"]).ravel()[1])
    expected = -rho * c * omega * d / M * dt          # single lever arm
    np.testing.assert_allclose(vy, expected, rtol=1e-3)
    assert abs(vy - 2.0 * expected) > 0.4 * abs(expected)   # not the 2× bug


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
         .add_field(GravityField().add_uniform((0.0, 0.0, -g)))
         .add_field(FluidField().add_uniform(density=rho)))
    c = Craft("sphere")
    c.add(Mass("body", mass=m, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface.isotropic_quadratic("hull", area=A, drag_coefficient=Cd))
    w.add_craft(c, position=(0, 0, 0))
    cw = TargetNumpy(Sim(w))

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
         .add_field(GravityField().add_uniform((0, 0, 0)))     # no gravity, isolate drag
         .add_field(FluidField().add_uniform(density=1000.0)))
    c = Craft("slug")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface.isotropic_quadratic("hull", area=0.01, drag_coefficient=1.0))
    w.add_craft(c, velocity=(2.0, 0.0, 0.0))
    cw = TargetNumpy(Sim(w))
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
               .add_field(GravityField().add_uniform((0, 0, 0)))
               .add_field(FluidField().add_uniform(density=1000.0)))
    c_still = Craft("still")
    c_still.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c_still.add(DragSurface.isotropic_quadratic("hull", area=0.01, drag_coefficient=1.0))
    w_still.add_craft(c_still, velocity=(2.0, 0.0, 0.0))

    w_curr = (World()
              .add_field(GravityField().add_uniform((0, 0, 0)))
              .add_field(FluidField().add_uniform(density=1000.0, velocity=(1.0, 0.0, 0.0))))
    c_curr = Craft("curr")
    c_curr.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c_curr.add(DragSurface.isotropic_quadratic("hull", area=0.01, drag_coefficient=1.0))
    w_curr.add_craft(c_curr, velocity=(2.0, 0.0, 0.0))

    cw_s = TargetNumpy(Sim(w_still)); s = cw_s.initial_state()
    cw_c = TargetNumpy(Sim(w_curr));  c = cw_c.initial_state()
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
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(FluidField().add_uniform(density=1000.0)))
    c = Craft("offset")
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    c.add(DragSurface.isotropic_quadratic("drogue",
                                           area=0.1, drag_coefficient=1.0,
                                           transform=(0.5, 0.0, 0.0)))   # +x offset
    # Craft moving in +y → drag at +x offset → moment about +z (yaw).
    w.add_craft(c, velocity=(0.0, 1.0, 0.0))
    cw = TargetNumpy(Sim(w))
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
    w = World().add_field(GravityField().add_uniform((0, 0, 0)))    # no fluid
    c = Craft("cruise")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface.isotropic_quadratic("hull", area=1.0, drag_coefficient=10.0))   # massive A·Cd
    w.add_craft(c, velocity=(1.0, 0.0, 0.0))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    for _ in range(1000):
        state = cw.step(state, dt=0.001)
    vx = float(np.array(state["cruise"]["velocity"]).ravel()[0])
    assert np.isclose(vx, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Polynomial drag (tensor form)
# ---------------------------------------------------------------------------

def test_polynomial_drag_quadratic_per_axis_matches_isotropic_helper():
    """A diagonal A_2 = -½·A·Cd·I3 reproduces the isotropic-quadratic
    behavior — both expressions of the same physics, just different
    constructor surfaces."""
    A, Cd, rho = 0.05, 0.7, 1000.0
    g = (0.0, 0.0, 0.0)

    # Setup A: helper-built craft.
    w1 = World().add_field(GravityField().add_uniform(g)).add_field(FluidField().add_uniform(density=rho))
    c1 = Craft("helper")
    c1.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c1.add(DragSurface.isotropic_quadratic("d", area=A, drag_coefficient=Cd))
    w1.add_craft(c1, velocity=(3.0, 0.0, 0.0))
    cw1 = TargetNumpy(Sim(w1))
    s1 = cw1.initial_state()

    # Setup B: hand-built diagonal A_2.
    w2 = World().add_field(GravityField().add_uniform(g)).add_field(FluidField().add_uniform(density=rho))
    c2 = Craft("hand")
    c2.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    A_2 = -0.5 * A * Cd * np.eye(3)
    c2.add(DragSurface("d", force_tensors=[np.zeros((3, 3)), A_2]))
    w2.add_craft(c2, velocity=(3.0, 0.0, 0.0))
    cw2 = TargetNumpy(Sim(w2))
    s2 = cw2.initial_state()

    for _ in range(200):
        s1 = cw1.step(s1, dt=0.001)
        s2 = cw2.step(s2, dt=0.001)

    np.testing.assert_allclose(s1["helper"]["velocity"],
                               s2["hand"]["velocity"], atol=1e-9)


def test_anisotropic_drag_via_diagonal_A1_decelerates_only_x():
    """A purely-x linear-drag tensor decelerates the x velocity component
    but leaves y and z untouched."""
    rho = 1000.0
    w = World().add_field(GravityField().add_uniform((0, 0, 0))).add_field(FluidField().add_uniform(density=rho))
    c = Craft("strip")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(DragSurface("d", force=(-0.5, 0.0, 0.0)))   # A_1 = diag(-0.5, 0, 0)
    w.add_craft(c, velocity=(2.0, 1.0, 0.5))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    for _ in range(200):
        state = cw.step(state, dt=0.001)
    v = np.array(state["strip"]["velocity"]).ravel()
    # x has decayed, y and z untouched.
    assert v[0] < 2.0
    assert np.isclose(v[1], 1.0, atol=1e-9)
    assert np.isclose(v[2], 0.5, atol=1e-9)


def test_drag_force_vs_force_tensors_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        DragSurface("d", force=(0.1, 0, 0),
                    force_tensors=[np.eye(3)])


def test_drag_polynomial_order_cap():
    with pytest.raises(ValueError, match="exceeds the cap"):
        DragSurface("d", force_tensors=[np.zeros((3, 3))] * 5)
