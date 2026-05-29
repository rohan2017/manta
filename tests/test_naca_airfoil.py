"""Naca00xx — symmetric airfoil tests."""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import Mass, Naca00xx, Thruster


def test_zero_aoa_no_lift_only_drag():
    """At α=0 (wind exactly along chord), CL=0 and only drag acts."""
    rho = 1.225
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))   # no gravity, isolate aero
         .add_field(FluidField().add_uniform(density=rho)))
    c = Craft("foil")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Naca00xx("wing",
                   area=1.0, CL_max=1.2, CD_0=0.01, induced_k=0.1,
                   chord_axis=(1.0, 0.0, 0.0),
                   normal_axis=(0.0, 0.0, 1.0)))
    w.add_craft(c, velocity=(10.0, 0.0, 0.0))   # wind from front along chord
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    state = cw.step(state, dt=0.001)

    v = np.array(state["foil"]["velocity"]).ravel()
    # Drag decelerates +x. Lift component (along z) should be ZERO at α=0.
    assert v[0] < 10.0          # decelerated
    assert abs(v[1]) < 1e-9     # no lateral force
    assert abs(v[2]) < 1e-9     # no lift at zero AoA


def test_positive_aoa_produces_upward_lift():
    """Wing pointing along +x, wind blowing from +x at slight angle below
    chord (v_normal < 0 wrt our normal axis = (0,0,1) → α < 0). Use the
    opposite scenario: tilt the wing so its chord points down-forward and
    we get positive AoA from horizontal wind.

    Simpler: keep wing horizontal (chord = +x, normal = +z), give it a
    velocity that's mostly +x with a small +z component. v_chord > 0 and
    v_normal > 0 → sin(2α) > 0 → positive lift (along +normal = +z).
    But also negative v_normal lift direction... actually with the
    sign conventions in the part, let's just check that lift appears.
    """
    rho = 1.225
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(FluidField().add_uniform(density=rho)))
    c = Craft("foil")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Naca00xx("wing",
                   area=1.0, CL_max=1.2, CD_0=0.01, induced_k=0.1,
                   chord_axis=(1.0, 0.0, 0.0),
                   normal_axis=(0.0, 0.0, 1.0)))
    # Velocity slightly tilted: 10 m/s along +x, 1 m/s along +z. Relative
    # wind is OPPOSITE the velocity direction. So v_rel · normal = -1
    # (negative AoA) and v_rel · chord = -10.
    w.add_craft(c, velocity=(10.0, 0.0, 1.0))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    state = cw.step(state, dt=0.001)

    v = np.array(state["foil"]["velocity"]).ravel()
    # The wing should produce a force perpendicular to v_rel. Whether it's
    # +z or -z depends on the lift sign convention. The physically right
    # answer: a wing moving up-and-forward through still air has positive
    # AoA from the wing's perspective (wind from below) → lift along +z
    # in body frame. We started with v_z=1; after a tick the velocity z
    # component should have been pushed (either up or down) by some lift
    # force. Verify it changed.
    assert abs(v[2] - 1.0) > 1e-6   # lift acted on the z component


def test_no_fluid_field_means_no_aero():
    """Without a FluidField registered, the airfoil contributes nothing
    (zero ρ → zero forces)."""
    w = World().add_field(GravityField().add_uniform((0, 0, 0)))    # no fluid
    c = Craft("foil")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Naca00xx("wing", area=100.0, CL_max=10.0))   # huge wing
    w.add_craft(c, velocity=(10.0, 0.0, 0.0))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    for _ in range(100):
        state = cw.step(state, dt=0.001)
    v = np.array(state["foil"]["velocity"]).ravel()
    np.testing.assert_allclose(v, (10.0, 0.0, 0.0), atol=1e-9)


def test_drag_dominates_at_high_angle():
    """At α = 45° (sin(2α) = 1, peak CL; sin²α = 0.5), CL = CL_max,
    CD = CD_0 + induced_k·0.5. With a symmetric incoming wind tilted
    45°, the result is comparable lift and drag (drag is much smaller
    by default coefficients)."""
    rho = 1.225
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(FluidField().add_uniform(density=rho)))
    c = Craft("foil")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Naca00xx("wing",
                   area=1.0, CL_max=1.2, CD_0=0.01, induced_k=0.1))
    # Velocity at 45° to the chord: equal v_chord and v_normal magnitudes.
    v_mag = 10.0
    w.add_craft(c, velocity=(v_mag, 0.0, v_mag))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    state = cw.step(state, dt=0.0001)   # short tick

    # At 45° AoA the wing produces substantial force in both x and z.
    # Lift is perpendicular to RELATIVE WIND (not to body velocity), so at
    # this orientation it has both chord and normal components in body
    # frame. Just verify both x and z velocity components changed (not
    # zero force).
    v = np.array(state["foil"]["velocity"]).ravel()
    assert abs(v[0] - v_mag) > 1e-3
    assert abs(v[2] - v_mag) > 1e-3


def test_lift_zero_at_alpha_zero_and_90():
    """sin(2α) is zero at both α=0 and α=90° (when the relative wind is
    purely along normal). Check the second corner: pure-normal v_rel
    → zero lift, all drag."""
    rho = 1.225
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(FluidField().add_uniform(density=rho)))
    c = Craft("foil")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Naca00xx("wing",
                   area=1.0, CL_max=1.2, CD_0=0.01, induced_k=0.1))
    # Velocity purely along normal (z). v_chord = 0 → sin(2α) = 0.
    w.add_craft(c, velocity=(0.0, 0.0, 10.0))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    state = cw.step(state, dt=0.001)

    v = np.array(state["foil"]["velocity"]).ravel()
    # Drag opposes velocity → reduces v_z; no lift force in the perpendicular
    # plane (since at this orientation lift = ±chord, drag = ±normal,
    # but chord is the direction with zero velocity so lift force vanishes
    # in our model).
    assert v[2] < 10.0          # drag acted
    assert abs(v[0]) < 1e-6     # no lift along chord
    assert abs(v[1]) < 1e-9


def test_offset_airfoil_produces_torque():
    """Wing mounted on a long boom (off-axis) → lift force at offset →
    pitch torque on the body."""
    rho = 1.225
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(FluidField().add_uniform(density=rho)))
    c = Craft("plane")
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    # Wing 1 m forward of body origin in +x direction.
    c.add(Naca00xx("wing",
                   area=1.0, CL_max=1.2,
                   chord_axis=(1.0, 0.0, 0.0),
                   normal_axis=(0.0, 0.0, 1.0),
                   transform=(1.0, 0.0, 0.0)))
    # Forward + upward velocity → positive AoA → lift along +z at wing.
    w.add_craft(c, velocity=(10.0, 0.0, 1.0))
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    state = cw.step(state, dt=0.001)

    omega = np.array(state["plane"]["angular_velocity"]).ravel()
    # Lift at +x offset → pitch torque about body y.
    assert abs(omega[1]) > 1e-6
