"""PlanetFrameFluid + Earth — the lambda-based replacement for the
old PlanetCoRotatingFluid.

Validates:
  * Density evaluated from PlanetFrame-coords lambdas; user can use
    `ca.if_else` for hard boundaries (no soft-smoothing artefacts).
  * Earth's combined ocean+atmosphere profile: full water density
    at any depth, exponential ISA density above sea level.
  * Co-rotation contribution still works: a craft at PlanetFrame
    rest gets WorldFrame velocity `ω × r`.
"""

import casadi as ca
import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, FluidState
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Mass
from manta.planets import Earth, SeaWaves
from manta.planets.disturbances import PlanetFrameFluid


def _sample(world: World, point: tuple[float, float, float],
            t: float = 0.0) -> FluidState:
    p = Vec3[WorldFrame].constant(point)
    return world.get_field(FluidField).value_at_sym(p, ca.MX(float(t)))


def _sample_density(world: World, point: tuple[float, float, float],
                    t: float = 0.0) -> float:
    return float(ca.evalf(_sample(world, point, t).density))


# ---------------------------------------------------------------------------

def test_ocean_full_density_at_any_depth():
    """The audit-flagged bug: shallow underwater density was attenuated
    to ~83% at 5 m. With if_else gating, it's exactly full density."""
    earth = Earth(rotation_rate=0.0)
    w = World()
    w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ, 0, 0))
    Sim(w)

    for depth in (0.1, 1.0, 5.0, 50.0, 500.0):
        rho = _sample_density(w, (earth.R_EQ - depth, 0, 0))
        np.testing.assert_allclose(rho, earth.water_density, atol=1e-9,
            err_msg=f"depth={depth}")


def test_atmosphere_exponential_falloff():
    """ISA-style exponential density profile above sea level."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ + 1.0, 0, 0))
    Sim(w)

    H = earth.atmosphere_scale_height
    for alt in (100.0, 1000.0, 10000.0):
        rho = _sample_density(w, (earth.R_EQ + alt, 0, 0))
        expected = earth.air_density * np.exp(-alt / H)
        np.testing.assert_allclose(rho, expected, rtol=1e-9,
            err_msg=f"alt={alt}")


def test_atmosphere_vanishes_at_lunar_distance():
    """At the Moon's distance the atmosphere density is ~10^-19 — the
    old PlanetCoRotatingFluid extended air to infinity."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ + 1.0, 0, 0))
    Sim(w)
    rho = _sample_density(w, (3.84e8, 0, 0))
    assert rho < 1e-15, f"atmosphere should be negligible, got {rho}"


def test_co_rotating_at_rest_in_planet_frame():
    """A craft seeded at PlanetFrame rest should pick up the
    `ω × r_world` WorldFrame velocity. `Planet.at_rest()` returns
    the corresponding PlanetState wrapper."""
    earth = Earth(rotation_rate=Earth.SIDEREAL)
    w = World(); w.add_planet(earth)
    c = Craft("buoy"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=earth.position(earth.R_EQ, 0, 0),
                   velocity=earth.at_rest())
    cw = TargetNumpy(Sim(w))
    state = cw.initial_state()
    expected_v_y = Earth.SIDEREAL * earth.R_EQ
    np.testing.assert_allclose(
        state["buoy"]["velocity"],
        (0.0, expected_v_y, 0.0), atol=1e-9)


def test_custom_planet_frame_fluid_lambda():
    """A user-authored PlanetFrameFluid with an arbitrary lambda
    works — confirms the design generalizes beyond Earth's preset."""
    from manta.planets import Planet
    moon = Planet(name="moon", rotation_axis=(0, 0, 1), omega=2.66e-6)
    w = World()
    w.add_planet(moon)
    ff = FluidField()
    # A donut of "thick fluid" between radii 5 and 10 from moon center.
    def density_fn(p_planet, t):
        r = ca.sqrt(ca.dot(p_planet, p_planet) + 1e-30)
        # CasADi MX uses `ca.logic_and` (no `&` overload).
        return ca.if_else(ca.logic_and(r > 5.0, r < 10.0), 1000.0, 0.0)
    ff.add(PlanetFrameFluid(planet=moon, density_fn=density_fn, name="donut"))
    w.add_field(ff)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c)
    Sim(w)
    # Sample at r=3 (inside hole), r=7 (donut), r=12 (outside).
    np.testing.assert_allclose(_sample_density(w, (3, 0, 0)),    0.0)
    np.testing.assert_allclose(_sample_density(w, (7, 0, 0)), 1000.0)
    np.testing.assert_allclose(_sample_density(w, (12, 0, 0)),   0.0)

# ---------------------------------------------------------------------------
# SeaWaves + surface smoothing
# ---------------------------------------------------------------------------

def _wave_world(**earth_kw):
    """Earth centred at (0,0,-R_EQ): sea level ≈ world z=0 at the origin."""
    earth = Earth(rotation_rate=0.0, position=(0, 0, -Earth.R_EQ), **earth_kw)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 1.0))
    Sim(w)
    return earth, w


def test_waves_move_the_boundary():
    """η = A·cos(kx − ωt): a point at z=+A/2 is in air under a trough and
    in water under a crest — both over position (x) and over time (t)."""
    wv = SeaWaves(amplitude=1.0, wavelength=40.0)
    earth, w = _wave_world(waves=wv)
    k = 2 * np.pi / wv.wavelength
    g0 = earth.gravity_mu / earth.planet_radius**2
    omega = k * np.sqrt(g0 * wv.wavelength / (2 * np.pi))

    # t=0: crest at x=0 (cos), trough half a wavelength away.
    assert _sample_density(w, (0.0, 0.0, 0.5)) == earth.water_density
    assert _sample_density(w, (20.0, 0.0, 0.5)) < 2.0
    # Half a period later the crest and trough have swapped.
    t_half = np.pi / omega
    assert _sample_density(w, (0.0, 0.0, 0.5), t=t_half) < 2.0
    assert _sample_density(w, (20.0, 0.0, 0.5), t=t_half) == \
        earth.water_density


def test_wave_orbital_velocity():
    """Under a crest the water moves with the wave (+x) at A·ω, decaying
    as e^{kz} with depth; in the air above it is still."""
    wv = SeaWaves(amplitude=0.5, wavelength=30.0)
    earth, w = _wave_world(waves=wv)
    k = 2 * np.pi / wv.wavelength
    g0 = earth.gravity_mu / earth.planet_radius**2
    omega = k * np.sqrt(g0 * wv.wavelength / (2 * np.pi))

    def vel(point, t=0.0):
        return np.asarray(ca.evalf(_sample(w, point, t)._mx_of_velocity()
                          if False else _sample(w, point, t).velocity._mx)
                          ).ravel()

    # Just under the crest at x=0: orbital speed ≈ A·ω along +x.
    v = vel((0.0, 0.0, -0.5))
    np.testing.assert_allclose(v[0], wv.amplitude * omega
                               * np.exp(-k * 0.5), rtol=1e-6)
    # Deep below: decayed by e^{k·z}.
    v_deep = vel((0.0, 0.0, -10.0))
    np.testing.assert_allclose(v_deep[0], wv.amplitude * omega
                               * np.exp(-k * 10.0), rtol=1e-6)
    # In the air above the trough: still.
    v_air = vel((15.0, 0.0, 2.0))
    np.testing.assert_allclose(v_air, 0.0, atol=1e-9)


def test_surface_smoothing_blends_density():
    """δ > 0: density ramps smoothly through the boundary — half-way
    between water and air exactly at the surface — instead of stepping."""
    earth, w = _wave_world(surface_smoothing=0.2)
    rho_mid = _sample_density(w, (0.0, 0.0, 0.0))
    mid = 0.5 * (earth.water_density + earth.air_density)
    np.testing.assert_allclose(rho_mid, mid, rtol=1e-3)
    # Compact support: exactly saturated beyond ±δ.
    np.testing.assert_allclose(_sample_density(w, (0, 0, -0.3)),
                               earth.water_density, rtol=1e-9)
    np.testing.assert_allclose(_sample_density(w, (0, 0, 0.3)),
                               earth.air_density, rtol=1e-4)
