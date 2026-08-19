"""PlanetFrameFluid + Earth Ocean/Atmosphere regimes.

Validates:
  * A user PlanetFrameFluid evaluated from PlanetFrame-coords lambdas;
    `ca.if_else` for hard boundaries (no soft-smoothing artefacts).
  * Earth's ocean + atmosphere baselines, layered by membership: full
    water density + hydrostatic pressure at any depth; ISA-troposphere
    temperature / pressure / (ideal-gas) density above sea level.
  * Co-rotation contribution still works: a craft at PlanetFrame
    rest gets WorldFrame velocity `ω × r`.
"""

import casadi as ca
import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, FluidState
from manta.fields.fluid_props import (
    R_AIR, ideal_gas_density, isa_pressure, isa_temperature,
)
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


def _scalar(world, point, attr, t: float = 0.0) -> float:
    return float(ca.evalf(getattr(_sample(world, point, t), attr)))


def _earth_isa_constants(earth):
    """The (P0, T0, lapse, g0, R) Earth feeds into the ISA equations."""
    T0 = earth.sea_level_temperature
    g0 = (earth.gravity_mu if earth.gravity_mu > 0.0 else Earth.MU) \
        / earth.planet_radius ** 2
    P0 = earth.air_density * R_AIR * T0
    return P0, T0, earth.lapse_rate, g0, R_AIR


# ---------------------------------------------------------------------------

def test_ocean_full_density_at_any_depth():
    """The audit-flagged bug: shallow underwater density was attenuated
    to ~83% at 5 m. With if_else gating, it's exactly full density."""
    earth = Earth(rotation_rate=0.0)
    w = World()
    w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ, 0, 0))
    w = Sim(w).world

    for depth in (0.1, 1.0, 5.0, 50.0, 500.0):
        rho = _sample_density(w, (earth.R_EQ - depth, 0, 0))
        np.testing.assert_allclose(rho, earth.water_density, atol=1e-9,
            err_msg=f"depth={depth}")


def test_atmosphere_isa_lapse_profile():
    """ISA troposphere above sea level: linear temperature lapse,
    consistent barometric pressure, ideal-gas density."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ + 1.0, 0, 0))
    w = Sim(w).world

    P0, T0, L, g0, R = _earth_isa_constants(earth)
    for alt in (100.0, 1000.0, 10000.0):
        p = (earth.R_EQ + alt, 0, 0)
        T_exp = float(ca.evalf(isa_temperature(alt, T0, L)))
        P_exp = float(ca.evalf(isa_pressure(alt, P0, T0, L, g0, R)))
        rho_exp = float(ca.evalf(ideal_gas_density(P_exp, T_exp, R)))
        np.testing.assert_allclose(_scalar(w, p, "temperature"), T_exp,
                                   rtol=1e-9, err_msg=f"T alt={alt}")
        np.testing.assert_allclose(_scalar(w, p, "pressure"), P_exp,
                                   rtol=1e-9, err_msg=f"P alt={alt}")
        np.testing.assert_allclose(_sample_density(w, p), rho_exp,
                                   rtol=1e-9, err_msg=f"rho alt={alt}")


def test_ocean_hydrostatic_pressure():
    """Underwater: constant density, pressure rising as P0 + ρ·g·depth,
    constant temperature."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ, 0, 0))
    w = Sim(w).world

    P0, T0, _, g0, _ = _earth_isa_constants(earth)
    for depth in (1.0, 10.0, 100.0, 1000.0):
        p = (earth.R_EQ - depth, 0, 0)
        P_exp = P0 + earth.water_density * g0 * depth
        np.testing.assert_allclose(_scalar(w, p, "pressure"), P_exp,
                                   rtol=1e-9, err_msg=f"depth={depth}")
        np.testing.assert_allclose(_scalar(w, p, "temperature"), T0,
                                   atol=1e-9, err_msg=f"depth={depth}")


def test_atmosphere_vanishes_at_lunar_distance():
    """At the Moon's distance the atmosphere density is ~10^-19 — the
    old PlanetCoRotatingFluid extended air to infinity."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ + 1.0, 0, 0))
    w = Sim(w).world
    rho = _sample_density(w, (3.84e8, 0, 0))
    assert rho < 1e-15, f"atmosphere should be negligible, got {rho}"


def test_co_rotating_at_rest_in_planet_frame():
    """A craft seeded at PlanetFrame rest should pick up the
    `ω × r_world` WorldFrame velocity."""
    earth = Earth(rotation_rate=Earth.SIDEREAL)
    w = World(); w.add_planet(earth)
    c = Craft("buoy"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=earth.position(earth.R_EQ, 0, 0),
                   velocity=earth.velocity(0.0, 0.0, 0.0))
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
    w = Sim(w).world
    # Sample at r=3 (inside hole), r=7 (donut), r=12 (outside).
    np.testing.assert_allclose(_sample_density(w, (3, 0, 0)),    0.0)
    np.testing.assert_allclose(_sample_density(w, (7, 0, 0)), 1000.0)
    np.testing.assert_allclose(_sample_density(w, (12, 0, 0)),   0.0)

# ---------------------------------------------------------------------------
# SeaWaves + surface smoothing
# ---------------------------------------------------------------------------

def _wave_world(**earth_kw):
    """Earth centred at (0,0,-polar radius): the north-pole sea level is
    world z=0 at the origin."""
    polar_r = Earth.R_EQ * (1.0 - Earth.FLATTENING)
    earth = Earth(rotation_rate=0.0, position=(0, 0, -polar_r), **earth_kw)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 1.0))
    w = Sim(w).world
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


# ---------------------------------------------------------------------------
# Realistic Earth default + local WeatherPatch overlays
# ---------------------------------------------------------------------------

def test_earth_spins_at_sidereal_by_default():
    """Out of the box Earth turns at its true sidereal rate; passing
    rotation_rate=0.0 explicitly opts into a non-rotating planet."""
    assert Earth().omega == Earth.SIDEREAL
    assert Earth(rotation_rate=0.0).omega == 0.0
    assert Earth(rotation_rate=1.5).omega == 1.5


def _patch_world(*patches):
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ + 100.0, 0, 0))
    ff = w.get_or_create_field(FluidField)
    for p in patches:
        ff.add(p)
    w = Sim(w).world
    return earth, w


def test_weather_patch_adds_local_temperature_and_pressure():
    """A WeatherPatch lays a constant (T, P) delta on top of the ISA
    baseline inside its region, and contributes nothing outside it."""
    from manta.fields import WeatherPatch, within_sphere
    earth = Earth(rotation_rate=0.0)
    ctr = (earth.R_EQ + 200.0, 0.0, 0.0)
    earth, w = _patch_world(
        WeatherPatch(temperature=5.0, pressure=-300.0,
                     membership=within_sphere(ctr, radius=50.0, width=5.0)))

    P0, T0, L, g0, R = _earth_isa_constants(earth)
    T_base = float(ca.evalf(isa_temperature(200.0, T0, L)))
    P_base = float(ca.evalf(isa_pressure(200.0, P0, T0, L, g0, R)))
    # Inside (membership ≈ 1): baseline + delta.
    np.testing.assert_allclose(_scalar(w, ctr, "temperature"),
                               T_base + 5.0, rtol=1e-6)
    np.testing.assert_allclose(_scalar(w, ctr, "pressure"),
                               P_base - 300.0, rtol=1e-6)
    # Far outside the sphere (membership ≈ 0): pure baseline, no delta.
    far = (earth.R_EQ + 1000.0, 0.0, 0.0)
    T_far = float(ca.evalf(isa_temperature(1000.0, T0, L)))
    np.testing.assert_allclose(_scalar(w, far, "temperature"),
                               T_far, rtol=1e-6)


def test_weather_patch_callable_curve():
    """The delta can be a callable `(point, t) -> MX` — a custom curve over
    the baseline, here a temperature that grows with altitude."""
    from manta.fields import WeatherPatch

    R_EQ = Earth.R_EQ

    def warm(point, t):
        return 0.01 * (point._mx[0] - R_EQ)      # +0.01 K per metre of altitude

    earth, w = _patch_world(WeatherPatch(temperature=warm))
    P0, T0, L, g0, R = _earth_isa_constants(earth)
    for alt in (100.0, 500.0):
        pt = (earth.R_EQ + alt, 0.0, 0.0)
        T_base = float(ca.evalf(isa_temperature(alt, T0, L)))
        np.testing.assert_allclose(_scalar(w, pt, "temperature"),
                                   T_base + 0.01 * alt, rtol=1e-6)


def test_wave_pressure_attenuates_with_depth():
    """Linear-wave dynamic pressure: P = P_surf + ρ·g·d_mean +
    ρ·g·A·e^{−k·d}·cos(kξ − ωt) — the oscillation a submerged pressure
    sensor feels decays with depth; the hydrostatic mean does not care
    about the instantaneous surface.

    Hand-derived (A=1 m, λ=100 m ⇒ k=0.062832; g0 = μ/R² = 9.79828;
    ρg = 1025·9.79828 = 10043.2 Pa/m):
      d= 2 m: e^{−0.12566} = 0.88192 → amplitude 8857 Pa
      d=30 m: e^{−1.88496} = 0.15195 → amplitude 1526 Pa   (detectable
              by a good depth sensor — and 6.6× SMALLER than the naive
              column to the instantaneous surface would claim)
      ratio d30/d2 = e^{−k·28} = 0.17230
    """
    wv = SeaWaves(amplitude=1.0, wavelength=100.0)
    earth, w = _wave_world(waves=wv)
    k = 2 * np.pi / wv.wavelength
    g0 = earth.gravity_mu / earth.planet_radius**2
    omega = k * np.sqrt(g0 * wv.wavelength / (2 * np.pi))
    rho = earth.water_density
    period = 2 * np.pi / omega

    def sweep(depth):
        ps = [_scalar(w, (0.0, 0.0, -depth), "pressure", t)
              for t in np.linspace(0.0, period, 128, endpoint=False)]
        return (max(ps) - min(ps)) / 2.0, float(np.mean(ps))

    amp2, _ = sweep(2.0)
    amp30, mean30 = sweep(30.0)

    np.testing.assert_allclose(amp2, rho * g0 * np.exp(-k * 2.0),
                               rtol=2e-3)
    np.testing.assert_allclose(amp30, rho * g0 * np.exp(-k * 30.0),
                               rtol=2e-3)
    # The attenuation LAW, independent of the ρg prefactor.
    np.testing.assert_allclose(amp30 / amp2, np.exp(-k * 28.0), rtol=2e-3)
    # Mean pressure is the column to the MEAN surface (dynamic term
    # averages out over a period).
    P0 = earth.air_density * R_AIR * earth.sea_level_temperature
    np.testing.assert_allclose(mean30, P0 + rho * g0 * 30.0, rtol=1e-4)


def test_flat_sea_pressure_unchanged():
    """No waves ⇒ the equivalent pressure depth is the plain hydrostatic
    column — the correction is invisible for a flat sea."""
    earth, w = _wave_world(waves=None)
    g0 = earth.gravity_mu / earth.planet_radius**2
    P0 = earth.air_density * R_AIR * earth.sea_level_temperature
    for d in (5.0, 50.0):
        np.testing.assert_allclose(
            _scalar(w, (0.0, 0.0, -d), "pressure"),
            P0 + earth.water_density * g0 * d, rtol=1e-6)
