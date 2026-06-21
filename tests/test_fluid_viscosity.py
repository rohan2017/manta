"""Viscosity in the fluid layer — Sutherland default-fill for gases,
explicit override for liquids, and correct combining across regimes.

Viscosity is an independent FluidState component (like density): a gas
baseline fills it from temperature via Sutherland's law, water sets it
directly. It is what makes the Reynolds number a foil sees vary from air
to water to a thin extraterrestrial atmosphere.
"""

import casadi as ca
import numpy as np

from manta import Craft, Sim, World
from manta.fields import FluidField
from manta.ir.frames import WorldFrame
from manta.ir.types import Vec3
from manta.parts import Mass
from manta.planets import Earth
from manta.planets.atmosphere import (
    MU_REF_AIR, T0_ISA, isa_temperature, sutherland_viscosity,
)


def _sample(world, point, t=0.0):
    p = Vec3[WorldFrame].constant(point)
    return world.get_field(FluidField).value_at_sym(p, ca.MX(float(t)))


def _visc(world, point, t=0.0):
    return float(ca.evalf(_sample(world, point, t).viscosity))


# --- Sutherland helper ------------------------------------------------------

def test_sutherland_air_sea_level():
    """Air viscosity at the ISA sea-level temperature is the textbook
    ~1.79e-5 Pa·s, and equals the reference μ at the reference T."""
    mu = float(ca.evalf(sutherland_viscosity(T0_ISA)))
    np.testing.assert_allclose(mu, 1.79e-5, rtol=0.01)
    mu_ref = float(ca.evalf(sutherland_viscosity(273.15)))
    np.testing.assert_allclose(mu_ref, MU_REF_AIR, rtol=1e-6)


def test_sutherland_monotonic_in_temperature():
    """Gas viscosity rises with temperature (unlike a liquid's)."""
    cold = float(ca.evalf(sutherland_viscosity(220.0)))
    warm = float(ca.evalf(sutherland_viscosity(320.0)))
    assert cold < warm


# --- UniformFluid default-fill ---------------------------------------------

def test_bare_air_fluid_gets_air_viscosity():
    """A bare `add_uniform(density=1.225)` air medium (no temperature)
    fills viscosity from Sutherland at ISA sea level."""
    w = World().add_field(FluidField().add_uniform(density=1.225))
    c = Craft("p"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 0)); Sim(w)
    mu = _visc(w, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(mu, float(ca.evalf(sutherland_viscosity(T0_ISA))),
                               rtol=1e-9)


def test_uniform_fluid_temperature_sets_viscosity():
    """With a temperature given, the air default-fill uses Sutherland at
    that temperature (a colder column is less viscous)."""
    w = World().add_field(FluidField().add_uniform(density=1.0, temperature=230.0))
    c = Craft("p"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 0)); Sim(w)
    mu = _visc(w, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(mu, float(ca.evalf(sutherland_viscosity(230.0))),
                               rtol=1e-9)
    assert mu < float(ca.evalf(sutherland_viscosity(T0_ISA)))


def test_explicit_viscosity_override_for_water():
    """A liquid sets viscosity directly — no Sutherland, ~1e-3 Pa·s."""
    w = World().add_field(
        FluidField().add_uniform(density=1000.0, viscosity=1.0e-3))
    c = Craft("p"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 0)); Sim(w)
    np.testing.assert_allclose(_visc(w, (0, 0, 0)), 1.0e-3, rtol=1e-9)


def test_thin_co2_atmosphere_viscosity_via_species_constants():
    """An exotic gas passes its own Sutherland triple (CO₂-ish here): the
    same model, different constants → a different μ at the same T."""
    # CO₂ Sutherland fit: μ_ref≈1.37e-5 at 273.15 K, S≈222 K.
    mu_co2 = float(ca.evalf(
        sutherland_viscosity(260.0, mu_ref=1.37e-5, T_ref=273.15, S=222.0)))
    w = World().add_field(
        FluidField().add_uniform(density=0.02, temperature=260.0,
                                 viscosity=mu_co2))
    c = Craft("p"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(0, 0, 0)); Sim(w)
    np.testing.assert_allclose(_visc(w, (0, 0, 0)), mu_co2, rtol=1e-9)
    # Different from air at the same temperature.
    assert abs(mu_co2 - float(ca.evalf(sutherland_viscosity(260.0)))) > 1e-7


# --- Earth regimes ----------------------------------------------------------

def test_earth_atmosphere_viscosity_drops_with_altitude():
    """Earth's air gets Sutherland(T); since ISA temperature lapses with
    altitude, viscosity decreases with height (and matches Sutherland of
    the local ISA temperature)."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ, 0, 0)); Sim(w)

    mu_sea = _visc(w, (earth.R_EQ + 0.0, 0, 0))
    mu_high = _visc(w, (earth.R_EQ + 8000.0, 0, 0))
    assert mu_high < mu_sea
    # Matches Sutherland of the local ISA temperature at 8 km.
    T_high = float(ca.evalf(isa_temperature(8000.0,
                                            earth.sea_level_temperature,
                                            earth.lapse_rate)))
    np.testing.assert_allclose(
        mu_high, float(ca.evalf(sutherland_viscosity(T_high))), rtol=1e-6)


def test_earth_ocean_viscosity_is_seawater():
    """Below the surface, viscosity is the seawater constant (~1.35e-3),
    three orders larger than the air above it."""
    earth = Earth(rotation_rate=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe"); c.add(Mass("body", mass=1.0))
    w.add_craft(c, position=(earth.R_EQ, 0, 0)); Sim(w)

    mu_wet = _visc(w, (earth.R_EQ - 50.0, 0, 0))
    mu_air = _visc(w, (earth.R_EQ + 50.0, 0, 0))
    np.testing.assert_allclose(mu_wet, 1.35e-3, rtol=1e-6)
    assert mu_wet > 50.0 * mu_air
