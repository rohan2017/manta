"""Barometer — reads FluidField pressure; feeds the EKF as a sensor."""

import casadi as ca
import numpy as np
import pytest

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields.fluid_props import R_AIR, isa_pressure
from manta.parts import Barometer, Mass
from manta.planets import Earth


def _earth_isa(earth):
    T0 = earth.sea_level_temperature
    g0 = (earth.gravity_mu if earth.gravity_mu > 0.0 else Earth.MU) \
        / earth.planet_radius ** 2
    P0 = earth.air_density * R_AIR * T0
    return P0, T0, earth.lapse_rate, g0


def _baro_world(alt, *, pressure_noise_sigma=0.0):
    """A probe carrying a barometer at altitude `alt` over a non-rotating,
    gravity-force-free Earth (so it stays put for a single tick; the
    fluid regimes are still registered)."""
    earth = Earth(rotation_rate=0.0, gravity_mu=0.0)
    w = World(); w.add_planet(earth)
    c = Craft("probe")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Barometer("baro", pressure_noise_sigma=pressure_noise_sigma))
    w.add_craft(c, position=(earth.R_EQ + alt, 0.0, 0.0))
    return earth, w, c


def _read_pressure(w):
    sim = TargetNumpy(Sim(w))
    sim.step(1e-3)
    return float(np.asarray(sim.outputs()["probe"]["baro.pressure"]).ravel()[0])


# ---------------------------------------------------------------------------

def test_barometer_reads_isa_pressure_in_air():
    """Above sea level the reading is the ISA-troposphere pressure at the
    sensor's altitude (using Earth's own surface gravity)."""
    for alt in (0.0, 1000.0, 8000.0):
        earth, w, _ = _baro_world(alt)
        P0, T0, L, g0 = _earth_isa(earth)
        expected = float(ca.evalf(isa_pressure(alt, P0, T0, L, g0, R_AIR)))
        assert abs(_read_pressure(w) - expected) < 1e-3, f"alt={alt}"


def test_barometer_reads_hydrostatic_pressure_underwater():
    """Below the surface the reading is P0 + ρ_w·g·depth."""
    earth, w, _ = _baro_world(-50.0)
    P0, _, _, g0 = _earth_isa(earth)
    expected = P0 + earth.water_density * g0 * 50.0
    assert abs(_read_pressure(w) - expected) < 1e-3


def test_barometer_requires_fluid_field():
    """A Barometer with no FluidField registered is a configuration
    error, caught at transform build — not a zero reading."""
    import pytest
    w = World()
    c = Craft("probe")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(Barometer("baro"))
    w.add_craft(c, position=(0.0, 0.0, 100.0))
    with pytest.raises(ValueError, match="FluidField"):
        Sim(w)


def test_noisy_barometer_is_an_ekf_sensor():
    """A nonzero pressure_noise σ gives the EKF a measurement R, so the
    barometer can be used as a sensor; a noiseless one cannot (no R)."""
    # Noisy: the EKF builds its measurement update for the barometer.
    _, w_noisy, _ = _baro_world(1000.0, pressure_noise_sigma=50.0)
    TargetNumpy(EKF(w_noisy, sensors=["probe.baro.pressure"]))

    # Noiseless: selecting it as a sensor has no R channel → rejected.
    _, w_clean, _ = _baro_world(1000.0, pressure_noise_sigma=0.0)
    with pytest.raises(Exception):
        TargetNumpy(EKF(w_clean, sensors=["probe.baro.pressure"]))
