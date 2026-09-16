"""Local waves share the planet model's linear deep-water wave law."""

import math

import casadi as ca
import numpy as np
import pytest

from manta.fields import FlatOcean
from manta.fields.wave_ocean import WaveOcean
from manta.ir.frames import WorldFrame
from manta.ir.types import Scalar, Vec3


def sample(ocean, xyz, time=0.0):
    p, t = ca.MX.sym("p", 3), ca.MX.sym("t")
    point, stamp = Vec3[WorldFrame].from_mx(p), Scalar.from_mx(t)
    state = ocean.contribute_at_sym(point, stamp)
    f = ca.Function(
        "ocean",
        [p, t],
        [ocean.membership(point, stamp), state.pressure, state.velocity.mx],
    )
    wet, pressure, flow = f(xyz, time)
    return float(wet), float(pressure), np.array(flow).ravel()


def test_empty_wave_ocean_matches_flat_ocean():
    for z in (-10.0, -0.03, 0.0, 0.03, 1.0):
        a, b = sample(WaveOcean(), (0, 0, z)), sample(FlatOcean(), (0, 0, z))
        np.testing.assert_allclose(a[:2], b[:2])
        np.testing.assert_allclose(a[2], b[2])


def test_boundary_orbits_and_pressure_share_direction_phase_and_dispersion():
    amplitude, wavelength, gravity = 0.12, 12.0, 9.80665
    ocean = WaveOcean(
        components=((amplitude, wavelength, (3, 4), 0.4),), gravity=gravity
    )
    k, t = 2 * math.pi / wavelength, 0.37
    omega = math.sqrt(gravity * k)
    x, y = 1.2, -0.5
    phase = k * (0.6 * x + 0.8 * y) - omega * t + 0.4
    eta = amplitude * math.cos(phase)
    assert sample(ocean, (x, y, eta), t)[0] == pytest.approx(0.5)
    for depth in (1.0, 10.0):
        wet, pressure, flow = sample(ocean, (x, y, -depth), t)
        decay = math.exp(-k * depth)
        assert wet == 1.0
        np.testing.assert_allclose(
            flow,
            amplitude
            * omega
            * decay
            * np.array((0.6 * math.cos(phase), 0.8 * math.cos(phase), math.sin(phase))),
        )
        head = depth + eta * decay
        rounded_head = (head + math.sqrt(head**2 + ocean.surface_blend**2)) / 2
        assert pressure == pytest.approx(101325 + 1025 * gravity * rounded_head)


@pytest.mark.parametrize(
    "component",
    (
        (-1, 12, (1, 0), 0),
        (1, 0, (1, 0), 0),
        (1, 12, (0, 0), 0),
        (1, 12, (1, 0), float("nan")),
    ),
)
def test_invalid_wave_components_are_rejected(component):
    with pytest.raises(ValueError, match="wave components"):
        WaveOcean(components=(component,))
