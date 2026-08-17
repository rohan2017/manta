"""Surface-piercing ``DisplacementHull`` physics and lowering contracts."""

from __future__ import annotations

import math
import shutil
import subprocess

import casadi as ca
import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.codegen.cpp.kernels import emit_kernel_list
from manta.fields import FluidField, GravityField, UniformFluid, below_surface
from manta.parts import DisplacementHull, Mass

RHO = 1000.0
G = 9.81


def _small_hull(name="hull", *, resolution=(3, 1, 4), **kwargs):
    return DisplacementHull(
        name,
        dimensions=(2.0, 1.0, 0.8),
        displacement_volume=0.8,
        sample_resolution=resolution,
        **kwargs,
    )


def _flat_surface_world(
    craft,
    *,
    density=RHO,
    current=(0, 0, 0),
    position=(0, 0, 0),
    orientation=(1, 0, 0, 0),
    angular_velocity=(0, 0, 0),
):
    fluid = FluidField().add(
        UniformFluid(
            density=density,
            velocity=current,
            viscosity=1e-3,
            membership=below_surface(lambda p, t: p._mx[2], 0.005),
        )
    )
    world = World().add_field(GravityField(g=(0, 0, -G))).add_field(fluid)
    world.add_craft(
        craft,
        position=position,
        orientation=orientation,
        angular_velocity=angular_velocity,
    )
    return world


def test_quadrature_preserves_calibrated_volume_area_and_symmetry():
    hull = _small_hull(resolution=(5, 2, 8))
    assert len(hull.samples) == 5 * 2 * 8
    assert sum(s.volume for s in hull.samples) == pytest.approx(0.8, abs=1e-14)
    np.testing.assert_allclose(
        np.sum([s.areas for s in hull.samples], axis=0),
        hull.reference_areas,
        atol=1e-14,
    )
    centroid = sum(np.asarray(s.offset) * s.volume for s in hull.samples) / 0.8
    np.testing.assert_allclose(centroid, (0, 0, 0), atol=1e-15)


def test_hydrostatic_calibration_is_independent_of_envelope_volume():
    default = DisplacementHull(
        "default", dimensions=(2.0, 1.0, 0.8), sample_resolution=(3, 1, 4)
    )
    calibrated = _small_hull("calibrated")
    assert default.displacement_volume == pytest.approx(math.pi * 2 * 1 * 0.8 / 6)
    assert calibrated.displacement_volume == pytest.approx(0.8)
    assert calibrated.geometric_volume == pytest.approx(default.geometric_volume)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dimensions": (0, 1, 1)}, "dimensions"),
        ({"displacement_volume": 0}, "displacement_volume"),
        ({"drag_coefficients": (0.2, -1, 1)}, "drag_coefficients"),
        ({"sample_resolution": (1, 1, 4)}, "axial"),
        ({"sample_resolution": (2, 1, 5)}, "even circumferential"),
    ],
)
def test_configuration_rejects_nonphysical_values(kwargs, match):
    base = {"dimensions": (2, 1, 1)}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DisplacementHull("bad", **base)


@pytest.mark.parametrize("waterline", [-0.15, 0.0, 0.15])
def test_calm_water_mass_selects_an_equilibrium_draft(waterline):
    """Light, middle, and heavy craft trim where their wet volume matches."""
    hull = _small_hull()
    wet_volume = hull.displaced_volume_below(waterline)
    craft = Craft("boat")
    craft.add(Mass("body", mass=RHO * wet_volume, moi=(20, 25, 30)))
    craft.add(hull)
    # The world waterline expressed in hull coordinates is -position_z.
    sim = TargetNumpy(Sim(_flat_surface_world(craft, position=(0, 0, -waterline))))

    for _ in range(25):
        sim.step(0.002)
    state = sim.state["boat"]
    np.testing.assert_allclose(state["position"], (0, 0, -waterline), atol=1e-9)
    np.testing.assert_allclose(state["velocity"], (0, 0, 0), atol=1e-9)


@pytest.mark.parametrize("axis", [0, 1])
def test_heel_and_trim_generate_righting_moment(axis):
    hull = _small_hull(resolution=(5, 2, 8), hydrostatic_offset=(0, 0, 0.08))
    wet_volume = hull.displaced_volume_below(0.0)
    craft = Craft("boat")
    # The mass below the hydrostatic cloud provides unambiguous positive GM
    # for both roll and pitch while the wet sample set changes with attitude.
    craft.add(
        Mass(
            "body", mass=RHO * wet_volume, moi=(20, 25, 30), mount_offset=(0, 0, -0.15)
        )
    )
    craft.add(hull)
    angle = math.radians(6)
    q = (
        [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
        if axis == 0
        else [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
    )
    sim = TargetNumpy(Sim(_flat_surface_world(craft, orientation=q)))

    sim.step(0.002)
    # Positive heel/trim must acquire angular velocity toward level.
    assert sim.state["boat"]["angular_velocity"][axis] < 0.0


def test_distributed_lateral_drag_damps_yaw():
    craft = Craft("boat")
    craft.add(Mass("body", mass=500, moi=(20, 25, 30)))
    craft.add(_small_hull())
    sim = TargetNumpy(
        Sim(
            _flat_surface_world(
                craft, position=(0, 0, -2), angular_velocity=(0, 0, 0.4)
            )
        )
    )
    before = sim.state["boat"]["angular_velocity"][2]
    sim.step(0.01)
    after = sim.state["boat"]["angular_velocity"][2]
    assert 0.0 < after < before


def test_current_acts_through_local_drag_samples():
    craft = Craft("boat")
    craft.add(Mass("body", mass=500, moi=(20, 25, 30)))
    craft.add(_small_hull())
    sim = TargetNumpy(
        Sim(_flat_surface_world(craft, current=(1.2, 0, 0), position=(0, 0, -2)))
    )
    sim.step(0.01)
    assert sim.state["boat"]["velocity"][0] > 0.0


def test_moving_wave_boundary_changes_the_wet_sample_set():
    """The composite consumes an ordinary time/position-varying fluid field."""
    hull = _small_hull()
    wet_volume = hull.displaced_volume_below(0.0)
    craft = Craft("boat")
    craft.add(Mass("body", mass=RHO * wet_volume, moi=(20, 25, 30)))
    craft.add(hull)
    amplitude, omega = 0.3, 2.0
    fluid = FluidField().add(
        UniformFluid(
            density=RHO,
            viscosity=1e-3,
            membership=below_surface(
                lambda p, t: p._mx[2] - amplitude * ca.sin(omega * t._mx), 0.005
            ),
        )
    )
    world = World().add_field(GravityField(g=(0, 0, -G))).add_field(fluid)
    world.add_craft(craft)
    sim = TargetNumpy(Sim(world))

    # At t=pi/(4 omega), the rising surface wets more volume and accelerates
    # the otherwise-trimmed craft upward.
    sim.step(0.001, t=math.pi / (4 * omega))
    assert sim.state["boat"]["velocity"][2] > 0.0


def _ellipsoid_fraction_below(normalized_z):
    s = float(normalized_z)
    return 0.75 * (s - s**3 / 3.0 + 2.0 / 3.0)


def test_partial_volume_converges_with_sample_resolution():
    waterline = 0.11
    # height=.8 => normalized ellipsoid coordinate s=z/(height/2)
    exact = 0.8 * _ellipsoid_fraction_below(waterline / 0.4)
    coarse = _small_hull("coarse", resolution=(3, 1, 8))
    fine = _small_hull("fine", resolution=(9, 4, 64))
    coarse_error = abs(coarse.displaced_volume_below(waterline) - exact)
    fine_error = abs(fine.displaced_volume_below(waterline) - exact)
    assert fine_error < coarse_error
    assert fine_error < 0.02 * 0.8


def _backend_world():
    craft = Craft("boat")
    craft.add(Mass("body", mass=500, moi=(20, 25, 30)))
    craft.add(_small_hull(resolution=(2, 1, 4)))
    fluid = FluidField().add_uniform(RHO, viscosity=1e-3)
    world = (
        World(name="hull_backend")
        .add_field(GravityField(g=(0, 0, -G)))
        .add_field(fluid)
    )
    world.add_craft(craft, position=(0, 0, -2), velocity=(0.5, 0.2, 0))
    return world


def test_numpy_and_jax_step_parity():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from manta import TargetJax

    model = Sim(_backend_world())
    np_runtime = TargetNumpy(model)
    jax_runtime = TargetJax(model)
    module = model.module()
    x = np.asarray(jax_runtime.initial_state())
    u = np.zeros(module.port("u").size)
    noise = np.zeros(module.port("noise").size)
    got_jax = np.asarray(jax_runtime.kernel("step")(x, u, noise, 0.002, 0)[0])
    np_runtime.step(0.002)
    from manta.ir.state_spec import flatten_nested

    got_numpy = module.spec.pack_any(flatten_nested(np_runtime.state))
    np.testing.assert_allclose(got_jax.ravel(), got_numpy.ravel(), atol=1e-11)


@pytest.mark.cpp
def test_generated_c_kernel_compiles(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    module = Sim(_backend_world()).deploy_module()
    paths = emit_kernel_list(module.functions.values(), tmp_path, basename=module.name)
    obj = tmp_path / "hull.o"
    proc = subprocess.run(
        [cc, "-c", "-O2", "-Wno-unused-parameter", str(paths["c"]), "-o", str(obj)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert obj.exists()
