"""OpticalField + Camera + field-source parts.

A Camera bounding-boxes the OpticalField's semantic ellipsoids; the three
FieldSource parts (gravity / magnetic / optical) emit craft-anchored
disturbances onto the world fields so other craft feel or see them.
"""

import numpy as np
import pytest

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields import GravityField, MagField, OpticalField
from manta.parts import (
    BBoxCamera, GravitySource, MagneticSource, Mass, Magnetometer,
    OpticalSource, PositionSensor,
)


def _probe_with_camera(**cam_kw):
    c = Craft("probe")
    c.add(Mass("body", mass=1.0))
    c.add(BBoxCamera("cam", **cam_kw))
    return c


# ---------------------------------------------------------------------------
# Camera bounding-box geometry
# ---------------------------------------------------------------------------

def test_static_ellipsoid_box_centered_and_aspect():
    """An on-axis ellipsoid projects to a box centered in the image with
    an aspect ratio matching its semi-axes."""
    w = World().add_field(
        OpticalField().add_ellipsoid((0, 0, 10), (2.0, 1.0, 1.0), label=3))
    w.add_craft(_probe_with_camera(width=640, height=480, hfov_deg=70),
                position=(0, 0, 0))
    sim = TargetNumpy(Sim(w))
    sim.step(0.01)
    o = sim.outputs()["probe"]
    name = next(k for k in o if k.endswith("_xmin"))[:-len("_xmin")]
    g = lambda s: float(np.asarray(o[f"{name}_{s}"]).ravel()[0])
    xc = 0.5 * (g("xmin") + g("xmax"))
    yc = 0.5 * (g("ymin") + g("ymax"))
    assert abs(xc - 320) < 1.0 and abs(yc - 240) < 1.0      # centered
    aspect = (g("xmax") - g("xmin")) / (g("ymax") - g("ymin"))
    assert abs(aspect - 2.0) < 0.05                          # 2:1 semi-axes
    assert g("vis") == 1.0


def test_box_shrinks_with_distance():
    """Doubling the range halves the projected box size."""
    def box_w(dist):
        w = World().add_field(
            OpticalField().add_ellipsoid((0, 0, dist), (1.0, 1.0, 1.0)))
        w.add_craft(_probe_with_camera(), position=(0, 0, 0))
        sim = TargetNumpy(Sim(w)); sim.step(0.01)
        o = sim.outputs()["probe"]
        nm = next(k for k in o if k.endswith("_xmin"))[:-len("_xmin")]
        return (float(np.asarray(o[f"{nm}_xmax"]).ravel()[0])
                - float(np.asarray(o[f"{nm}_xmin"]).ravel()[0]))
    assert box_w(20.0) == pytest.approx(0.5 * box_w(10.0), rel=0.05)


def test_behind_camera_not_visible():
    w = World().add_field(
        OpticalField().add_ellipsoid((0, 0, -10), (2.0, 1.0, 1.0)))
    w.add_craft(_probe_with_camera(), position=(0, 0, 0))
    sim = TargetNumpy(Sim(w)); sim.step(0.01)
    o = sim.outputs()["probe"]
    vis = float(np.asarray(next(o[k] for k in o if k.endswith("_vis"))).ravel()[0])
    assert vis == 0.0


def test_offscreen_not_visible():
    """An object in front of the camera but projecting outside the image
    rectangle reports vis = 0."""
    w = World().add_field(
        OpticalField().add_ellipsoid((100.0, 0, 5), (0.5, 0.5, 0.5)))
    w.add_craft(_probe_with_camera(width=640, height=480, hfov_deg=70),
                position=(0, 0, 0))
    sim = TargetNumpy(Sim(w)); sim.step(0.01)
    o = sim.outputs()["probe"]
    vis = float(np.asarray(next(o[k] for k in o if k.endswith("_vis"))).ravel()[0])
    assert vis == 0.0


def test_optical_source_tracks_moving_target():
    """A target carrying an OpticalSource is seen by another craft's
    camera, and its box shifts as the target moves laterally."""
    def box_center_x(tx):
        w = World().add_field(OpticalField())
        w.add_craft(_probe_with_camera(), position=(0, 0, 0))
        tgt = Craft("target"); tgt.add(Mass("b", mass=1.0))
        tgt.add(OpticalSource("hull", semi_axes=(2, 1, 1), label=7))
        w.add_craft(tgt, position=(tx, 0, 15))
        sim = TargetNumpy(Sim(w)); sim.step(0.01)
        o = sim.outputs()["probe"]
        return 0.5 * (float(np.asarray(o["cam.target_hull_xmin"]).ravel()[0])
                      + float(np.asarray(o["cam.target_hull_xmax"]).ravel()[0]))
    assert box_center_x(0.0) == pytest.approx(320, abs=1.0)   # centered
    assert box_center_x(3.0) > 360                            # shifted right


def test_camera_does_not_see_own_craft():
    """A camera and an OpticalSource on the SAME craft: the camera must
    not box itself."""
    c = Craft("solo")
    c.add(Mass("body", mass=1.0))
    c.add(BBoxCamera("cam"))
    c.add(OpticalSource("self", semi_axes=(1, 1, 1)))
    w = World().add_field(OpticalField())
    w.add_craft(c, position=(0, 0, 0))
    sim = TargetNumpy(Sim(w)); sim.step(0.01)
    o = sim.outputs().get("solo", {})
    assert not any(k.startswith("cam.") for k in o)           # no boxes


# ---------------------------------------------------------------------------
# Gravity / magnetic sources
# ---------------------------------------------------------------------------

def test_gravity_source_pulls_other_craft():
    w = World().add_field(GravityField(g=(0, 0, 0)))   # only the source
    rock = Craft("rock"); rock.add(Mass("c", mass=1.0))
    rock.add(GravitySource("g", GM=4.0e5))
    probe = Craft("probe"); probe.add(Mass("b", mass=1.0))
    w.add_craft(rock, position=(0, 0, 0))
    w.add_craft(probe, position=(100.0, 0, 0))
    sim = TargetNumpy(Sim(w))
    for _ in range(20):
        sim.step(0.02)
    x = float(sim.state["probe"]["position"][0])
    assert x < 100.0                                   # pulled inward


def test_magnetic_source_on_axis_dipole():
    """On-axis field of a body dipole: B = 2·μ₀/4π·m/r³."""
    w = World().add_field(MagField())
    a = Craft("a"); a.add(Mass("b", mass=1.0)); a.add(Magnetometer("mag"))
    b = Craft("bee"); b.add(Mass("b", mass=1.0))
    b.add(MagneticSource("motor", moment=(0, 0, 0.5)))
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(0, 0, 1.0))               # 1 m on +z
    sim = TargetNumpy(Sim(w)); sim.step(0.01)
    B = np.asarray(sim.outputs()["a"]["mag.B"]).ravel()
    expected = 2 * 1e-7 * 0.5 / 1.0**3
    assert B[2] == pytest.approx(expected, rel=1e-3)
    assert abs(B[0]) < 1e-12 and abs(B[1]) < 1e-12


def test_magnetic_source_moment_rotates_with_craft():
    """A body-fixed moment turns with its craft, so the reading changes."""
    def reading(orientation):
        w = World().add_field(MagField())
        a = Craft("a"); a.add(Mass("b", mass=1.0)); a.add(Magnetometer("mag"))
        b = Craft("bee"); b.add(Mass("b", mass=1.0))
        b.add(MagneticSource("motor", moment=(0, 0, 0.5)))
        w.add_craft(a, position=(0, 0, 0))
        w.add_craft(b, position=(2.0, 0, 0), orientation=orientation)
        sim = TargetNumpy(Sim(w)); sim.step(0.01)
        return np.asarray(sim.outputs()["a"]["mag.B"]).ravel()
    upright = reading((1, 0, 0, 0))
    # 90° about +y rotates body +z moment toward world +x.
    s = np.sqrt(0.5)
    turned = reading((s, 0, s, 0))
    assert not np.allclose(upright, turned, atol=1e-10)


# ---------------------------------------------------------------------------
# Registration plumbing
# ---------------------------------------------------------------------------

def test_source_registration_idempotent():
    """Building two transforms over one world must not double-register
    the emitted disturbances."""
    w = World().add_field(OpticalField())
    p = _probe_with_camera(); w.add_craft(p, position=(0, 0, 0))
    tgt = Craft("t"); tgt.add(Mass("b", mass=1.0))
    tgt.add(OpticalSource("hull", semi_axes=(1, 1, 1)))
    tgt.add(PositionSensor("gps", position_noise_sigma=0.1))
    w.add_craft(tgt, position=(0, 0, 10))
    Sim(w)
    # Camera boxes are noiseless → not EKF sensors; track the gps only.
    EKF(w, sensors=["t.gps.position"])
    Sim(w)
    assert len(w.get_field(OpticalField).ellipsoids) == 1


def test_field_source_adds_no_wrench():
    """A source is a pure emitter — it must not perturb its carrier's
    dynamics (a free body stays put)."""
    w = World().add_field(MagField())
    c = Craft("c"); c.add(Mass("b", mass=1.0))
    c.add(MagneticSource("motor", moment=(0, 0, 0.5)))
    w.add_craft(c, position=(0, 0, 0))
    sim = TargetNumpy(Sim(w))
    for _ in range(10):
        sim.step(0.01)
    assert np.allclose(np.asarray(sim.state["c"]["position"]).ravel(), 0.0)


def test_source_requires_its_field_registered():
    """A source contributes to a field — it does not create one. Using a
    GravitySource with no GravityField registered is a config error."""
    w = World()                                    # no GravityField
    rock = Craft("rock"); rock.add(Mass("c", mass=1.0))
    rock.add(GravitySource("g", GM=4.0e5))
    w.add_craft(rock, position=(0, 0, 0))
    with pytest.raises(ValueError, match="GravityField"):
        Sim(w)


def test_optical_field_is_not_a_superposition():
    # OpticalField is an enumerated field, not a SuperposedField: it
    # carries its ellipsoids for the camera to project, and has no
    # summed value_at_sym at all.
    from manta.fields.base import SuperposedField
    f = OpticalField().add_ellipsoid((0, 0, 1), (1, 1, 1))
    assert not isinstance(f, SuperposedField)
    assert not hasattr(f, "value_at_sym")
    assert len(f.ellipsoids) == 1
