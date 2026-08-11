"""A camera can be pointed.

`mount_orientation` reached the kinematic pass and most parts, but not
the camera constructors — which forward `mount_offset` only. For the one
part family where pointing IS the job, that meant a camera could look
nowhere but along its parent's +z unless it was hung off a joint, and
`BBoxCamera("cam", mount_orientation=...)` was a TypeError.

These tests pin the fix from the outside: a yawed camera sees a target
a forward-looking one cannot, and its bearing follows the mount.
"""

import math

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField, OpticalField
from manta.parts import BBoxCamera, CentroidCamera, Mass, OpticalSource

# Target off to the craft's +x, at the same height — 90° off the default
# boresight (+z), so only a rotated camera can see it.
TARGET = (30.0, 0.0, 0.0)


def _quat_about_y(deg):
    """Rotation about +y takes the camera's +z boresight toward +x."""
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _world(camera):
    tracker = Craft("tracker")
    tracker.add(Mass("base", mass=1.0))
    tracker.add(camera)
    target = Craft("target")
    target.add(Mass("body", mass=1.0))
    target.add(OpticalSource("hull", semi_axes=(1.0, 1.0, 1.0), label=1))
    w = (World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
                .add_field(OpticalField()))
    w.add_craft(tracker, position=(0.0, 0.0, 0.0))
    w.add_craft(target, position=TARGET)
    return w


def _observe(camera, keys):
    sim = TargetNumpy(Sim(_world(camera)))
    sim.step(1e-3)
    out = sim.outputs()["tracker"]
    return {k: float(np.asarray(out[f"cam.target_hull_{k}"]).ravel()[0])
            for k in keys}


# ---------------------------------------------------------------------------
# The constructors accept it at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [BBoxCamera, CentroidCamera])
def test_constructor_accepts_mount_orientation(cls):
    cam = cls("cam", width=640, height=480, hfov_deg=60,
              mount_orientation=_quat_about_y(90))
    assert np.allclose(cam.mount_orientation, _quat_about_y(90))


@pytest.mark.parametrize("cls", [BBoxCamera, CentroidCamera])
def test_default_mount_is_upright(cls):
    assert cls("cam", width=640, height=480).mounted_upright


# ---------------------------------------------------------------------------
# …and it actually points the camera
# ---------------------------------------------------------------------------

def test_a_yawed_camera_sees_what_a_forward_one_cannot():
    forward = CentroidCamera("cam", width=640, height=480, hfov_deg=60)
    turned = CentroidCamera("cam", width=640, height=480, hfov_deg=60,
                            mount_orientation=_quat_about_y(90))
    assert _observe(forward, ["vis"])["vis"] < 0.5
    assert _observe(turned, ["vis"])["vis"] > 0.5


def test_a_turned_camera_puts_the_target_on_its_boresight():
    """Rotated to look straight at it, the centroid lands on the principal
    point — the pointing is exact, not merely 'in frame'."""
    w, h = 640, 480
    turned = CentroidCamera("cam", width=w, height=h, hfov_deg=60,
                            mount_orientation=_quat_about_y(90))
    got = _observe(turned, ["u", "v"])
    assert got["u"] == pytest.approx(w / 2.0, abs=1e-6)
    assert got["v"] == pytest.approx(h / 2.0, abs=1e-6)


def test_a_partial_yaw_moves_the_bearing_off_centre_in_the_right_direction():
    """Under-rotate and the target sits off-centre, on the side the
    remaining angle puts it — a sign check on the mount composition."""
    w = 640
    under = CentroidCamera("cam", width=w, height=480, hfov_deg=90,
                           mount_orientation=_quat_about_y(70))
    over = CentroidCamera("cam", width=w, height=480, hfov_deg=90,
                          mount_orientation=_quat_about_y(110))
    u_under = _observe(under, ["u"])["u"]
    u_over = _observe(over, ["u"])["u"]
    assert u_under > w / 2.0            # target still ahead of boresight
    assert u_over < w / 2.0             # overshot past it
    # Symmetric about the principal point for symmetric mis-pointing.
    assert (u_under - w / 2.0) == pytest.approx(w / 2.0 - u_over, rel=1e-9)


def test_a_bbox_camera_is_pointable_too():
    turned = BBoxCamera("cam", width=640, height=480, hfov_deg=60,
                        mount_orientation=_quat_about_y(90))
    assert _observe(turned, ["vis"])["vis"] > 0.5
