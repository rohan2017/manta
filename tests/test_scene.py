"""Scene — local-frame placement / reporting / rendering convenience.

A `Scene` is a frame fixed in a planet's body frame. These tests verify
the numeric I/O translation it provides:

  * `at_rest(...)` places a craft co-rotating with the planet at a local
    scene coordinate (orbital velocity + body spin rate filled in).
  * `relative(...)` is the exact inverse on the output side: a craft
    placed with `at_rest` reports back its placement coordinate, identity-
    ish attitude, and ~zero velocity / body rate relative to the scene.
  * `world_pose(...)` returns the scene's own world transform.
"""

import numpy as np

from manta import Planet
from manta.planets import Earth, Scene
from manta.ir._rotation import quat_to_rotmat_np


def _rotmat(q):
    return quat_to_rotmat_np(np.asarray(q, dtype=float))


def test_scene_at_rest_on_pole_is_at_rest_and_co_spins():
    """A scene at the north pole (planet at the origin): a craft placed at
    rest sits on the spin axis (ω × r = 0) and co-spins (body rate = ω)."""
    omega, R = 7.272e-5, 6.371e6
    earth = Planet(rotation_axis=(0, 0, 1), omega=omega)
    scene = earth.scene_at((0.0, 0.0, R))
    ks = scene.at_rest()
    np.testing.assert_allclose(ks["position"], (0, 0, R), atol=1e-6)
    np.testing.assert_allclose(ks["velocity"], (0, 0, 0), atol=1e-9)
    np.testing.assert_allclose(ks["angular_velocity"], (0, 0, omega), atol=1e-18)


def test_scene_at_rest_body_rate_reconstructs_world_spin():
    """Whatever the local position + heading, the body rate rotated by the
    placed attitude equals the planet's WorldFrame spin — the craft is
    genuinely rigid on the planet."""
    R = Earth.R_EQ
    lat = np.radians(45.0)
    earth = Earth()                                   # real +z axis, sidereal
    scene = earth.scene_at((R * np.cos(lat), 0.0, R * np.sin(lat)))
    omega_w = earth.omega_vec_world()
    for pos in ((0, 0, 0), (1.0, 2.0, -0.3)):
        for h in (0.0, np.radians(35.0), np.radians(-120.0)):
            ks = scene.at_rest(pos, heading=h)
            Rwc = _rotmat(ks["orientation"])
            np.testing.assert_allclose(Rwc @ np.asarray(ks["angular_velocity"]),
                                       omega_w, atol=1e-15)


def test_scene_relative_inverts_at_rest():
    """`relative` is the inverse of `at_rest`: a craft placed at scene
    coordinate `pos` with yaw `h` reports back `pos`, a pure yaw-`h`
    attitude, and ~zero velocity / body rate relative to the scene."""
    R = Earth.R_EQ
    lat = np.radians(30.0)
    earth = Earth()
    scene = earth.scene_at((R * np.cos(lat), 0.0, R * np.sin(lat)))
    pos, h = (1.0, 2.0, -0.3), np.radians(35.0)
    ks = scene.at_rest(pos, heading=h)
    state = {k: np.asarray(v) for k, v in ks.items()}
    rel = scene.relative(state, t=0.0)
    np.testing.assert_allclose(rel["position"], pos, atol=1e-6)
    # Attitude relative to the scene is a pure yaw by h about up.
    np.testing.assert_allclose(
        rel["orientation"], (np.cos(h / 2), 0, 0, np.sin(h / 2)), atol=1e-9)
    np.testing.assert_allclose(rel["velocity"], (0, 0, 0), atol=1e-9)
    np.testing.assert_allclose(rel["angular_velocity"], (0, 0, 0), atol=1e-15)


def test_scene_relative_passes_through_part_state():
    """Non-rigid-body keys (joint angles, etc.) pass through untouched."""
    earth = Earth()
    scene = earth.scene_at((Earth.R_EQ, 0.0, 0.0))
    state = dict(scene.at_rest())
    state["wheel.angle"] = 0.7
    rel = scene.relative({k: np.asarray(v) if k != "wheel.angle" else v
                          for k, v in state.items()})
    assert rel["wheel.angle"] == 0.7


def test_scene_world_pose_matches_origin_and_basis():
    """`world_pose` returns the scene origin and an orientation whose +z is
    local up (radial) and +x is local north."""
    R = Earth.R_EQ
    lat = np.radians(45.0)
    earth = Earth()
    anchor = (R * np.cos(lat), 0.0, R * np.sin(lat))
    scene = earth.scene_at(anchor)
    origin, q = scene.world_pose(0.0)
    np.testing.assert_allclose(origin, anchor, atol=1e-6)
    Rws = _rotmat(q)
    up = np.asarray(anchor) / np.linalg.norm(anchor)
    np.testing.assert_allclose(Rws[:, 2], up, atol=1e-12)         # +z = up
    # +x = north = spin axis (+z world) projected into the tangent plane.
    north = np.array([0, 0, 1.0]) - np.dot([0, 0, 1.0], up) * up
    north /= np.linalg.norm(north)
    np.testing.assert_allclose(Rws[:, 0], north, atol=1e-12)


def test_scene_relative_reports_ground_relative_velocity():
    """A craft moving over the ground reports its ground-relative velocity
    in scene coords — the co-rotation (ω × r) is subtracted out."""
    R = Earth.R_EQ
    earth = Earth()
    scene = earth.scene_at((R, 0.0, 0.0))             # on the equator
    base = scene.at_rest()                            # at rest on the ground
    # Add a known ground-relative velocity expressed in the scene frame:
    # +x (north) at 3 m/s. Convert to a world velocity on top of co-rotation.
    Rws = scene.R_world_from_scene(0.0)
    extra_world = Rws @ np.array([3.0, 0.0, 0.0])
    state = {
        "position": np.asarray(base["position"]),
        "orientation": np.asarray(base["orientation"]),
        "velocity": np.asarray(base["velocity"]) + extra_world,
        "angular_velocity": np.asarray(base["angular_velocity"]),
    }
    rel = scene.relative(state, t=0.0)
    np.testing.assert_allclose(rel["velocity"], (3.0, 0.0, 0.0), atol=1e-6)
