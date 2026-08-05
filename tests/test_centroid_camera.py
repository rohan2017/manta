"""CentroidCamera — size-independent bearing measurements, EKF triangulation.

A centroid is the projection of the target's centre: a pure bearing that does
NOT depend on the target's size. One camera pins the target to a ray; spacing
several apart and folding their centroids as EKF measurements makes the
Kalman update the noise-weighted intersection of the rays — triangulation,
with no triangulation code. Folds are independent, so cameras can update the
estimate asynchronously (a subset per tick).
"""

import numpy as np
import pytest

from manta import ALL, Craft, EKF, Sim, TargetNumpy, World
from manta.fields import GravityField, OpticalField
from manta.parts import CentroidCamera, Mass, OpticalSource

# A ground "tracker": cameras spread over a wide baseline, all looking up
# (+z), watching a target high overhead. The wide baseline relative to the
# range is what makes depth well-observable (a real phototheodolite array).
CAM_XY = [(0.0, 0.0), (600.0, 0.0), (0.0, 600.0), (600.0, 600.0)]
TGT = (300.0, 300.0, 600.0)
CAM_NAMES = [f"c{i}" for i in range(len(CAM_XY))]


def _world(*, pixel_sigma=1.0, semi=(3.0, 3.0, 3.0), tgt=TGT):
    tracker = Craft("tracker")
    tracker.add(Mass("base", mass=1.0))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        tracker.add(CentroidCamera(nm, width=1024, height=1024, hfov_deg=100,
                                   pixel_sigma=pixel_sigma, mount_offset=(x, y, 0)))
    target = Craft("target")
    target.add(Mass("body", mass=1.0))
    target.add(OpticalSource("hull", semi_axes=semi, label=1))
    w = (World().add_field(GravityField(g=(0, 0, 0)))   # static target
                .add_field(OpticalField()))
    w.add_craft(tracker, position=(0, 0, 0))
    w.add_craft(target, position=tgt)
    return w


def _bearings(w):
    """One frame of true centroids: {cam: {'u':.., 'v':.., 'vis':bool}}."""
    truth = TargetNumpy(Sim(w))
    truth.step(1e-3)
    out = truth.outputs()["tracker"]
    g = lambda k: float(np.asarray(out[k]).ravel()[0])
    return {nm: {"u": g(f"{nm}.target_hull_u"), "v": g(f"{nm}.target_hull_v"),
                 "vis": g(f"{nm}.target_hull_vis") > 0.5} for nm in CAM_NAMES}


def _sensors(names):
    return [f"tracker.{nm}.target_hull_{c}" for nm in names for c in ("u", "v")]


def _tan(spec, name):
    s = next(s for s in spec.slots if s.name == name)
    return s.tangent_offset, s.tangent_offset + s.tangent_dim


def _make_ekf(w, names, *, p0, pos_var=4e3):
    ekf = TargetNumpy(EKF(w, sensors=_sensors(names)))
    spec = ekf.spec
    # Pin EVERYTHING tiny — the ground array is surveyed, so its camera poses
    # are known; leaving them uncertain makes the bearing rays uncertain and
    # the target gain collapses. Then open up only the target position so the
    # rays pin it.
    ekf.reset(state={"target": {"position": np.asarray(p0, float)}},
              P=np.eye(spec.tangent_dim) * 1e-8)
    P = ekf.P
    a, b = _tan(spec, "target.position")
    for i in range(a, b):
        P[i, i] = pos_var
    ekf.reset(state=ekf.state_dict(), P=P)
    return ekf


def _converge(ekf, bear, names_per_round, *, rounds=20):
    """Run the filter as a recursive estimator: each round re-inflates the
    TARGET position (a tiny predict with target-only Q) then folds the given
    cameras once. The re-inflation re-linearizes h at the updated estimate
    (iterated-EKF / Gauss-Newton) instead of over-counting one frozen
    linearization. `names_per_round` is round_index -> list of camera names."""
    spec = ekf.spec
    q = np.full(spec.tangent_dim, 1e-12)       # keep the array pinned
    a, b = _tan(spec, "target.position")
    q[a:b] = 0.25                              # re-open only the target
    Q = np.diag(q)
    for r in range(rounds):
        ekf.predict(dt=1e-3, Q=Q)
        for nm in names_per_round(r):
            for c in ("u", "v"):
                ekf.update(f"tracker.{nm}.target_hull_{c}",
                           np.array([bear[nm][c]]))
    return np.asarray(ekf.state_dict()["target"]["position"]).ravel()


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------

def test_centroid_declares_uv_noise_channels():
    w = _world(pixel_sigma=1.5)
    w.finalize()
    cam = next(p for c in w.crafts if c.name == "tracker"
               for p in c.parts if p.name == "c0")
    assert set(cam.noise_declarations()) == {"target_hull_u_noise",
                                             "target_hull_v_noise"}
    # bearing-only camera: no size/range component in its outputs
    outs = cam.output_declarations()
    assert set(outs) == {"target_hull_u", "target_hull_v", "target_hull_vis"}


# ---------------------------------------------------------------------------
# Triangulation — the Kalman update IS the ray intersection
# ---------------------------------------------------------------------------

def test_all_cameras_visible():
    b = _bearings(_world())
    assert all(b[nm]["vis"] for nm in CAM_NAMES)


def test_two_cameras_triangulate_target():
    # Spaced cameras → the bearings intersect at the target. From a badly
    # displaced guess (200 m off, mostly in depth), folding two cameras pins
    # all three position DOF.
    w = _world(pixel_sigma=1.0)
    bear = _bearings(w)
    ekf = _make_ekf(w, CAM_NAMES[:2], p0=(350.0, 250.0, 450.0))
    est = _converge(ekf, bear, lambda r: CAM_NAMES[:2])
    assert np.linalg.norm(est - np.array(TGT)) < 2.0


def test_one_camera_is_rank_deficient_in_range():
    # A single bearing constrains the two image DOF but NOT depth along the
    # ray. The cross-range error collapses; the range error does not.
    w = _world(pixel_sigma=1.0)
    bear = _bearings(w)
    ekf = _make_ekf(w, [CAM_NAMES[0]], p0=(360.0, 260.0, 420.0))  # off in all axes
    est = _converge(ekf, bear, lambda r: [CAM_NAMES[0]], rounds=40)
    # c0 at the origin sees the target up-and-over; its single ray pins the
    # two cross-ray DOF but leaves depth along the ray unobserved, so the
    # estimate slides onto the ray (small perpendicular error) yet keeps a
    # large error along it.
    cam = np.array([CAM_XY[0][0], CAM_XY[0][1], 0.0])
    ray = (np.array(TGT) - cam) / np.linalg.norm(np.array(TGT) - cam)
    err = est - np.array(TGT)
    perp = err - np.dot(err, ray) * ray
    assert np.linalg.norm(perp) < 3.0           # on the ray (cross-range pinned)
    assert abs(np.dot(err, ray)) > 40.0         # range along the ray unobserved


def test_size_independent():
    # The centroid does not depend on the target's size: a 10× larger target
    # triangulates to the same position from the same bearings model.
    est = []
    for semi in ((3.0, 3.0, 3.0), (30.0, 30.0, 30.0)):
        w = _world(pixel_sigma=1.0, semi=semi)
        bear = _bearings(w)
        ekf = _make_ekf(w, CAM_NAMES[:3], p0=(350.0, 250.0, 450.0))
        est.append(_converge(ekf, bear, lambda r: CAM_NAMES[:3]))
    assert np.linalg.norm(est[0] - est[1]) < 0.5     # size made no difference


def test_asynchronous_partial_folds_converge():
    # Cameras report at different times — fold ONE camera per tick, cycling
    # through them. Each fold is an independent measurement, so the estimate
    # still converges to the ray intersection.
    w = _world(pixel_sigma=1.0)
    bear = _bearings(w)
    ekf = _make_ekf(w, CAM_NAMES, p0=(350.0, 250.0, 450.0))
    # one different camera per round — partial captures, never all at once
    est = _converge(ekf, bear, lambda r: [CAM_NAMES[r % len(CAM_NAMES)]],
                    rounds=60)
    assert np.linalg.norm(est - np.array(TGT)) < 2.0
