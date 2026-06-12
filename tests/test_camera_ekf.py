"""Camera as an EKF measurement source.

With `bbox_sigma > 0` the camera's four box edges become noisy measurements.
A box of a target is a function of BOTH the camera craft's pose and the
target craft's pose, so selecting the edges as EKF sensors yields a single
JOINT filter that couples the two craft blocks — and folding the boxes
recovers the target's 3-D position (bearing from the box center, range from
its apparent size given the known semi-axes).
"""

import numpy as np
import pytest

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields import GravityField, OpticalField
from manta.parts import BBoxCamera, Mass, OpticalSource, PositionSensor

EDGES = ("xmin", "ymin", "xmax", "ymax")
SEMI = (0.5, 0.5, 0.5)


def _world(*, bbox_sigma=1.0, tgt_pos=(-7.0, 0.0, 9.0), tgt_vel=(0.0, 0.0, 0.0),
           g=0.0):
    obs = Craft("obs")
    obs.add(Mass("body", mass=10.0, moi=(1, 1, 1)))
    obs.add(BBoxCamera("cam", width=640, height=480, hfov_deg=100,
                       bbox_sigma=bbox_sigma))
    obs.add(PositionSensor("gps", position_noise_sigma=0.01))
    tgt = Craft("tgt")
    tgt.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    tgt.add(OpticalSource("hull", semi_axes=SEMI, label=1))
    w = (World().add_field(GravityField(g=(0, 0, -g)))
                .add_field(OpticalField()))
    w.add_craft(obs, position=(0, 0, 0))
    w.add_craft(tgt, position=tgt_pos, velocity=tgt_vel)
    return w


def _camera_of(w):
    # Field sources (and so the camera's target list) are wired lazily at
    # the first transform; trigger it so the dynamic declarations exist.
    w.finalize()
    return next(p for c in w.crafts if c.name == "obs"
                for p in c.parts if p.name == "cam")


def _edge_sensors():
    return [f"obs.cam.tgt_hull_{s}" for s in EDGES]


def _tan(spec, name):
    s = next(s for s in spec.slots if s.name == name)
    return s.tangent_offset, s.tangent_offset + s.tangent_dim


# ---------------------------------------------------------------------------
# Noise channels — opt-in, byte-identical when off
# ---------------------------------------------------------------------------

def test_no_noise_channels_when_sigma_zero():
    cam = _camera_of(_world(bbox_sigma=0.0))
    assert cam.noise_declarations() == {}
    # outputs unchanged: 4 edges + vis per target
    outs = cam.output_declarations()
    assert any(k.endswith("_vis") for k in outs)
    assert sum(k.endswith("_xmin") for k in outs) == 1


def test_four_noise_channels_per_target_when_sigma_positive():
    cam = _camera_of(_world(bbox_sigma=1.5))
    nd = cam.noise_declarations()
    assert set(nd) == {f"tgt_hull_{s}_noise" for s in EDGES}
    # the tick-signature walk reads `<name>_sigma` off the owner
    for name in nd:
        assert float(getattr(cam, f"{name}_sigma")) == pytest.approx(1.5)


def test_noise_declarations_is_pure_read():
    # The `<name>_sigma` attributes are materialized by set_targets()
    # (the single mutation point); noise_declarations() never setattr's.
    cam = _camera_of(_world(bbox_sigma=1.5))
    for s in EDGES:
        assert float(getattr(cam, f"tgt_hull_{s}_noise_sigma")) \
            == pytest.approx(1.5)
    delattr(cam, "tgt_hull_xmin_noise_sigma")
    cam.noise_declarations()
    assert not hasattr(cam, "tgt_hull_xmin_noise_sigma")


def test_ekf_rejects_zero_noise_camera_edges():
    # bbox_sigma=0 ⇒ the edge outputs carry no noise ⇒ singular update.
    w = _world(bbox_sigma=0.0)
    with pytest.raises(ValueError, match="noise"):
        EKF(w, sensors=_edge_sensors())


# ---------------------------------------------------------------------------
# Cross-craft coupling
# ---------------------------------------------------------------------------

def test_camera_couples_both_crafts_into_one_block():
    w = _world(bbox_sigma=1.0)
    # The camera measurement spans both craft tangent blocks, so the
    # block-diagonal partition unions everything into a single joint block.
    assert EKF(w, sensors=["obs.gps.position", *_edge_sensors()]).n_blocks == 1
    # Without the camera there is no cross-craft measurement, so the crafts
    # stay in separate dynamics blocks (more than one).
    assert EKF(w, sensors=["obs.gps.position"]).n_blocks > 1


# ---------------------------------------------------------------------------
# The boxes actually locate the target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x0", [-4.0, -10.0])
def test_boxes_recover_target_position(x0):
    # Static scene; feed the true boxes from a laterally displaced guess.
    # The bearing (box center) + size (box extent, known semi-axes) pin the
    # target's 3-D position regardless of which side we start on.
    truth = TargetNumpy(Sim(_world(bbox_sigma=1.0)))
    truth.step(1e-3)
    out = truth.outputs()["obs"]
    z = {s: float(np.asarray(out[f"cam.tgt_hull_{s}"]).ravel()[0]) for s in EDGES}
    assert float(np.asarray(out["cam.tgt_hull_vis"]).ravel()[0]) > 0.5

    ekf = TargetNumpy(EKF(_world(bbox_sigma=1.0), sensors=_edge_sensors()))
    spec = ekf.spec
    ekf.reset(state={"obs": {"position": np.zeros(3)},
                     "tgt": {"position": np.array([x0, 0.0, 9.0])}},
              P=np.eye(spec.tangent_dim) * 1e-3)
    P = ekf.P
    a, b = _tan(spec, "tgt.position")
    for i in range(a, b):
        P[i, i] = 9.0
    ekf.reset(state=ekf.state_dict(), P=P)

    for _ in range(80):
        for s in EDGES:
            ekf.update(f"obs.cam.tgt_hull_{s}", np.array([z[s]]))

    est = np.asarray(ekf.state_dict()["tgt"]["position"]).ravel()
    assert np.linalg.norm(est - np.array([-7.0, 0.0, 9.0])) < 0.5
