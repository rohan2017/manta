"""EKF → LQR estimate transfer: `lqr.control(ekf.state_dict())`.

The LQR maps an estimate to commands by name: `control()` packs the
(possibly partial) estimate over the operating-point base, so slots the
estimate omits sit at the reference (zero error) and a filter that tracks
only a subset of the world still drives a full-world regulator correctly.
This is the name-keyed gather the wired `compute()` path used to do; it now
lives in `control()` / the spec's `pack_any`.
"""

import numpy as np

from manta import (
    EKF,
    LQR,
    POSE,
    TWIST,
    Craft,
    TargetNumpy,
    World,
)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster


def test_omitted_slots_sit_at_reference():
    """An estimate that supplies only position leaves velocity at the
    reference: control() of a position-only dict equals control() of the
    same dict with velocity explicitly set to the reference value."""
    M, G, dt = 2.0, 9.81, 0.02
    c = Craft("c")
    c.add(Mass("body", mass=M))
    for n, f in [("tx", (1, 0, 0)), ("ty", (0, 1, 0)), ("tz", (0, 0, 1))]:
        c.add(Thruster(n, force=f))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10))
    lqr = TargetNumpy(LQR(
        w, x_ref={"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, regulate=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))

    u_partial = lqr.control({"c": {"position": (3.0, -2.0, 7.0)}})
    u_full = lqr.control({"c": {"position": (3.0, -2.0, 7.0),
                                "velocity": (0.0, 0.0, 0.0)}})
    for k in u_full:
        assert abs(u_partial[k] - u_full[k]) < 1e-12


def test_subset_estimate_drives_full_world_regulator():
    """EKF tracks only craft `a`; the full world layout is [b…, a…]. Feeding
    `ekf.state_dict()` (a-only) into a full-world LQR must scatter a's slots
    by name into the full layout and produce a finite, correct command."""
    M, G, dt = 2.0, 9.81, 0.02
    b = Craft("b"); b.add(Mass("body", mass=M))
    a = Craft("a")
    a.add(Mass("body", mass=M))
    for n, f in [("tx", (1, 0, 0)), ("ty", (0, 1, 0)), ("tz", (0, 0, 1))]:
        a.add(Thruster(n, force=f))
    a.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(b, position=(5, 0, 10))
    w.add_craft(a, position=(0, 0, 10))

    ekf = TargetNumpy(EKF(w, track={"a": POSE | TWIST}))   # subset: a only
    lqr = TargetNumpy(LQR(
        w, x_ref={"a": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
        u_ref={"a.tz.throttle": M * G}, regulate=["a.position", "a.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))
    assert lqr.spec.ambient_dim > ekf.spec.ambient_dim     # full spans b+a

    # a sits 1 m below + offset from its hover ref → thrust above weight.
    ekf.reset_from_model_record(
        {"a": a.initial_state(position=(0.5, 0.0, 9.0))},
        P=np.eye(ekf.spec.tangent_dim))
    u = lqr.control(ekf.state_dict())
    assert np.isfinite(list(u.values())).all()
    assert u["a.tz.throttle"] > M * G                      # climbs to recover
