"""EKF → LQR estimate transfer via a flat vector + fixed name-keyed gather.

`ekf.estimate` carries the flat ambient state (EKF spec layout) plus a
`layout` map; `lqr.compute()` scatters those slots — by name, via a
precomputed gather — into its own full ambient layout over the
operating-point base, then applies the gain. This is the codegen-honest
form (a flat buffer + a static index map, not a Python nested dict). The
gather must produce the same control as the legacy dict path.
"""

import numpy as np

from manta import (
    Craft, EKF, LQR, POSE, TWIST, Sim, TargetNumpy, World, wire,
)
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor, Thruster


def _stack(seed_pos):
    M, G, dt = 2.0, 9.81, 0.02
    target = np.array([0.0, 0.0, 10.0])
    c = Craft("c")
    c.add(Mass("body", mass=M))
    for n, f in [("tx", (1, 0, 0)), ("ty", (0, 1, 0)), ("tz", (0, 0, 1))]:
        c.add(Thruster(n, force=f))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10))

    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"c": c.initial_state(position=tuple(seed_pos))},
              P=np.eye(ekf.spec.tangent_dim))
    lqr = TargetNumpy(LQR(
        w, x_ref={"c": {"position": tuple(target), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, regulate=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=dt))
    return ekf, lqr, c


def test_estimate_port_is_flat_vector_with_layout():
    ekf, _, _ = _stack([3, -2, 7])
    est = ekf.estimate
    assert est.dim == ekf.spec.ambient_dim
    assert est.layout is not None
    # layout maps every spec slot to (offset, dim).
    assert est.layout["c.position"] == (0, 3)
    assert est.layout["c.velocity"] == (7, 3)        # after pos(3)+quat(4)
    np.testing.assert_array_equal(est.read(), ekf.x)


def test_gather_matches_dict_path():
    """compute() (flat + gather) must equal control(state_dict) (legacy)."""
    ekf, lqr, _ = _stack([3, -2, 7])
    wire(ekf.estimate, lqr.estimate_in)
    # Move the estimate off the seed so the control is nontrivial.
    ekf.predict(dt=0.02, u={"c.tz.throttle": 19.62})
    ekf.estimate.set(ekf.x)                          # refresh the port

    u_gather = lqr.compute()
    u_dict   = lqr.control(ekf.state_dict())
    for k in u_dict:
        assert abs(u_gather[k] - u_dict[k]) < 1e-9


def test_gather_built_once_and_indexes_by_name():
    ekf, lqr, _ = _stack([1, 1, 9])
    wire(ekf.estimate, lqr.estimate_in)
    lqr.compute()
    g = lqr._gather
    # Identity here (EKF full spec == LQR full spec): each slot maps to
    # the same offset in both layouts.
    for foff, dim, soff in g:
        assert foff == soff
    # position + quaternion + velocity + angular_velocity = 13 entries.
    assert sum(dim for _, dim, _ in g) == ekf.spec.ambient_dim


def test_non_identity_gather_two_crafts():
    """EKF estimates only craft `a`; the full world layout is [b…, a…], so
    a's slots sit at a nonzero full-spec offset while the a-only EKF lays
    them out from 0 — a genuine non-identity name-keyed gather."""
    M, G, dt = 2.0, 9.81, 0.02

    # `b` (no actuation/sensor) is added FIRST so it occupies the front of
    # the full layout; `a` (fully actuated, sensed) follows and is what we
    # estimate + regulate.
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
    wire(ekf.estimate, lqr.estimate_in)
    lqr.compute()

    # EKF layout carries only craft-a slots; full spec spans b+a.
    assert all(n.startswith("a.") for n in ekf.estimate.layout)
    assert lqr.spec.ambient_dim > ekf.spec.ambient_dim
    # The gather is genuinely non-identity: a's full-spec offsets are
    # shifted past craft b, so dst (full) ≠ src (a-only) for every slot.
    assert lqr._gather and all(foff != soff for foff, _, soff in lqr._gather)
    # Each dst offset is exactly craft-a's slot offset in the full layout.
    for name, (soff, dim) in ekf.estimate.layout.items():
        foff = lqr.spec.slot(name).ambient_offset
        assert (foff, dim, soff) in lqr._gather
