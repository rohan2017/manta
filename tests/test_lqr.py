"""LQR — control-Jacobian B, gain synthesis, closed-loop regulation.

The canonical controllable manta system here is a 3-axis-thruster
free-flyer regulating position + velocity (the rigid-body attitude is
uncontrolled, so it's frozen via `track=`). Underactuated full-state LQR
is expected to fail (not stabilizable) — that's tested too.
"""

import numpy as np
import pytest
import casadi as ca

from manta import World, Craft, Sim, LQR, TargetNumpy
from manta.fields import GravityField
from manta.parts import Mass, Thruster
from manta.ir.state_spec import StateSpec
from manta.linearized_system import LinearizedSystem

M, G = 2.0, 9.81


def _flyer():
    """3-axis-thrust free-flyer: position + velocity controllable."""
    c = Craft("c")
    c.add(Mass("body", mass=M))
    c.add(Thruster("tx", force=(1, 0, 0)))
    c.add(Thruster("ty", force=(0, 1, 0)))
    c.add(Thruster("tz", force=(0, 0, 1)))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10))
    return w, c


def _lqr(**kw):
    w, _ = _flyer()
    defaults = dict(
        x_ref={"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G},
        track=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=0.02)
    defaults.update(kw)
    return LQR(w, **defaults), w


# ---------------------------------------------------------------------------
# Control Jacobian B = ∂f/∂u (the seam extension LQR relies on)
# ---------------------------------------------------------------------------

def test_control_jacobian_matches_finite_difference():
    w, c = _flyer()
    lin = LinearizedSystem(w, close_track=False, control=True)
    spec = lin.spec
    x0 = spec.pack({f"c.{k}": v for k, v in
                    c.initial_state(position=(0, 0, 10)).items()
                    if f"c.{k}" in spec})
    u0 = np.zeros(len(lin.input_names))
    u0[lin.input_names.index("c.tz.throttle")] = M * G
    dt = 0.02
    B = np.array(lin.B_fn(x0, u0, dt, 0.0))
    assert B.shape == (spec.tangent_dim, len(lin.input_names))

    xa = ca.MX.sym("a", spec.ambient_dim); xb = ca.MX.sym("b", spec.ambient_dim)
    bm = ca.Function("bm", [xa, xb], [spec.boxminus_sym(xa, xb)])
    base = np.array(lin.predict_fn(x0, u0, dt, 0.0)).ravel()
    eps = 1e-6
    for j in range(len(u0)):
        up = u0.copy(); up[j] += eps
        pert = np.array(lin.predict_fn(x0, up, dt, 0.0)).ravel()
        fd = np.array(bm(pert, base)).ravel() / eps
        assert np.allclose(B[:, j], fd, atol=1e-5), f"B col {j} mismatch"


def test_control_jacobian_off_by_default():
    """The EKF doesn't pay for B — it's only built with control=True."""
    w, _ = _flyer()
    lin = LinearizedSystem(w, close_track=False)
    assert lin.B_sym is None and lin.B_fn is None


# ---------------------------------------------------------------------------
# Gain synthesis + stability
# ---------------------------------------------------------------------------

def test_lqr_gain_is_stabilizing():
    lqr, _ = _lqr()
    assert lqr.K.shape == (3, 6)            # n_u × tracked tangent
    assert np.max(np.abs(lqr.closed_loop_eigs)) < 1.0


def test_higher_state_cost_gives_tighter_gain():
    soft, _ = _lqr(Q=np.eye(6) * 1.0)
    hard, _ = _lqr(Q=np.eye(6) * 100.0)
    assert np.linalg.norm(hard.K) > np.linalg.norm(soft.K)


# ---------------------------------------------------------------------------
# Closed-loop regulation
# ---------------------------------------------------------------------------

def test_closed_loop_regulates_to_setpoint():
    lqr, w = _lqr()
    ctrl = TargetNumpy(lqr)
    sim = TargetNumpy(Sim(w))
    sim.state["c"]["position"] = np.array([2.0, -1.0, 8.0])
    for _ in range(600):
        for full, v in ctrl.control(sim.state).items():
            owner, rest = full.split(".", 1)      # "c.tx.throttle" → c / tx.throttle
            sim.state[owner][rest] = v
        sim.step(0.02)
    np.testing.assert_allclose(
        np.asarray(sim.state["c"]["position"]).ravel(), [0, 0, 10], atol=1e-2)
    np.testing.assert_allclose(
        np.asarray(sim.state["c"]["velocity"]).ravel(), [0, 0, 0], atol=1e-2)


def test_control_maps_to_named_inputs():
    lqr, _ = _lqr()
    ctrl = TargetNumpy(lqr)
    u = ctrl.control({"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}})
    assert set(u) == set(lqr.input_names)
    # At the setpoint the command is exactly the trim (tz = m·g, others 0).
    assert u["c.tz.throttle"] == pytest.approx(M * G, abs=1e-6)
    assert u["c.tx.throttle"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Controllability + validation
# ---------------------------------------------------------------------------

def test_underactuated_full_state_raises():
    """Full-state LQR on a thrust-only craft is not stabilizable (attitude
    integrators are uncontrolled) — the Riccati iteration must reject it."""
    w, _ = _flyer()
    with pytest.raises(RuntimeError, match="stabilizable"):
        LQR(w, x_ref={"c": {"position": (0, 0, 10)}},
            u_ref={"tz.throttle": M * G}, dt=0.02)   # track=None → full


def test_unknown_track_slot_raises():
    w, _ = _flyer()
    with pytest.raises(KeyError, match="unknown slot"):
        LQR(w, x_ref={"c": {"position": (0, 0, 10)}},
            track=["c.bogus"], dt=0.02)


def test_bad_Q_shape_raises():
    w, _ = _flyer()
    with pytest.raises(ValueError, match="Q must be"):
        LQR(w, x_ref={"c": {"position": (0, 0, 10)}},
            track=["c.position", "c.velocity"], Q=np.eye(3), dt=0.02)


def test_no_inputs_raises():
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    w = World().add_field(GravityField(g=(0, 0, 0))); w.add_craft(c)
    with pytest.raises(ValueError, match="no Part Inputs"):
        LQR(w, x_ref={"c": {"position": (0, 0, 0)}}, dt=0.02)
