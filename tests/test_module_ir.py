"""The typed Module IR — transforms emit Modules; one runtime lowers them.

Pins the IR contract: every transform's `.module()` carries typed State
fields, Role-tagged Ports, and `StateRef`/`PortRef` kernel args; the one
`NumpyRuntime` reproduces direct kernel evaluation; the Sim's oracle and
deploy Modules agree at noise = 0.
"""

import numpy as np
import pytest

from manta import World, Craft, Sim, LQR, PID, TargetNumpy
from manta.estimation import EKF
from manta.fields import GravityField
from manta.parts import Mass, Thruster, PositionSensor
from manta.ir.module import Hosting, Module, PortRef, Role, StateRef

M, G = 2.0, 9.81


def _flyer():
    c = Craft("c")
    c.add(Mass("body", mass=M))
    c.add(Thruster("tx", force=(1, 0, 0)))
    c.add(Thruster("ty", force=(0, 1, 0)))
    c.add(Thruster("tz", force=(0, 0, 1)))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(0, 0, 10), velocity=(1, 0, 0))
    return w, c


def _gps_world():
    c = Craft("c")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.5))
    c.add(Thruster("t", force=(0, 0, 1), force_noise_sigma=1.0))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 100), velocity=(1, 0, 0))
    return w, c


# ---------------------------------------------------------------------------
# Typed IR shapes
# ---------------------------------------------------------------------------

def test_sim_oracle_module_is_typed():
    w, _ = _flyer()
    m = Sim(w).module()
    assert isinstance(m, Module) and m.hosting is Hosting.THREADED
    assert m.state.field("x").kind == "manifold"
    assert m.port("u").role is Role.CONTROL
    assert m.port("noise").role is Role.NOISE
    assert m.port("dt").role is Role.TIMESTEP
    step = m.entry("step")
    assert all(isinstance(a, (StateRef, PortRef)) for a in step.args)
    assert step.writes == ("x",)


def test_sim_deploy_module_has_measures_and_jacobians():
    w, _ = _gps_world()
    m = Sim(w).deploy_module()
    methods = {e.method for e in m.entry_points}
    assert {"predict", "predict_jacobian", "measure_c_gps_position",
            "measure_c_gps_position_jacobian"} <= methods
    assert not any(p.role is Role.NOISE for p in m.ports)
    # the measurement channel is a typed port shared with any filter
    assert m.port("c.gps.position").role is Role.MEASUREMENT
    # honest kernels: a measure entry takes no dt
    margs = [a.name for a in m.entry("measure_c_gps_position").args
             if isinstance(a, PortRef)]
    assert "dt" not in margs


def test_ekf_module_is_typed():
    w, _ = _gps_world()
    m = EKF(w).module()
    assert m.hosting is Hosting.HELD
    assert set(m.state.names) == {"x", "P"}
    assert m.state.field("P").kind == "matrix"
    methods = {e.method for e in m.entry_points}
    assert {"predict", "predict_with_Q", "update_c_gps_position"} <= methods
    # auto-Q predict takes no Q port — the model noise is baked in
    pargs = [a.name for a in m.entry("predict").args]
    assert "Q" not in pargs
    assert "Q" in [a.name for a in m.entry("predict_with_Q").args]


def test_lqr_module_is_typed():
    w, _ = _flyer()
    lqr = LQR(w, x_ref={"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
              u_ref={"tz.throttle": M * G}, regulate=["c.position", "c.velocity"],
              Q=np.eye(6), R=np.eye(3) * 0.1, dt=0.02)
    m = lqr.module()
    assert m.state.fields == ()                  # stateless
    assert m.port("x").role is Role.STATE and m.port("x").spec is not None
    assert m.entry("control").returns == ("u",)


# ---------------------------------------------------------------------------
# Oracle == deploy at noise = 0; runtime == direct kernel evaluation
# ---------------------------------------------------------------------------

def test_sim_predict_equals_noiseless_step():
    w, _ = _gps_world()
    sim = Sim(w)
    oracle, deploy = sim.module(), sim.deploy_module()
    a, b = TargetNumpy(oracle), TargetNumpy(deploy)
    x = None                       # THREADED: the caller threads the state
    for _ in range(20):
        a.step(0.01)                                   # oracle, zero noise
        vals = {"u": b.build_u(None), "dt": 0.01, "t": 0.0}
        if x is not None:
            vals["x"] = x          # feed the returned write back in
        x = b.call("predict", vals)["x"]   # first call: module's initial x
    np.testing.assert_allclose(
        oracle.spec.pack_any(a.state), x, rtol=1e-12, atol=1e-12)


def test_threaded_call_returns_writes_and_honors_supplied_state():
    """The Hosting contract on the bare numpy engine: a THREADED module's
    `call()` hands its state writes back to the caller, and a state field
    supplied in `values` is used instead of the engine's hidden copy."""
    w, _ = _gps_world()
    deploy = Sim(w).deploy_module()
    rt = TargetNumpy(deploy)
    x0 = np.asarray(deploy.state.field("x").init, dtype=float).reshape(-1)
    vals = {"u": rt.build_u(None), "dt": 0.01, "t": 0.0}
    out = rt.call("predict", vals)
    assert "x" in out and not np.allclose(out["x"], x0)
    # Re-supplying x0 reproduces the first step exactly, even though the
    # engine's internal copy has since advanced.
    out2 = rt.call("predict", {"x": x0, **vals})
    np.testing.assert_array_equal(out2["x"], out["x"])


def test_held_call_keeps_writes_in_place():
    """A HELD module's writes stay in the runtime — `call()` returns only
    the entry's declared returns."""
    w, _ = _gps_world()
    rt = TargetNumpy(EKF(w))
    x0, P0 = rt.x, rt.P
    out = rt.call("predict", {"u": rt.build_u(None), "dt": 0.01, "t": 0.0})
    assert out == {}
    assert not np.allclose(rt.x, x0) and not np.allclose(rt.P, P0)


def test_unknown_value_key_raises():
    """A typo'd key in `values` must never be silently ignored."""
    w, _ = _gps_world()
    rt = TargetNumpy(Sim(w).deploy_module())
    with pytest.raises(TypeError, match="noize"):
        rt.call("predict", {"u": rt.build_u(None), "dt": 0.01,
                            "t": 0.0, "noize": 1.0})


def test_build_u_packs_vector_fields():
    """`build_u` is dim-aware: a dim-3 input lands whole in the flat
    control vector (the old scalar packer silently truncated it)."""
    from manta import Madgwick
    rt = TargetNumpy(Madgwick(beta=0.1))
    g, a = np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.0, 9.81])
    u = rt.build_u({"gyro": g, "accel": a})
    np.testing.assert_array_equal(u, np.concatenate([g, a]))
    with pytest.raises(ValueError, match="dim 3"):
        rt.build_u({"gyro": [1.0, 2.0]})


def test_runtime_matches_direct_kernel():
    """The NumpyRuntime engine is a faithful gather→call→scatter."""
    w, _ = _gps_world()
    ekf = EKF(w)
    m = ekf.module()
    rt = TargetNumpy(ekf)
    x0, P0 = rt.x, rt.P
    u = rt.build_u({"t.throttle": 9.81})
    # direct kernel evaluation
    x1d, P1d = m.functions["predict"](x0, P0, u, 0.02, 0.0)
    rt.predict(0.02, u={"t.throttle": 9.81})
    np.testing.assert_allclose(rt.x, np.asarray(x1d).ravel(), atol=1e-14)
    np.testing.assert_allclose(rt.P, np.asarray(P1d), atol=1e-14)
    # update by name == direct update kernel
    z = rt.x[:3] + np.array([0.1, -0.2, 0.3])
    x2d, P2d = m.functions["update_c_gps_position"](rt.x, rt.P, z, u, 0.0)
    rt.update("gps.position", z, u={"t.throttle": 9.81})
    np.testing.assert_allclose(rt.x, np.asarray(x2d).ravel(), atol=1e-14)
    np.testing.assert_allclose(rt.P, np.asarray(P2d), atol=1e-14)


def test_explicit_Q_override_matches_kernel():
    w, _ = _gps_world()
    ekf = EKF(w)
    rt = TargetNumpy(ekf)
    Q = np.eye(ekf.spec.tangent_dim) * 1e-3
    u = rt.build_u(None)
    x1d, P1d = ekf.module().functions["predict_with_Q"](
        rt.x, rt.P, Q, u, 0.02, 0.0)
    rt.predict(0.02, Q=Q)
    np.testing.assert_allclose(rt.x, np.asarray(x1d).ravel(), atol=1e-14)
    np.testing.assert_allclose(rt.P, np.asarray(P1d), atol=1e-14)


def test_recurrence_module_runtime():
    pid = PID(kp=2.0, ki=0.5, kd=0.1, integral_limit=10.0)
    m = pid.module()
    assert m.hosting is Hosting.HELD
    rt = TargetNumpy(pid)
    rng = np.random.default_rng(0)
    x = np.asarray(m.state.field("x").init, dtype=float).copy()
    for _ in range(30):
        sp, meas = float(rng.normal()), float(rng.normal())
        ref = rt.step(0.1, setpoint=sp, measurement=meas)
        xn, y = m.functions["step"](x, np.array([sp, meas]), 0.1, 0.0)
        x = np.asarray(xn).ravel()
        assert ref["command"] == pytest.approx(float(np.asarray(y).ravel()[0]),
                                               abs=1e-12)


# ---------------------------------------------------------------------------
# Module structural queries
# ---------------------------------------------------------------------------

def test_sole_port_unique_missing_and_duplicate():
    from manta.ir.module import Port, StateLayout
    u = Port("u", Role.CONTROL, shape=(2,))
    za = Port("a.s.za", Role.MEASUREMENT, shape=(3,))
    zb = Port("a.s.zb", Role.MEASUREMENT, shape=(3,))
    x = Port("x", Role.STATE, shape=(4,))
    x_ref = Port("x_ref", Role.STATE, shape=(4,))
    m = Module(name="m", state=StateLayout(), ports=(u, za, zb, x, x_ref),
               functions={}, entry_points=())
    assert m.sole_port(Role.CONTROL) is u
    assert m.sole_port(Role.NOISE) is None
    # per-channel roles refuse to pick one silently
    with pytest.raises(ValueError, match="MEASUREMENT"):
        m.sole_port(Role.MEASUREMENT)
    # STATE may repeat by contract: primary first, reference after (LQR)
    assert m.sole_port(Role.STATE) is x


def test_initial_x_and_spec_fall_back_past_none_manifold_field():
    """A manifold State field with init=None must not short-circuit the
    STATE-port fallback (and the same for `spec`)."""
    from manta.ir.module import Port, StateField, StateLayout
    init = np.arange(4.0)
    f = StateField("x", "manifold", (4,), init=None, spec=None)
    p = Port("x", Role.STATE, shape=(4,), spec="the-spec", init=init)
    m = Module(name="m", state=StateLayout((f,)), ports=(p,),
               functions={}, entry_points=())
    assert m.initial_x is init
    assert m.spec == "the-spec"
