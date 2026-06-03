"""Module IR — the generic `NumpyModule` reproduces the per-block façades.

The target architecture collapses every backend's per-block runtime classes
into one generic `lower_module(Module)`. These tests pin that the generic
numpy lowering (`to_module(block)` → `NumpyModule`) matches the existing
hand-written façades (`NumpyWorld`/`NumpyLQR`/`NumpyRecurrence`) numerically,
for all three current evaluator shapes — the parity backstop that lets the
façades be deleted later (Step 1/3 of ARCHITECTURE_TARGET.md).
"""

import numpy as np
import pytest

from manta import World, Craft, Sim, LQR, PID, TargetNumpy
from manta.estimation import EKF
from manta.fields import GravityField
from manta.parts import Mass, Thruster, PositionSensor
from manta.estimation.state_spec import StateSpec
from manta.tick import walk_tick_signature
from manta.codegen.module_build import to_module
from manta.codegen.numpy.module import NumpyModule
from manta.ir.module import Module, StateLayout

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


def _u_defaults(sim, spec):
    sig = walk_tick_signature(sim.tick.casadi_function, sim.world, spec)
    return np.array([sig.input_defaults[n] for n in sig.input_names], float)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_to_module_shapes():
    w, _ = _flyer()
    m = to_module(Sim(w))
    assert isinstance(m, Module)
    assert m.state.names == ("x",)
    assert {p.name for p in m.ports} == {"u", "noise", "dt", "t"}
    assert m.entry("step").writes_state == ("x",)
    assert "F" in m.analysis            # analysis fn, not a runtime method

    lqr_m = to_module(LQR(
        w, x_ref={"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, track=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=0.02))
    assert lqr_m.state == StateLayout(())          # stateless
    assert lqr_m.entry("control").returns == ("u",)


def test_lower_module_factory():
    """`TargetNumpy` backend exposes the generic `lower_module`."""
    from manta.codegen.numpy import _NumpyBackend
    m = to_module(PID(kp=1.0))
    rt = _NumpyBackend().lower_module(m)
    assert isinstance(rt, NumpyModule)
    assert hasattr(rt, "step")


# ---------------------------------------------------------------------------
# Sim parity — generic step == NumpyWorld functional step, over a trajectory
# ---------------------------------------------------------------------------

def test_sim_step_parity():
    w, _ = _flyer()
    sim = Sim(w)
    facade = TargetNumpy(sim)
    spec = StateSpec.from_world(w)
    u = _u_defaults(sim, spec)

    nm = NumpyModule(to_module(sim))
    noise = np.zeros(nm.module.port("noise").size)
    state = facade.initial_state()
    # generic runtime seeds x from the Module's declared init — same source.
    np.testing.assert_allclose(
        nm.state["x"], spec.pack({k: v for k, v in
                                  _flat(state).items() if k in spec}))

    dt, t = 0.01, 0.0
    for _ in range(50):
        state = facade.step(state, dt, t)
        nm.step(u=u, noise=noise, dt=dt, t=t)
        np.testing.assert_allclose(
            nm.state["x"],
            spec.pack({k: v for k, v in _flat(state).items() if k in spec}),
            rtol=1e-9, atol=1e-9)
        t += dt


def test_sim_predict_equals_noiseless_step():
    """The deploy-shaped noiseless `predict` == the driven `step`'s x output
    at noise=0 (cpp runs `predict`; numpy runs `step`)."""
    w, _ = _flyer()
    sim = Sim(w)
    spec = StateSpec.from_world(w)
    u = _u_defaults(sim, spec)
    a = NumpyModule(to_module(sim))
    b = NumpyModule(to_module(sim))
    noise = np.zeros(a.module.port("noise").size)
    dt = 0.01
    for _ in range(20):
        a.step(u=u, noise=noise, dt=dt, t=0.0)     # driven (noise=0)
        b.predict(u=u, dt=dt, t=0.0)               # deploy-shaped
        np.testing.assert_allclose(a.state["x"], b.state["x"], rtol=1e-12,
                                   atol=1e-12)


def _flat(nested):
    out = {}
    for owner, slots in nested.items():
        for s, v in slots.items():
            out[f"{owner}.{s}"] = v
    return out


def test_sim_measure_entry():
    """A measurement entry point reads the sensor's model at the state."""
    c = Craft("c")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps"))
    w = World().add_field(GravityField(g=(0, 0, -G)))
    w.add_craft(c, position=(2, 3, 50))
    sim = Sim(w)
    nm = NumpyModule(to_module(sim))
    out = nm.measure_c_gps_position(u=np.zeros(0))
    np.testing.assert_allclose(out["c.gps.position"], [2, 3, 50], atol=1e-9)


# ---------------------------------------------------------------------------
# Recurrence parity — generic step == NumpyRecurrence step
# ---------------------------------------------------------------------------

def test_recurrence_parity():
    pid = PID(kp=2.0, ki=0.5, kd=0.1, integral_limit=10.0)
    facade = TargetNumpy(pid)
    nm = NumpyModule(to_module(pid))

    dt, t = 0.1, 0.0
    rng = np.random.default_rng(0)
    for _ in range(30):
        sp, meas = float(rng.normal()), float(rng.normal())
        ref = facade.step(dt, setpoint=sp, measurement=meas)
        u = np.array([sp, meas], float)        # input port order
        ret = nm.step(u=u, dt=dt, t=t)
        assert ret["y"][0] == pytest.approx(ref["command"], rel=1e-12, abs=1e-12)
        t += dt


# ---------------------------------------------------------------------------
# LQR parity — generic control == NumpyLQR.u
# ---------------------------------------------------------------------------

def test_lqr_parity():
    w, _ = _flyer()
    lqr = LQR(
        w, x_ref={"c": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
        u_ref={"tz.throttle": M * G}, track=["c.position", "c.velocity"],
        Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=0.02)
    facade = TargetNumpy(lqr)
    nm = NumpyModule(to_module(lqr))

    rng = np.random.default_rng(1)
    for _ in range(20):
        x = facade.spec.pack({s.name: rng.normal(size=s.ambient_dim)
                              for s in facade.spec.slots})
        ref = facade.u(x)
        got = nm.control(x=x)["u"]
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# EKF parity — the filter "is not special": a two-field (x, P) Module whose
# generic predict/update reproduce NumpyEKF bit-for-bit.
# ---------------------------------------------------------------------------

def _gps_world(noise=False):
    c = Craft("c")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.5))
    if noise:
        c.add(Thruster("t", force=(0, 0, 1), force_noise_sigma=1.0))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 100), velocity=(1, 0, 0))
    return w, c


def test_ekf_module_shape():
    w, _ = _gps_world(noise=True)
    m = to_module(EKF(w))
    assert set(m.state.names) == {"x", "P"}
    assert m.state.field("P").kind == "matrix"
    methods = {e.method for e in m.entry_points}
    assert "predict" in methods
    assert "process_noise" in methods            # force_noise ⇒ model Q
    assert "update_c_gps_position" in methods
    assert m.entry("predict").writes_state == ("x", "P")


def test_ekf_predict_update_parity():
    w, c = _gps_world(noise=True)
    gps = next(p for p in c.parts if p.name == "gps")
    ekf_ir = EKF(w)
    facade = TargetNumpy(ekf_ir)
    nm = NumpyModule(to_module(ekf_ir))
    tan = ekf_ir.spec.tangent_dim

    np.testing.assert_allclose(nm.state["x"], facade.x, atol=1e-12)
    np.testing.assert_allclose(nm.state["P"], facade.P, atol=1e-12)

    u = ekf_ir._build_u({"t.throttle": 9.81})
    dt = 0.02
    rng = np.random.default_rng(3)
    t = 0.0
    for k in range(15):
        # predict — model Q via the generic process_noise entry.
        Q = nm.process_noise(u=u, dt=dt, t=t)["Q"]
        assert Q.shape == (tan, tan)
        nm.predict(u=u, dt=dt, t=t, Q=Q)
        facade.predict(dt=dt, t=t, u={"t.throttle": 9.81})
        np.testing.assert_allclose(nm.state["x"], facade.x, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(nm.state["P"], facade.P, rtol=1e-9, atol=1e-9)

        if k % 3 == 0:               # fold a position fix in
            z = facade.x[:3] + rng.normal(scale=0.2, size=3)
            nm.update_c_gps_position(z_c_gps_position=z, u=u, t=0.0)
            facade.update(gps, position=z)
            np.testing.assert_allclose(nm.state["x"], facade.x, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(nm.state["P"], facade.P, rtol=1e-9, atol=1e-9)
        t += dt
