"""TargetJax — SX-instruction → JAX lowering parity + autodiff."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
# The recommended usage: opt into float64 explicitly at startup. (Without
# this, TargetJax enables it at first use with a RuntimeWarning — manta
# kernels are float64 physics.)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                    # noqa: E402

import casadi as ca                                        # noqa: E402

from manta import Craft, Sim, TargetJax, TargetNumpy, World   # noqa: E402
from manta.codegen.jax import TargetJax as JaxModule       # noqa: E402
from manta.codegen.jax._translate import translate         # noqa: E402
from manta.fields import GravityField                      # noqa: E402
from manta.parts import Mass                               # noqa: E402
from manta.parts.articulation.joint import RevoluteJoint   # noqa: E402

from .test_fit import DT, _drone


def test_translate_simple_kernel():
    x = ca.MX.sym("x", 3)
    y = ca.MX.sym("y", 3)
    f = ca.Function("f", [x, y], [x + 2 * y, ca.dot(x, y)])
    jf = translate(f)
    xa, ya = np.array([1.0, 2, 3]), np.array([4.0, 5, 6])
    ref = f(xa, ya)
    out = jf(xa, ya)
    assert np.allclose(np.array(out[0]).ravel(), np.array(ref[0]).ravel())
    assert np.allclose(np.array(out[1]).ravel(), np.array(ref[1]).ravel())


def test_translate_matrix_input_column_major():
    A = ca.MX.sym("A", 2, 3)
    g = ca.Function("g", [A], [A @ ca.DM.ones(3, 1), A.T])
    jg = translate(g)
    Am = np.arange(6, dtype=float).reshape(2, 3)
    assert np.allclose(np.array(jg(Am)[0]).ravel(),
                       np.array(g(Am)[0]).ravel())
    assert np.allclose(np.array(jg(Am)[1]), np.array(g(Am)[1]))


_CACHE = {}


def _lowered():
    """One shared promoted Sim + JAX lowering for all parity tests."""
    if "sim" not in _CACHE:
        sim = Sim(_drone(), parameters=["t1.force_quad", "body.mass"])
        _CACHE["sim"] = (sim.module(), TargetJax(sim))
    return _CACHE["sim"]


def test_step_kernel_parity():
    mod, jm = _lowered()
    rng = np.random.default_rng(0)
    x = np.asarray(jm.initial_state())
    u = rng.uniform(0.2, 0.8, 4)
    noise = rng.standard_normal(mod.port("noise").size) * 0.01
    p = np.asarray(jm.param_defaults())
    ref = mod.functions["step"](x, u, noise, p, DT, 0.0)
    out = jm.kernel("step")(x, u, noise, p, DT, 0.0)
    for a, b in zip(ref, out):
        assert np.allclose(np.array(a).ravel(), np.array(b).ravel(),
                           atol=1e-12)


def test_grad_matches_casadi_jacobian():
    mod, jm = _lowered()
    rng = np.random.default_rng(1)
    x = np.asarray(jm.initial_state())
    u = rng.uniform(0.2, 0.8, 4)
    noise = np.zeros(mod.port("noise").size)
    p = np.asarray(jm.param_defaults())
    step = mod.functions["step"]

    xs = ca.MX.sym("x", x.size)
    us = ca.MX.sym("u", u.size)
    ns = ca.MX.sym("n", noise.size)
    ps = ca.MX.sym("p", p.size)
    expr = step(xs, us, ns, ps, DT, 0.0)[0]
    Jca = ca.Function("J", [xs, us, ns, ps],
                      [ca.jacobian(ca.sum1(expr), ps)])
    g_ref = np.array(Jca(x, u, noise, p)).ravel()

    step_jx = jm.kernel("step")
    g = jax.grad(lambda pp: jnp.sum(
        step_jx(x, u, noise, pp, DT, 0.0)[0]))(jnp.asarray(p))
    assert np.allclose(np.array(g), g_ref, atol=1e-12)


def test_scan_rollout_matches_numpy_sim():
    mod, jm = _lowered()
    rng = np.random.default_rng(2)
    u = rng.uniform(0.2, 0.8, 4)
    K = 30
    rollout = jm.make_rollout()
    Xs, readings = rollout(
        jm.initial_state(),
        jnp.tile(jnp.asarray(u), (K, 1)),
        jnp.zeros((K, mod.port("noise").size)),
        jm.param_defaults(), DT, 0.0)

    ref = TargetNumpy(Sim(_drone()))
    for nm, val in zip(["t1", "t2", "t3", "t4"], u):
        ref.state["drone"][f"{nm}.throttle"] = val
    for _ in range(K):
        ref.step(DT)
    from manta.ir.state_spec import flatten_nested
    x_ref = mod.spec.pack_any(flatten_nested(ref.state))
    assert np.allclose(np.array(Xs[-1]), x_ref, atol=1e-12)
    assert readings["drone.imu.gyro"].shape == (K, 3)
    # Last readings row matches the numpy sim's final outputs.
    assert np.allclose(np.array(readings["drone.imu.gyro"][-1]),
                       ref.outputs()["drone"]["imu.gyro"], atol=1e-12)


def test_grad_through_rollout_is_finite():
    mod, jm = _lowered()
    K = 20
    rollout = jm.make_rollout()
    x0 = jm.initial_state()
    U = jnp.full((K, 4), 0.5)
    NO = jnp.zeros((K, mod.port("noise").size))

    def loss(p):
        _, r = rollout(x0, U, NO, p, DT, 0.0)
        return jnp.sum(r["drone.imu.accel"] ** 2)

    g = jax.grad(loss)(jm.param_defaults())
    assert g.shape == jm.param_defaults().shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_deploy_module_lowering():
    sim = Sim(_drone())
    jm = TargetJax(sim.deploy_module())
    x = np.asarray(jm.initial_state())
    u = np.full(4, 0.5)
    writes, _ = jm.call("predict", x=x, u=u, dt=DT)
    ref = sim.deploy_module().functions["predict"](x, u, DT, 0.0)
    assert np.allclose(np.array(writes["x"]).ravel(),
                       np.array(ref).ravel(), atol=1e-12)


def _spinner_world():
    """A jointed craft — its tick carries runtime-pivoting Linsol solve
    nodes that cannot SX-expand whole."""
    c = Craft("spinner")
    c.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.01)))
    j = c.add(RevoluteJoint("wheel", mode="passive"))
    j.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, 0.01),
               transform=(0.05, 0.0, 0.0)))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c, position=(0, 0, 10), **{"wheel.rate": 3.0})
    return w


def test_jointed_craft_step_parity():
    """A jointed craft's Linsol-bearing step kernel lowers via the
    solve-node cut (`jnp.linalg.solve`) and matches CasADi bit-tight
    over a multi-step trajectory."""
    w = _spinner_world()
    mod = Sim(w).module()
    jm = JaxModule(mod)
    step_jx = jm.kernel("step")
    step_ca = mod.functions["step"]

    x_j = np.asarray(jm.initial_state())
    x_c = x_j.copy()
    u = np.zeros(mod.port("u").size)
    noise = np.zeros(mod.port("noise").size)
    dt = 0.002
    for k in range(50):
        t = k * dt
        x_j = np.array(step_jx(x_j, u, noise, dt, t)[0]).ravel()
        x_c = np.array(step_ca(x_c, u, noise, dt, t)).ravel()
    assert np.allclose(x_j, x_c, atol=1e-9)
    # The joint actually moved (the solve did real work).
    from manta.ir.state_spec import StateSpec
    spec = StateSpec.from_world(w)
    slot = spec.slot("spinner.wheel.angle")
    assert abs(x_j[slot.ambient_offset]) > 1e-3


def test_jointed_craft_grad_is_finite():
    """jax.grad flows through the jnp.linalg.solve cut."""
    jax_ = pytest.importorskip("jax")
    w = _spinner_world()
    mod = Sim(w).module()
    jm = JaxModule(mod)
    step_jx = jm.kernel("step")
    u = np.zeros(mod.port("u").size)
    noise = np.zeros(mod.port("noise").size)

    def loss(x0):
        out = step_jx(x0, u, noise, 0.002, 0.0)[0]
        return jnp.sum(out ** 2)

    g = jax_.grad(loss)(jnp.asarray(jm.initial_state()))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_call_unknown_value_key_raises():
    mod, jm = _lowered()
    x = np.asarray(jm.initial_state())
    with pytest.raises(TypeError, match="unknown value key"):
        jm.call("step", x=x, u=np.zeros(4),
                noise=np.zeros(mod.port("noise").size), dt=DT, tt=0.0)


def test_initial_state_on_stateless_module_raises():
    from manta.ir.module import Module, StateLayout
    m = Module(name="empty", state=StateLayout(), ports=(),
               functions={}, entry_points=())
    with pytest.raises(ValueError, match="manifold state"):
        JaxModule(m).initial_state()
