"""Free-via-CasADi: symbolic Jacobian extraction from a graph."""

import math

import numpy as np

from manta import ir
from manta.ir.frames import CraftFrame, WorldFrame


def test_linear_jacobian_is_constant():
    """f(x, v, dt) = x + v * dt. df/dx = identity (a 3x3 I)."""
    with ir.Graph() as g:
        x  = ir.Vec3[WorldFrame].input("x")
        v  = ir.Vec3[WorldFrame].input("v")
        dt = ir.Scalar.input("dt")
        g.output(x + v * dt, "x_next")

    J = g.jacobian(of="x_next", wrt="x")
    out = J(x=[0.0, 0.0, 0.0], v=[1.0, 2.0, 3.0], dt=0.1)
    assert np.allclose(out["jacobian"], np.eye(3))


def test_jacobian_wrt_velocity():
    """Same expression. df/dv = dt * I."""
    with ir.Graph() as g:
        x  = ir.Vec3[WorldFrame].input("x")
        v  = ir.Vec3[WorldFrame].input("v")
        dt = ir.Scalar.input("dt")
        g.output(x + v * dt, "x_next")

    J = g.jacobian(of="x_next", wrt="v")
    out = J(x=[1.0, 2.0, 3.0], v=[0.0, 0.0, 0.0], dt=0.25)
    assert np.allclose(out["jacobian"], 0.25 * np.eye(3))


def test_quat_jacobian_at_identity():
    """For q = identity quaternion, d(q.apply(v))/d(v) should be the
    identity matrix (R(q=identity) = I)."""
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[CraftFrame].input("v")
        g.output(q.apply(v), "v_scene")

    J = g.jacobian(of="v_scene", wrt="v")
    out = J(q=[1.0, 0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0])
    assert np.allclose(out["jacobian"], np.eye(3))


def test_scalar_jacobian():
    """f(a, b) = a² + a·b. df/da = 2a + b."""
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        b = ir.Scalar.input("b")
        g.output(a*a + a*b, "f")

    J = g.jacobian(of="f", wrt="a")
    for (a_val, b_val) in [(0.0, 1.0), (1.5, 2.5), (-1.0, 3.0)]:
        out = J(a=a_val, b=b_val)
        expected = 2 * a_val + b_val
        # Output is a 1x1 matrix here; flatten.
        got = float(np.asarray(out["jacobian"]).flatten()[0])
        assert np.isclose(got, expected), \
            f"da: expected {expected}, got {got} at a={a_val}, b={b_val}"


def test_compiled_function_exposes_casadi_function():
    """User can drop down to the CasADi function directly for codegen etc."""
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        g.output(a * 2.0, "double")

    fn = g.compile()
    cf = fn.casadi_function
    # Spot-check: it has the right name and arity.
    assert cf.name() == "graph"   # default
    assert cf.n_in() == 1
    assert cf.n_out() == 1
