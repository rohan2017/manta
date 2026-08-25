"""Numerical correctness of the IR types: arithmetic and operator semantics."""

import math

import numpy as np

from manta import ir
from manta.ir.frames import CraftFrame, PartFrame, WorldFrame

# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------

def test_scalar_arithmetic():
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        b = ir.Scalar.input("b")
        g.output(a + b, "add")
        g.output(a - b, "sub")
        g.output(a * b, "mul")
        g.output(a / b, "div")
        g.output(-a,    "neg")
        g.output(a ** 2,"sq")

    out = g.compile()(a=3.0, b=4.0)
    assert np.isclose(out["add"], 7.0)
    assert np.isclose(out["sub"], -1.0)
    assert np.isclose(out["mul"], 12.0)
    assert np.isclose(out["div"], 0.75)
    assert np.isclose(out["neg"], -3.0)
    assert np.isclose(out["sq"],  9.0)


def test_scalar_python_literals_promote():
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        g.output(a * 0.5,  "half")
        g.output(2.0 - a,  "rev")
    out = g.compile()(a=4.0)
    assert np.isclose(out["half"], 2.0)
    assert np.isclose(out["rev"],  -2.0)


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------

def test_vec3_basic_arithmetic():
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        g.output(a + b,           "sum")
        g.output(a - b,           "diff")
        g.output(-a,              "neg")
        g.output(a * 2.0,         "scaled")
        g.output(a.dot(b),        "dot")
        g.output(a.cross(b),      "cross")
        g.output(a.norm(),        "norm_a")

    out = g.compile()(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0])
    assert np.allclose(out["sum"],    [5.0, 7.0, 9.0])
    assert np.allclose(out["diff"],   [-3.0, -3.0, -3.0])
    assert np.allclose(out["neg"],    [-1.0, -2.0, -3.0])
    assert np.allclose(out["scaled"], [2.0, 4.0, 6.0])
    assert np.isclose(out["dot"],     32.0)        # 1·4 + 2·5 + 3·6
    assert np.allclose(out["cross"],  [-3.0, 6.0, -3.0])  # (2·6−3·5, 3·4−1·6, 1·5−2·4)
    assert np.isclose(out["norm_a"],  math.sqrt(14.0))


def test_vec3_normalize():
    with ir.Graph() as g:
        v = ir.Vec3[CraftFrame].input("v")
        g.output(v.normalize(), "n")
    out = g.compile()(v=[3.0, 4.0, 0.0])
    assert np.allclose(out["n"], [0.6, 0.8, 0.0])


def test_vec3_constants():
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        c = ir.Vec3[CraftFrame].constant((10.0, 20.0, 30.0))
        g.output(a + c, "out")
    out = g.compile()(a=[1.0, 2.0, 3.0])
    assert np.allclose(out["out"], [11.0, 22.0, 33.0])


def test_vec3_components():
    with ir.Graph() as g:
        v = ir.Vec3[CraftFrame].input("v")
        g.output(v.x, "x")
        g.output(v.y, "y")
        g.output(v.z, "z")
    out = g.compile()(v=[10.0, 20.0, 30.0])
    assert np.isclose(out["x"], 10.0)
    assert np.isclose(out["y"], 20.0)
    assert np.isclose(out["z"], 30.0)


# ---------------------------------------------------------------------------
# Mat3
# ---------------------------------------------------------------------------

def test_mat3_apply_vec3():
    with ir.Graph() as g:
        R = ir.Mat3[WorldFrame, CraftFrame].input("R")
        v = ir.Vec3[CraftFrame].input("v")
        g.output(R @ v, "u")

    R_val = np.array([[0.0, -1.0, 0.0],
                      [1.0,  0.0, 0.0],
                      [0.0,  0.0, 1.0]])   # 90° about +z
    out = g.compile()(R=R_val, v=[1.0, 0.0, 0.0])
    assert np.allclose(out["u"], [0.0, 1.0, 0.0])


def test_mat3_matmul():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        B = ir.Mat3[CraftFrame, PartFrame].input("B")
        g.output(A @ B, "C")
    R_z = np.array([[0.0, -1.0, 0.0],
                    [1.0,  0.0, 0.0],
                    [0.0,  0.0, 1.0]])
    R_x = np.array([[1.0, 0.0,  0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0,  0.0]])
    out = g.compile()(A=R_z, B=R_x)
    assert np.allclose(out["C"], R_z @ R_x)


def test_mat3_transpose_and_inverse():
    with ir.Graph() as g:
        A = ir.Mat3[CraftFrame, CraftFrame].input("A")
        g.output(A.transpose(), "AT")
        g.output(A.inv(),       "AI")
    M = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]])
    out = g.compile()(A=M)
    assert np.allclose(out["AT"], M.T)
    assert np.allclose(out["AI"], np.linalg.inv(M))


# ---------------------------------------------------------------------------
# Quat
# ---------------------------------------------------------------------------

def test_quat_identity_apply():
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[CraftFrame].input("v")
        g.output(q.apply(v), "u")
    out = g.compile()(q=[1.0, 0.0, 0.0, 0.0], v=[1.0, 2.0, 3.0])
    assert np.allclose(out["u"], [1.0, 2.0, 3.0])


def test_quat_apply_z_axis():
    """90° rotation about +z: (1,0,0) → (0,1,0)."""
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[CraftFrame].input("v")
        g.output(q.apply(v), "u")

    half = math.pi / 4   # half of 90°
    q90z = [math.cos(half), 0.0, 0.0, math.sin(half)]
    out = g.compile()(q=q90z, v=[1.0, 0.0, 0.0])
    assert np.allclose(out["u"], [0.0, 1.0, 0.0], atol=1e-10)


def test_quat_compose():
    """Composing two 90° rotations about +z = 180° about +z."""
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        # Apply (q * q) to v.
        g.output((q * ir.Quat[CraftFrame, CraftFrame].constant(
                    [math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)])).apply(
                    ir.Vec3[CraftFrame].input("v")), "u")

    half = math.pi / 4
    q90 = [math.cos(half), 0.0, 0.0, math.sin(half)]
    # q · q90 = 180° rotation. (1,0,0) → (-1,0,0).
    out = g.compile()(q=q90, v=[1.0, 0.0, 0.0])
    assert np.allclose(out["u"], [-1.0, 0.0, 0.0], atol=1e-10)


def test_quat_conjugate_inverts_rotation():
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[CraftFrame].input("v")
        rotated   = q.apply(v)
        unrotated = q.conjugate().apply(rotated)
        g.output(unrotated, "u")

    half = math.pi / 5
    q_arbitrary = [math.cos(half), 0.3, -0.5, 0.7]
    # Normalize so q is unit:
    n = math.sqrt(sum(x*x for x in q_arbitrary))
    q_arbitrary = [x / n for x in q_arbitrary]

    out = g.compile()(q=q_arbitrary, v=[1.0, 2.0, 3.0])
    assert np.allclose(out["u"], [1.0, 2.0, 3.0], atol=1e-10)


def test_quat_to_rotmat_matches_apply():
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[CraftFrame].input("v")
        R = q.to_rotmat()
        g.output(q.apply(v), "via_apply")
        g.output(R @ v,      "via_rotmat")

    half = math.pi / 3
    q_val = [math.cos(half), 0.1, 0.2, math.sin(half)]
    n = math.sqrt(sum(x*x for x in q_val))
    q_val = [x / n for x in q_val]
    out = g.compile()(q=q_val, v=[1.0, 0.5, -0.3])
    assert np.allclose(out["via_apply"], out["via_rotmat"], atol=1e-10)
