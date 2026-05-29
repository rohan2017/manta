"""Manifold ops: SO3 exp/log roundtrip, boxplus/boxminus identities, R3 trivial.

Exercises the unified `Manifold` value-typed API — `SO3Manifold` /
`R3Manifold` instances operating on frame-tagged `Quat` / `Vec3` values.
The same `boxplus_sym` math backs the ESKF tracer and codegen, so these
identities also guard those paths.
"""

import math

import numpy as np
import pytest

from manta import ir
from manta.ir.frames import CraftFrame, WorldFrame
from manta.ir.manifold import R3Manifold, SO3Manifold

SO3 = SO3Manifold()
R3 = R3Manifold(frame=CraftFrame)


def test_so3_identity_log_is_zero():
    with ir.Graph() as g:
        q = ir.Quat[CraftFrame, CraftFrame].input("q")
        q_id = ir.Quat.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        g.output(SO3.boxminus(q, q_id), "delta")

    out = g.compile()(q=[1.0, 0.0, 0.0, 0.0])
    assert np.allclose(out["delta"], [0.0, 0.0, 0.0], atol=1e-9)


def test_so3_exp_log_roundtrip_small():
    with ir.Graph() as g:
        omega = ir.Vec3[CraftFrame].input("omega")
        q_id = ir.Quat.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        rot = SO3.boxplus(q_id, omega)
        # Recover omega via boxminus from identity.
        g.output(SO3.boxminus(rot, q_id), "omega_back")

    fn = g.compile()
    for omega_val in ([0.0, 0.0, 0.0],
                       [0.01, 0.0, 0.0],
                       [0.1, 0.05, -0.07],
                       [0.5, -0.3, 0.4]):
        out = fn(omega=omega_val)
        assert np.allclose(out["omega_back"], omega_val, atol=1e-9), \
            f"roundtrip failed for {omega_val}"


def test_so3_exp_log_roundtrip_large():
    """Larger angle (close to π) still roundtrips."""
    with ir.Graph() as g:
        omega = ir.Vec3[CraftFrame].input("omega")
        q_id = ir.Quat.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        rot = SO3.boxplus(q_id, omega)
        g.output(SO3.boxminus(rot, q_id), "omega_back")

    # 2.5 rad about a unit axis — well below π wraparound but big.
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    omega_val = axis * 2.5

    out = g.compile()(omega=omega_val)
    assert np.allclose(out["omega_back"], omega_val, atol=1e-8)


def test_so3_boxplus_zero_is_identity():
    """`q ⊞ 0 = q` for any unit quaternion. Verified via boxminus(q ⊞ 0, q) = 0."""
    with ir.Graph() as g:
        q = ir.Quat[CraftFrame, CraftFrame].input("q")
        zero = ir.Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        q_new = SO3.boxplus(q, zero)
        g.output(SO3.boxminus(q_new, q), "delta")

    half = math.pi / 5
    q_val = np.array([math.cos(half), 0.3, -0.5, 0.7])
    q_val /= np.linalg.norm(q_val)
    out = g.compile()(q=q_val)
    assert np.allclose(out["delta"], [0.0, 0.0, 0.0], atol=1e-9)


def test_so3_boxplus_boxminus_consistency():
    """`(a ⊞ δ) ⊟ a = δ` for any rotation a and small tangent δ."""
    with ir.Graph() as g:
        q       = ir.Quat[CraftFrame, CraftFrame].input("q")
        omega   = ir.Vec3[CraftFrame].input("omega")
        q_new   = SO3.boxplus(q, omega)
        g.output(SO3.boxminus(q_new, q), "delta_back")

    half = math.pi / 5
    q_val = np.array([math.cos(half), 0.3, -0.5, 0.7])
    q_val /= np.linalg.norm(q_val)

    for omega_val in ([0.0, 0.0, 0.0],
                       [0.01, 0.0, 0.0],
                       [0.1, 0.05, -0.07]):
        out = g.compile()(q=q_val, omega=omega_val)
        assert np.allclose(out["delta_back"], omega_val, atol=1e-8), \
            f"roundtrip failed for omega={omega_val}"


def test_so3_boxplus_then_apply_rotates_vector():
    """exp(z·π/2) applied to (1,0,0) should give (0,1,0)."""
    with ir.Graph() as g:
        omega = ir.Vec3[CraftFrame].input("omega")     # axis·angle tangent
        v     = ir.Vec3[CraftFrame].input("v")
        q_id  = ir.Quat.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        rot   = SO3.boxplus(q_id, omega)                # Quat[Craft, Craft]
        g.output(rot.apply(v), "u")

    out = g.compile()(omega=[0.0, 0.0, math.pi / 2],
                      v=[1.0, 0.0, 0.0])
    assert np.allclose(out["u"], [0.0, 1.0, 0.0], atol=1e-10)


def test_r3_trivial_boxplus_boxminus():
    """R3 boxplus is plain addition; boxminus is subtraction."""
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        delta = R3.boxminus(b, a)
        c = R3.boxplus(a, delta)
        g.output(c,     "c")        # should equal b
        g.output(delta, "delta")    # should equal b - a

    out = g.compile()(a=[1.0, 2.0, 3.0], b=[4.0, 7.0, 11.0])
    assert np.allclose(out["delta"], [3.0, 5.0, 8.0])
    assert np.allclose(out["c"],     [4.0, 7.0, 11.0])


# ---------------------------------------------------------------------------
# Frame validation on the value-typed ops
# ---------------------------------------------------------------------------

def test_r3_boxplus_rejects_mismatched_delta_frame():
    from manta.ir.frames import FrameError
    with ir.Graph():
        v = ir.Vec3[CraftFrame].input("v")
        d = ir.Vec3[WorldFrame].input("d")     # wrong frame
        with pytest.raises(FrameError, match="R3Manifold.boxplus"):
            R3.boxplus(v, d)


def test_r3_boxminus_rejects_mismatched_frame():
    from manta.ir.frames import FrameError
    with ir.Graph():
        v1 = ir.Vec3[CraftFrame].input("v1")
        v2 = ir.Vec3[WorldFrame].input("v2")
        with pytest.raises(FrameError, match="R3Manifold.boxminus"):
            R3.boxminus(v1, v2)


def test_so3_boxplus_rejects_mismatched_delta_frame():
    from manta.ir.frames import FrameError
    with ir.Graph():
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        # delta must live in q's from_frame (WorldFrame); pass CraftFrame.
        d = ir.Vec3[CraftFrame].input("d")
        with pytest.raises(FrameError, match="SO3Manifold.boxplus"):
            SO3.boxplus(q, d)
