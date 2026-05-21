"""Manifold helpers: SO3 exp/log roundtrip, boxplus/boxminus identities, R3 trivial."""

import math

import numpy as np
import pytest

from manta_next import ir
from manta_next.ir.frames import CraftFrame, AnchorFrame
from manta_next.math.manifold import R3, SO3


def test_so3_identity_log_is_zero():
    with ir.Graph() as g:
        q = ir.Quat[CraftFrame, CraftFrame].input("q")
        rot = SO3.from_quat(q)
        rot_id = SO3.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        g.output(rot.boxminus(rot_id), "delta")

    out = g.compile()(q=[1.0, 0.0, 0.0, 0.0])
    assert np.allclose(out["delta"], [0.0, 0.0, 0.0], atol=1e-9)


def test_so3_exp_log_roundtrip_small():
    with ir.Graph() as g:
        omega = ir.Vec3[CraftFrame].input("omega")
        rot = SO3.identity(from_frame=CraftFrame, to_frame=CraftFrame).boxplus(omega)
        # Recover omega via boxminus from identity.
        rot_id = SO3.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        g.output(rot.boxminus(rot_id), "omega_back")

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
        rot = SO3.identity(from_frame=CraftFrame, to_frame=CraftFrame).boxplus(omega)
        rot_id = SO3.identity(from_frame=CraftFrame, to_frame=CraftFrame)
        g.output(rot.boxminus(rot_id), "omega_back")

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
        rot_q  = SO3.from_quat(q)
        rot_new = rot_q.boxplus(zero)
        g.output(rot_new.boxminus(rot_q), "delta")

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
        rot_a   = SO3.from_quat(q)
        rot_new = rot_a.boxplus(omega)
        delta_back = rot_new.boxminus(rot_a)
        g.output(delta_back, "delta_back")

    half = math.pi / 5
    q_val = np.array([math.cos(half), 0.3, -0.5, 0.7])
    q_val /= np.linalg.norm(q_val)

    for omega_val in ([0.0, 0.0, 0.0],
                       [0.01, 0.0, 0.0],
                       [0.1, 0.05, -0.07]):
        out = g.compile()(q=q_val, omega=omega_val)
        assert np.allclose(out["delta_back"], omega_val, atol=1e-8), \
            f"roundtrip failed for omega={omega_val}"


def test_so3_from_axis_angle():
    """from_axis_angle(z, π/2) applied to (1,0,0) should give (0,1,0)."""
    with ir.Graph() as g:
        axis = ir.Vec3[CraftFrame].input("axis")
        angle = ir.Scalar.input("angle")
        v    = ir.Vec3[CraftFrame].input("v")
        rot  = SO3.from_axis_angle(axis, angle)
        # rot is Quat[CraftFrame, CraftFrame]; apply to v.
        g.output(rot.quat.apply(v), "u")

    out = g.compile()(axis=[0.0, 0.0, 1.0],
                      angle=math.pi / 2,
                      v=[1.0, 0.0, 0.0])
    assert np.allclose(out["u"], [0.0, 1.0, 0.0], atol=1e-10)


def test_r3_trivial_boxplus_boxminus():
    """R3 boxplus is plain addition; boxminus is subtraction."""
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        ra = R3(a)
        rb = R3(b)
        delta = rb.boxminus(ra)
        rc = ra.boxplus(delta)
        g.output(rc.value, "rc")        # should equal b
        g.output(delta,     "delta")    # should equal b - a

    out = g.compile()(a=[1.0, 2.0, 3.0], b=[4.0, 7.0, 11.0])
    assert np.allclose(out["delta"], [3.0, 5.0, 8.0])
    assert np.allclose(out["rc"],    [4.0, 7.0, 11.0])


# ---------------------------------------------------------------------------
# Frame validation on R3 / RigidBody boxplus/boxminus
# ---------------------------------------------------------------------------

def test_r3_boxplus_rejects_mismatched_delta_frame():
    from manta_next.math.manifold import R3
    from manta_next.ir.frames import FrameError
    with ir.Graph() as g:
        v = ir.Vec3[CraftFrame].input("v")
        d = ir.Vec3[AnchorFrame].input("d")     # wrong frame
        r = R3(v)
        with pytest.raises(FrameError, match="R3.boxplus"):
            r.boxplus(d)


def test_r3_boxminus_rejects_mismatched_other_frame():
    from manta_next.math.manifold import R3
    from manta_next.ir.frames import FrameError
    with ir.Graph() as g:
        v1 = ir.Vec3[CraftFrame].input("v1")
        v2 = ir.Vec3[AnchorFrame].input("v2")
        r1 = R3(v1); r2 = R3(v2)
        with pytest.raises(FrameError, match="R3.boxminus"):
            r1.boxminus(r2)


def test_rigid_body_boxplus_rejects_mismatched_position_delta():
    from manta_next.math.manifold import R3, SO3, RigidBody
    from manta_next.ir.frames import FrameError
    with ir.Graph() as g:
        p   = ir.Vec3[AnchorFrame].input("p")
        ori = ir.Quat[AnchorFrame, CraftFrame].input("ori")
        vL  = ir.Vec3[AnchorFrame].input("vL")
        vA  = ir.Vec3[CraftFrame].input("vA")
        rb  = RigidBody(position=p, orientation=SO3(ori),
                        vel_linear=vL, vel_angular=vA)
        # δp must be in AnchorFrame; pass CraftFrame to trigger the check.
        bad_dp = ir.Vec3[CraftFrame].input("bad_dp")
        dθ     = ir.Vec3[AnchorFrame].input("dθ")
        dvL    = ir.Vec3[AnchorFrame].input("dvL")
        dvA    = ir.Vec3[CraftFrame].input("dvA")
        with pytest.raises(FrameError, match="d_position"):
            rb.boxplus(bad_dp, dθ, dvL, dvA)
