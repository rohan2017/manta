"""Frame mismatch errors fire at op-construction time with source location."""

import re

import pytest

from manta import ir
from manta.ir.frames import (
    WorldFrame,
    CraftFrame,
    Frame,
    FrameError,
    PartFrame,
    PlanetFrame,
)


class CustomFrame(Frame):
    pass


# ---------------------------------------------------------------------------
# Vec3 frame checks
# ---------------------------------------------------------------------------

def test_vec3_add_same_frame_ok():
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        c = a + b   # OK
        g.output(c, "out")


def test_vec3_add_different_frames_raises():
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[PartFrame].input("b")
        with pytest.raises(FrameError) as exc:
            _ = a + b
        msg = str(exc.value)
        assert "CraftFrame" in msg
        assert "PartFrame" in msg


def test_vec3_cross_different_frames_raises():
    with ir.Graph() as g:
        a = ir.Vec3[WorldFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        with pytest.raises(FrameError):
            _ = a.cross(b)


def test_vec3_dot_different_frames_raises():
    with ir.Graph() as g:
        a = ir.Vec3[WorldFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        with pytest.raises(FrameError):
            _ = a.dot(b)


def test_frame_error_includes_source_location():
    """The FrameError message should include the user .py:line, not manta internals."""
    with ir.Graph() as g:
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[PartFrame].input("b")
        try:
            _ = a + b
            assert False, "expected FrameError"
        except FrameError as e:
            # Source should point at this test file.
            assert e.source is not None
            assert "test_frame_checking.py" in e.source
            # And NOT at internal ir code.
            assert "ir/types.py" not in e.source


# ---------------------------------------------------------------------------
# Mat3 / Quat composition checks
# ---------------------------------------------------------------------------

def test_mat3_matmul_compatible_frames_ok():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        B = ir.Mat3[CraftFrame, PartFrame].input("B")
        C = A @ B
        g.output(C, "out")
        # Result frames: from=WorldFrame, to=PartFrame
        assert C.from_frame is WorldFrame   # WorldFrame == WorldFrame
        assert C.to_frame is PartFrame


def test_mat3_matmul_incompatible_frames_raises():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        B = ir.Mat3[PartFrame,  PlanetFrame].input("B")  # middle mismatch
        with pytest.raises(FrameError):
            _ = A @ B


def test_mat3_apply_vec3_correct_frame_ok():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        v = ir.Vec3[CraftFrame].input("v")
        u = A @ v
        g.output(u, "out")
        assert u.frame is WorldFrame


def test_mat3_apply_vec3_wrong_frame_raises():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        v = ir.Vec3[PartFrame].input("v")
        with pytest.raises(FrameError):
            _ = A @ v


def test_quat_compose_ok():
    with ir.Graph() as g:
        q1 = ir.Quat[WorldFrame, CraftFrame].input("q1")
        q2 = ir.Quat[CraftFrame, PartFrame].input("q2")
        q3 = q1 * q2
        g.output(q3, "out")
        assert q3.from_frame is WorldFrame
        assert q3.to_frame is PartFrame


def test_quat_compose_mismatch_raises():
    with ir.Graph() as g:
        q1 = ir.Quat[WorldFrame, CraftFrame].input("q1")
        q2 = ir.Quat[PartFrame,  PlanetFrame].input("q2")
        with pytest.raises(FrameError):
            _ = q1 * q2


def test_quat_apply_correct_frame_ok():
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[CraftFrame].input("v")
        u = q.apply(v)
        g.output(u, "out")
        assert u.frame is WorldFrame


def test_quat_apply_wrong_frame_raises():
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        v = ir.Vec3[PartFrame].input("v")
        with pytest.raises(FrameError):
            _ = q.apply(v)


def test_quat_conjugate_swaps_frames():
    with ir.Graph() as g:
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        qc = q.conjugate()
        g.output(qc, "out")
        assert qc.from_frame is CraftFrame
        assert qc.to_frame is WorldFrame


def test_mat3_transpose_swaps_frames():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        At = A.transpose()
        g.output(At, "out")
        assert At.from_frame is CraftFrame
        assert At.to_frame is WorldFrame


def test_mat3_inv_requires_endomorphism():
    with ir.Graph() as g:
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")  # not endomorphism
        with pytest.raises(FrameError):
            _ = A.inv()


def test_user_defined_frame_works():
    with ir.Graph() as g:
        a = ir.Vec3[CustomFrame].input("a")
        b = ir.Vec3[CustomFrame].input("b")
        c = a + b   # OK
        g.output(c, "out")


def test_non_frame_arg_raises_type_error():
    class NotAFrame:
        pass
    with pytest.raises(TypeError, match="Frame"):
        ir.Vec3[NotAFrame]    # type: ignore[type-arg]


def test_frames_cannot_be_instantiated():
    with pytest.raises(TypeError, match="frame tag"):
        WorldFrame()


# ---------------------------------------------------------------------------
# Scalar-only operator guards — an elementwise Vec3*Vec3 would bypass every
# frame check, so the scaling operators reject non-Scalar IR values.
# ---------------------------------------------------------------------------

def test_vec3_times_vec3_raises():
    with ir.Graph():
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        with pytest.raises(TypeError, match=r"\.dot\(\)/\.cross\(\)"):
            _ = a * b


def test_vec3_div_vec3_raises():
    with ir.Graph():
        a = ir.Vec3[CraftFrame].input("a")
        b = ir.Vec3[CraftFrame].input("b")
        with pytest.raises(TypeError, match="scalar"):
            _ = a / b


def test_mat3_times_mat3_raises():
    with ir.Graph():
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        B = ir.Mat3[WorldFrame, CraftFrame].input("B")
        with pytest.raises(TypeError, match="@"):
            _ = A * B


def test_mat3_times_vec3_raises():
    with ir.Graph():
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        v = ir.Vec3[CraftFrame].input("v")
        with pytest.raises(TypeError, match="@"):
            _ = A * v


def test_vec3_and_mat3_times_scalar_value_ok():
    with ir.Graph() as g:
        s = ir.Scalar.input("s")
        v = ir.Vec3[CraftFrame].input("v")
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        g.output(v * s, "vs")
        g.output(s * v, "sv")
        g.output(v / s, "vd")
        g.output(A * s, "As")


def test_scalar_times_quat_raises():
    with ir.Graph():
        s = ir.Scalar.input("s")
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        with pytest.raises(TypeError, match="quaternion"):
            _ = s * q


# ---------------------------------------------------------------------------
# coerce frame checks — two-frame types validate from_frame/to_frame
# ---------------------------------------------------------------------------

def test_quat_coerce_wrong_frames_raises():
    with ir.Graph():
        q = ir.Quat[WorldFrame, CraftFrame].input("q")
        assert ir.Quat[WorldFrame, CraftFrame].coerce(q) is q
        with pytest.raises(FrameError):
            ir.Quat[WorldFrame, PartFrame].coerce(q)     # wrong to_frame
        with pytest.raises(FrameError):
            ir.Quat[CraftFrame, CraftFrame].coerce(q)    # wrong from_frame


def test_mat3_coerce_wrong_frames_raises():
    with ir.Graph():
        A = ir.Mat3[WorldFrame, CraftFrame].input("A")
        assert ir.Mat3[WorldFrame, CraftFrame].coerce(A) is A
        with pytest.raises(FrameError):
            ir.Mat3[WorldFrame, PartFrame].coerce(A)     # wrong to_frame
        with pytest.raises(FrameError):
            ir.Mat3[PartFrame, CraftFrame].coerce(A)     # wrong from_frame
