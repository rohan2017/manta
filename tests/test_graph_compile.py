"""End-to-end: build a graph, compile to a CasADi Function, evaluate on numpy."""

import numpy as np
import pytest

from manta import ir
from manta.ir.frames import CraftFrame, WorldFrame


def test_translate_by_velocity():
    """The end-state target from the M0 spec — verifies tracing + compile + call."""
    with ir.Graph() as g:
        p_scene = ir.Vec3[WorldFrame].input("p_scene")
        v_craft = ir.Vec3[CraftFrame].input("v_craft")
        q       = ir.Quat[WorldFrame, CraftFrame].input("q")
        dt      = ir.Scalar.input("dt")

        v_scene = q.apply(v_craft)
        p_next  = p_scene + v_scene * dt
        ke      = v_scene.dot(v_scene) * 0.5

        g.output(p_next, "p_next")
        g.output(ke,     "ke")

    fn = g.compile()
    out = fn(
        p_scene=np.array([0.0, 0.0, 5.0]),
        v_craft=np.array([1.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),     # identity
        dt=0.001,
    )
    assert np.allclose(out["p_next"], [0.001, 0.0, 5.0])
    assert np.isclose(out["ke"], 0.5)


def test_compile_then_call_with_python_scalars():
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        b = ir.Scalar.input("b")
        g.output(a + b, "sum")
        g.output(a * b, "prod")

    fn = g.compile()
    out = fn(a=3.0, b=4.0)
    assert np.isclose(out["sum"], 7.0)
    assert np.isclose(out["prod"], 12.0)


def test_missing_input_raises():
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        g.output(a * 2, "out")

    fn = g.compile()
    with pytest.raises(ValueError, match="missing input"):
        fn()  # 'a' not provided


def test_extra_input_raises():
    with ir.Graph() as g:
        a = ir.Scalar.input("a")
        g.output(a * 2, "out")

    fn = g.compile()
    with pytest.raises(ValueError, match="unknown input"):
        fn(a=1.0, b=2.0)


def test_duplicate_input_name_raises():
    with ir.Graph():
        ir.Scalar.input("dup")
        with pytest.raises(ValueError, match="duplicate input"):
            ir.Scalar.input("dup")


def test_compile_with_no_outputs_raises():
    with ir.Graph() as g:
        ir.Scalar.input("x")
        with pytest.raises(ValueError, match="no outputs"):
            g.compile()


def test_summary_includes_frames_and_io():
    with ir.Graph(name="demo") as g:
        a = ir.Vec3[WorldFrame].input("a")
        b = ir.Vec3[WorldFrame].input("b")
        g.output(a + b, "sum")

    s = g.summary()
    assert "demo" in s
    assert "WorldFrame" in s     # WorldFrame is the alias
    assert "sum" in s
    assert "Inputs (2)" in s
    assert "Outputs (1)" in s


def test_no_active_graph_raises():
    with pytest.raises(RuntimeError, match="No active Graph"):
        ir.Scalar.input("x")


def test_nested_graphs_swap_active():
    with ir.Graph() as outer:
        ir.Scalar.input("a")
        with ir.Graph() as inner:
            ir.Scalar.input("b")     # registers on inner
            assert "b" in inner.inputs
            assert "b" not in outer.inputs
        ir.Scalar.input("c")          # registers on outer again
    assert set(outer.inputs.keys()) == {"a", "c"}
    assert set(inner.inputs.keys()) == {"b"}
