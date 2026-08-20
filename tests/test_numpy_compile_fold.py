"""TargetNumpy compilation + substep folding — both must be bit-identical
to the plain interpreted sequential stepping.

`compile=True` swaps the interpreted CasADi functions for `cc -O1`-built
externals or raises an actionable compilation error;
`NumpySim.step_n(dt, n)` folds n substeps through one `mapaccum` call. Neither
may change the answer.
"""

import shutil

import casadi as ca
import numpy as np
import pytest

from manta import (
    CompilationError,
    Craft,
    Sim,
    TargetNumpy,
    World,
    compile_functions,
)
from manta.codegen.numpy import _compile
from manta.fields import FluidField, GravityField
from manta.parts import DragSurface, Mass, RevoluteJoint, Thruster


def _world():
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(1.0, 1.0, 0.5), mount_offset=(0, 0, 0.2)))
    c.add(DragSurface.isotropic_quadratic("aero", area=0.1, drag_coefficient=0.5))
    j = RevoluteJoint("gim", axis=(1, 0, 0), mode="saturating",
                      stall_torque=20.0, damping=1.0, mount_offset=(0, 0, -0.5))
    j.add(Thruster("t", force=(0, 0, 60.0), mount_offset=(0, 0, -0.2)))
    c.add(j)
    w = (World().add_field(GravityField(g=(0, 0, -9.81)))
                .add_field(FluidField().add_uniform(density=1.225)))
    w.add_craft(c, position=(0, 0, 100.0), velocity=(2.0, 0.0, 8.0))
    return w


def _drive(sim, *, fold, nsub=8, nctrl=40):
    for i in range(nctrl):
        u = {"c.t.throttle": 0.4 + 0.2 * np.sin(i * 0.3),
             "c.gim.torque_cmd": 3.0 * np.cos(i * 0.2)}
        if fold:
            sim.step_n(0.001, nsub, u=u)
        else:
            for _ in range(nsub):
                sim.step(0.001, u=u)
    return (np.asarray(sim.state["c"]["position"]).ravel(),
            np.asarray(sim.state["c"]["velocity"]).ravel())


def test_compiled_matches_interpreted():
    if shutil.which("cc") is None:
        pytest.skip("no C compiler on PATH")
    base = _drive(TargetNumpy(Sim(_world())), fold=False)
    runtime = TargetNumpy(Sim(_world()), compile=True)
    assert runtime._functions["step"].class_name() == "External"
    comp = _drive(runtime, fold=False)
    # cc may reorder floating point, but must never diverge.
    assert np.allclose(comp[0], base[0], rtol=0, atol=1e-9)
    assert np.allclose(comp[1], base[1], rtol=0, atol=1e-9)


def test_step_n_matches_sequential():
    base = _drive(TargetNumpy(Sim(_world())), fold=False)
    fold = _drive(TargetNumpy(Sim(_world())), fold=True)
    # mapaccum applies the very same kernel in the same order — exactly equal.
    assert np.array_equal(fold[0], base[0])
    assert np.array_equal(fold[1], base[1])


def test_compile_and_fold_together():
    if shutil.which("cc") is None:
        pytest.skip("no C compiler on PATH")
    base = _drive(TargetNumpy(Sim(_world())), fold=False)
    both = _drive(TargetNumpy(Sim(_world()), compile=True), fold=True)
    assert np.allclose(both[0], base[0], rtol=0, atol=1e-9)
    assert np.allclose(both[1], base[1], rtol=0, atol=1e-9)


def test_explicit_compile_reports_an_unwritable_cache(monkeypatch, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    monkeypatch.setattr(_compile, "_cache_dir", lambda: str(blocked))
    monkeypatch.setattr(_compile.shutil, "which", lambda _name: "/usr/bin/cc")
    x = ca.MX.sym("compile_failure_x")
    fn = ca.Function("compile_failure_probe", [x], [x + 1.0])
    with pytest.raises(CompilationError, match="cannot create.*cache"):
        _compile._compiled_functions({"probe": fn})


def test_explicit_compile_reports_a_missing_compiler(monkeypatch):
    monkeypatch.setattr(_compile.shutil, "which", lambda _name: None)
    x = ca.MX.sym("missing_compiler_x")
    fn = ca.Function("missing_compiler_probe", [x], [x + 1.0])
    with pytest.raises(CompilationError, match="no 'cc' compiler"):
        _compile._compiled_functions({"probe": fn})


def test_explicit_compile_reports_the_cost_gate():
    x = ca.MX.sym("oversized_compile_x")
    fn = ca.Function("oversized_compile_probe", [x], [x + 1.0])
    with pytest.raises(CompilationError, match="above the configured limit"):
        _compile._compiled_functions({"probe": fn}, max_instr=0)


def test_public_compile_functions_supports_owned_prototype_kernels(
    monkeypatch, tmp_path,
):
    if shutil.which("cc") is None:
        pytest.skip("no C compiler on PATH")
    monkeypatch.setattr(_compile, "_cache_dir", lambda: str(tmp_path))
    x = ca.MX.sym("public_compile_x")
    fn = ca.Function("public_compile_probe", [x], [x * x + 2.0])
    compiled = compile_functions({"probe": fn})["probe"]
    assert compiled.class_name() == "External"
    assert float(compiled(3.0)) == pytest.approx(11.0)
    renamed = compile_functions({"renamed": fn})
    assert set(renamed) == {"renamed"}
    assert float(renamed["renamed"](4.0)) == pytest.approx(18.0)


def test_runtime_can_compile_only_a_selected_hot_function(monkeypatch, tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("no C compiler on PATH")
    monkeypatch.setattr(_compile, "_cache_dir", lambda: str(tmp_path))
    runtime = TargetNumpy(Sim(_world()))
    before = dict(runtime._functions)

    returned = runtime.compile_functions(("step",), optimization="startup")

    assert returned is runtime
    assert runtime._functions["step"].class_name() == "External"
    assert all(
        runtime._functions[name] is function
        for name, function in before.items()
        if name != "step"
    )
    with pytest.raises(KeyError, match="unknown Module function"):
        runtime.compile_functions(("not_a_kernel",))
