"""TargetNumpy compilation + substep folding — both must be bit-identical
to the plain interpreted sequential stepping.

`compile=True` swaps the interpreted CasADi functions for `cc -O3`-built
externals (or stays interpreted if no compiler / the module is huge);
`NumpySim.step_n(dt, n)` folds n substeps through one `mapaccum` call. Neither
may change the answer.
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import DragSurface, Mass, RevoluteJoint, Thruster


def _world():
    c = Craft("c")
    c.add(Mass("body", mass=2.0, moi=(1.0, 1.0, 0.5), transform=(0, 0, 0.2)))
    c.add(DragSurface.isotropic_quadratic("aero", area=0.1, drag_coefficient=0.5))
    j = RevoluteJoint("gim", axis=(1, 0, 0), mode="saturating",
                      stall_torque=20.0, damping=1.0, transform=(0, 0, -0.5))
    j.add(Thruster("t", force=(0, 0, 60.0), transform=(0, 0, -0.2)))
    c.add(j)
    w = (World().add_field(GravityField(g=(0, 0, -9.81)))
                .add_field(FluidField().add_uniform(density=1.225)))
    w.add_craft(c, position=(0, 0, 100.0), velocity=(2.0, 0.0, 8.0))
    return w


def _drive(sim, *, fold, nsub=8, nctrl=40):
    for i in range(nctrl):
        sim.command("c.t.throttle").set(0.4 + 0.2 * np.sin(i * 0.3))
        sim.command("c.gim.torque_cmd").set(3.0 * np.cos(i * 0.2))
        if fold:
            sim.step_n(0.001, nsub)
        else:
            for _ in range(nsub):
                sim.step(0.001)
    return (np.asarray(sim.state["c"]["position"]).ravel(),
            np.asarray(sim.state["c"]["velocity"]).ravel())


def test_compiled_matches_interpreted():
    base = _drive(TargetNumpy(Sim(_world())), fold=False)
    comp = _drive(TargetNumpy(Sim(_world()), compile=True), fold=False)
    # bit-identical when interpreted-fallback; ~machine-eps when actually
    # compiled (cc may reorder FP), never divergence.
    assert np.allclose(comp[0], base[0], rtol=0, atol=1e-9)
    assert np.allclose(comp[1], base[1], rtol=0, atol=1e-9)


def test_step_n_matches_sequential():
    base = _drive(TargetNumpy(Sim(_world())), fold=False)
    fold = _drive(TargetNumpy(Sim(_world())), fold=True)
    # mapaccum applies the very same kernel in the same order — exactly equal.
    assert np.array_equal(fold[0], base[0])
    assert np.array_equal(fold[1], base[1])


def test_compile_and_fold_together():
    base = _drive(TargetNumpy(Sim(_world())), fold=False)
    both = _drive(TargetNumpy(Sim(_world()), compile=True), fold=True)
    assert np.allclose(both[0], base[0], rtol=0, atol=1e-9)
    assert np.allclose(both[1], base[1], rtol=0, atol=1e-9)
