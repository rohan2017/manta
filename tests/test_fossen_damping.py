"""FossenDamping — the reduced template's 6×6 D as one part.

Three contracts: the diagonal is EXACTLY DragSurface + RotationalDrag
(so the fitted template subsumes the hand-built hulls); the
off-diagonal quadrants really couple (sway↔yaw both ways — the
Fossen terms Y_r and N_v that no single-point part can express); and
the polynomial orders follow the house sign-preserving-power
convention shared with the other damping parts.
"""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import (
    DragSurface, FossenDamping, Mass, RotationalDrag,
)

RHO = 1025.0


def _world(*parts, v0=(0.0, 0.0, 0.0), w0=(0.0, 0.0, 0.0)):
    c = Craft("hull")
    c.add(Mass("m", mass=12.0, moi=(0.3, 1.5, 1.5)))
    for p in parts:
        c.add(p)
    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
         .add_field(FluidField().add_uniform(density=RHO)))
    w.add_craft(c, velocity=v0, angular_velocity=w0)
    return TargetNumpy(Sim(w))


def _run(sim, n=40, dt=0.05):
    for _ in range(n):
        sim.step(dt)
    st = sim.state["hull"]
    return (np.asarray(st["velocity"], dtype=float),
            np.asarray(st["angular_velocity"], dtype=float),
            np.asarray(st["position"], dtype=float))


def test_diagonal_is_exactly_dragsurface_plus_rotationaldrag():
    """One FossenDamping with a diagonal D₁ must integrate bit-for-bit
    like the DragSurface(force=) + RotationalDrag(torque=) pair it
    subsumes — from a state that exercises every axis at once."""
    kv = (-2.0 / RHO, -5.0 / RHO, -5.0 / RHO)
    kw = (-0.5 / RHO, -1.5 / RHO, -1.5 / RHO)
    v0, w0 = (1.2, -0.4, 0.3), (0.8, -0.5, 1.1)

    fused = _world(FossenDamping("d", damping=kv + kw), v0=v0, w0=w0)
    split = _world(DragSurface("lin", force=kv),
                   RotationalDrag("spin", torque=kw), v0=v0, w0=w0)
    for a, b in zip(_run(fused), _run(split)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_cross_coupling_quadrants_couple_both_ways():
    """The reason this part exists: N_v (yaw moment from sway) and
    Y_r (sway force from yaw rate) — the off-diagonal quadrants no
    single-point part can express."""
    D = np.zeros((6, 6))
    D[5, 1] = -3.0 / RHO          # N_v: sway velocity → yaw torque

    sim = _world(FossenDamping("d", tensors=[D]), v0=(0.0, 1.0, 0.0))
    _, w_end, _ = _run(sim, n=10)
    assert w_end[2] < -1e-3, "sway did not induce yaw"
    assert abs(w_end[0]) < 1e-12 and abs(w_end[1]) < 1e-12

    D = np.zeros((6, 6))
    D[1, 5] = -2.0 / RHO          # Y_r: yaw rate → sway force
    sim = _world(FossenDamping("d", tensors=[D]), w0=(0.0, 0.0, 1.0))
    v_end, w_end, _ = _run(sim, n=10)
    # the only force in the world is the coupling — speed from rest
    # proves it acted (the body yaws while pushed, so the WORLD
    # velocity spreads over x and y; z stays untouched)
    assert np.linalg.norm(v_end) > 1e-3, "yaw rate induced no force"
    assert abs(v_end[2]) < 1e-12
    assert w_end[2] == pytest.approx(1.0)     # and no phantom torque


def test_quadratic_order_follows_the_house_power_convention():
    """A pure second-order D₂ produces force ∝ v·|v|: doubling the
    initial speed quadruples the initial deceleration (compared over a
    vanishing first step), and the sign is preserved for negative v."""
    D2 = np.zeros((6, 6))
    D2[0, 0] = -4.0 / RHO
    dt = 1e-3

    decels = []
    for vx in (1.0, 2.0, -1.0):
        sim = _world(FossenDamping("d", tensors=[np.zeros((6, 6)), D2]),
                     v0=(vx, 0.0, 0.0))
        sim.step(dt)
        v = float(np.asarray(sim.state["hull"]["velocity"])[0])
        decels.append((vx - v) / dt)
    assert decels[1] / decels[0] == pytest.approx(4.0, rel=1e-3)
    assert decels[2] == pytest.approx(-decels[0], rel=1e-6)


def test_d_tensor_promotes_to_a_live_parameter():
    """The reducer's whole point, end to end: D1 is a promotable R36
    Parameter (the generalized manifold grammar), so one compiled
    graph serves every D — `manta.fit` iterates without recompiling,
    and a deployed runtime takes a refit D through `set_parameters`.

    Pinned both ways: promoted-at-default integrates bit-for-bit like
    the baked world, and promoted-then-retargeted integrates bit-for-
    bit like a world REBUILT with the new tensor."""
    Da = np.zeros((6, 6))
    Da[0, 0], Da[1, 1], Da[5, 5] = -2.0 / RHO, -5.0 / RHO, -1.5 / RHO
    Da[5, 1], Da[1, 5] = -3.0 / RHO, -1.0 / RHO
    Db = 2.5 * Da
    v0, w0 = (1.0, -0.6, 0.2), (0.3, -0.2, 0.9)

    def world(D):
        c = Craft("hull")
        c.add(Mass("m", mass=12.0, moi=(0.3, 1.5, 1.5)))
        c.add(FossenDamping("d", tensors=[D]))
        w = (World()
             .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
             .add_field(FluidField().add_uniform(density=RHO)))
        w.add_craft(c, velocity=v0, angular_velocity=w0)
        return w

    prom = TargetNumpy(Sim(world(Da), parameters=["hull.d.D1"]))
    for a, b in zip(_run(prom), _run(TargetNumpy(Sim(world(Da))))):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    prom2 = TargetNumpy(Sim(world(Da), parameters=["hull.d.D1"]))
    prom2.set_parameters({"hull.d.D1": Db.flatten(order="F")})
    for a, b in zip(_run(prom2), _run(TargetNumpy(Sim(world(Db))))):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_constructor_contracts():
    with pytest.raises(ValueError, match="exactly one"):
        FossenDamping("d")
    with pytest.raises(ValueError, match="exactly one"):
        FossenDamping("d", damping=(0,) * 6, tensors=[np.zeros((6, 6))])
    with pytest.raises(ValueError, match="length-6"):
        FossenDamping("d", damping=(-1.0, -1.0, -1.0))
    with pytest.raises(ValueError):
        FossenDamping("d", tensors=[np.zeros((3, 3))])
