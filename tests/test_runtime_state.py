"""Numpy runtime state-handling contracts.

Pins the failure modes the 2026-08 audit found in the mutable nested-dict
state API (the surface shiver's sim runner drives): typo'd keys must
raise instead of silently no-oping, runtimes must not alias the Module's
init arrays, the filter clock must advance like the sim's, and the edge
cases (`step_n(dt, 0)`, whole-dict assignment) must behave.
"""

import numpy as np
import pytest

from manta import EKF, Sim, TargetNumpy, World
from manta.craft import Craft
from manta.fields import GravityField
from manta.parts import Mass, PositionSensor


def _world(name="w"):
    c = Craft("d")
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=0.05))
    w = World(name=name).add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c, position=(0.0, 0.0, 10.0))
    return w


def test_typoed_state_key_raises():
    """A misspelled slot in `sim.state` fails loudly at the next step —
    `pack_any`'s unknown-keys-ignored default made it an invisible no-op
    that read exactly like a physics bug."""
    sim = TargetNumpy(Sim(_world()))
    sim.state["d"]["positon"] = np.array([1.0, 2.0, 3.0])   # typo
    with pytest.raises(KeyError, match="positon"):
        sim.step(0.01)


def test_state_setter_validates():
    """Whole-dict assignment must carry every state slot and no unknown
    keys — `sim.state = {}` used to be accepted and fail one step later
    with a bare KeyError from inside the packer."""
    sim = TargetNumpy(Sim(_world()))
    with pytest.raises(ValueError, match="missing"):
        sim.state = {}
    with pytest.raises(TypeError, match="dict"):
        sim.state = None
    # A complete replacement is fine.
    fresh = sim.initial_state()
    fresh["d"]["position"] = np.array([0.0, 0.0, 5.0])
    sim.state = fresh
    assert sim.state["d"]["position"][2] == 5.0


def test_runtimes_do_not_alias_module_init():
    """Two runtimes over one transform own independent state — the
    matrix branch (EKF's P) used to be a reshape VIEW onto the shared
    Module's `StateField.init`, so writing one runtime's P mutated the
    other's (and the supposedly immutable Module)."""
    ir = EKF(_world())
    r1, r2 = TargetNumpy(ir), TargetNumpy(ir)
    module_init = np.asarray(ir.module().state.field("P").init)
    before = module_init.copy()
    r1._state["P"][0, 0] = 42.0
    assert r2.P[0, 0] != 42.0, "runtime P aliased across runtimes"
    assert np.array_equal(module_init, before), "Module init was mutated"


def test_filter_clock_advances_like_sim():
    """`predict(dt)` with no explicit t advances the filter clock, same
    convention as `NumpySim.step` — a filter pinned at t=0 while the sim
    advanced left time-dependent worlds linearized at t=0, silently."""
    f = TargetNumpy(EKF(_world()))
    assert f._t == 0.0
    f.predict(0.25)
    f.predict(0.25)
    assert f._t == pytest.approx(0.5)
    f.predict(0.25, t=10.0)                    # explicit t resyncs
    assert f._t == pytest.approx(10.25)
    f.reset()
    assert f._t == 0.0


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_runtimes_reject_invalid_timestep(dt):
    sim = TargetNumpy(Sim(_world()))
    filt = TargetNumpy(EKF(_world()))
    with pytest.raises(ValueError):
        sim.step(dt)
    with pytest.raises(ValueError):
        filt.predict(dt)


def test_reset_resets_covariance_and_explicit_state_move_preserves_it():
    filt = TargetNumpy(EKF(_world()))
    default = filt.P.copy()
    filt._state["P"] = np.eye(filt.spec.tangent_dim) * 7.0
    filt.reset(state={"d": {"position": [1.0, 2.0, 3.0]}})
    assert np.array_equal(filt.P, default)
    kept = np.eye(filt.spec.tangent_dim) * 5.0
    filt._state["P"] = kept.copy()
    filt.set_state_keep_covariance({"d": {"position": [4.0, 5.0, 6.0]}})
    assert np.array_equal(filt.P, kept)


def test_step_n_zero_returns_state():
    """`step_n(dt, 0)` is a no-op that still returns the state dict (it
    used to return None because the loop never seeded the lazy state)."""
    sim = TargetNumpy(Sim(_world()))
    out = sim.step_n(0.01, 0)
    assert out is not None
    assert out["d"]["position"][2] == 10.0
