"""Block-diagonal predict — when a world holds several independent
subsystems (uncoupled crafts estimated jointly), the EKF partitions the
tangent state and propagates each block's covariance on its own. The
result is exactly the dense `F P Fᵀ + Q` when the partition is honored.
"""

import numpy as np

from manta import World, TargetNumpy
from manta.craft import Craft
from manta.fields import GravityField
from manta.estimation import EKF
from manta.parts.structure.mass import Mass
from manta.parts.sensor.position_sensor import PositionSensor


def _gps_craft(name):
    c = Craft(name)
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", position_noise_sigma=0.2))
    return c


# ---------------------------------------------------------------------------
# Single craft → one block (dense path)
# ---------------------------------------------------------------------------

def test_single_craft_is_one_block():
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(_gps_craft("solo"))
    assert EKF(w).n_blocks == 1


# ---------------------------------------------------------------------------
# Independent crafts → one block each
# ---------------------------------------------------------------------------

def test_independent_crafts_split_into_blocks():
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(_gps_craft("a"))
    w.add_craft(_gps_craft("b"))
    w.add_craft(_gps_craft("c"))

    ekf = EKF(w)
    assert ekf.n_blocks == 3
    # Every block is exactly one craft's 12 tangent dims.
    assert sorted(len(b) for b in ekf.sys.blocks) == [12, 12, 12]


# ---------------------------------------------------------------------------
# Coupled crafts collapse into a single block
# ---------------------------------------------------------------------------

def test_tethered_crafts_are_one_block():
    from manta.couplings import Tether
    from manta.parts import TetherEndpoint

    def craft(name):
        c = Craft(name)
        c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        c.add(TetherEndpoint("hook"))
        return c

    a, b = craft("a"), craft("b")
    w = World()
    w.add_craft(a, position=(0, 0, 0))
    w.add_craft(b, position=(5, 0, 0))
    w.add_coupling(Tether(a, "hook", b, "hook",
                          stiffness=10.0, damping=1.0, rest_length=5.0))
    # The tether makes each craft's dynamics depend on the other → the
    # partition must not split them.
    assert EKF(w).n_blocks == 1


# ---------------------------------------------------------------------------
# Block predict reproduces the dense propagation exactly
# ---------------------------------------------------------------------------

def test_block_predict_matches_dense():
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(_gps_craft("a"))
    w.add_craft(_gps_craft("b"))

    ir = EKF(w)
    assert ir.n_blocks == 2
    rt = TargetNumpy(ir)

    # A block-diagonal P (default is identity-scaled, already block-diag).
    rng = np.random.default_rng(3)
    n = ir.spec.tangent_dim
    P0 = np.zeros((n, n))
    for blk in ir.sys.blocks:
        M = rng.normal(size=(len(blk), len(blk)))
        P0[np.ix_(blk, blk)] = M @ M.T + np.eye(len(blk))
    rt.reset(P=P0)

    dt, t = 0.02, 0.0
    for step in range(5):
        x_before = rt.x.copy()
        P_before = rt.P.copy()
        rt.predict(dt=dt, t=t)

        # Dense reference from the same pre-step state.
        u_vec = ir.sys.resolve_u(None, who="test")
        F = np.asarray(ir.sys.F_fn(x_before, u_vec, dt, t))
        L = np.asarray(ir.sys.L_fn(x_before, u_vec, dt, t))
        Q = L @ ir.sys.Sigma @ L.T
        P_dense = F @ P_before @ F.T + Q
        P_dense = 0.5 * (P_dense + P_dense.T)

        np.testing.assert_allclose(rt.P, P_dense, atol=1e-12)
        t += dt

    # Cross-block covariance must remain exactly zero.
    a_idx, b_idx = ir.sys.blocks
    assert np.all(rt.P[np.ix_(a_idx, b_idx)] == 0.0)
