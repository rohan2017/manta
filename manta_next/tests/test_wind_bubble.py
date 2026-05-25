"""CraftWindBubble — per-craft estimable wind disturbance.

Capstone test for the field-bus design:
  * Each craft maintains its own RW-evolving wind state.
  * The bubble is a hard-edged sphere centered on the craft's
    symbolic position (via `craft._sym_state`).
  * Overlapping bubbles use the FluidField's "averaged" combining so
    crafts compromise on the mean of their estimates inside overlap.
  * EKF picks up each bubble's wind as a separate state slot with
    auto-Q = dt·σ²·I per slot.
"""

import casadi as ca
import numpy as np

from manta_next import Craft, EKF, World, TargetNumpy
from manta_next.fields import (
    CraftWindBubble, FluidField, GravityField,
)
from manta_next.parts import Mass


def _build_world(*, n_crafts: int = 1, sigma: float = 1e-3,
                 radius: float = 20.0):
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    ff = FluidField()
    crafts = []
    for i in range(n_crafts):
        c = Craft(f"c{i}"); c.add(Mass("body", mass=1.0))
        w.add_craft(c, position=(float(i * 5), 0.0, 5.0))
        ff.add(CraftWindBubble(craft=c, radius=radius, sigma=sigma))
        crafts.append(c)
    w.add_field(ff)
    return w, crafts


# ---------------------------------------------------------------------------

def test_compiled_world_exposes_per_craft_wind_state():
    w, crafts = _build_world(n_crafts=2)
    cw = TargetNumpy(w.compile())
    init = cw.initial_state()
    assert {c.name for c in crafts} <= set(init)
    assert "c0_wind" in init and "c1_wind" in init
    for owner in ("c0_wind", "c1_wind"):
        assert "wind"        in init[owner]
        assert "wind_driver" in init[owner]


def test_wind_state_evolves_via_rw():
    w, _ = _build_world(n_crafts=1)
    cw = TargetNumpy(w.compile())
    state = cw.initial_state()
    drv = np.array([0.5, -0.25, 0.1])
    dt = 0.01
    expected = np.zeros(3)
    for _ in range(20):
        state["c0_wind"]["wind_driver"] = drv.copy()
        state = cw.step(state, dt=dt)
        expected = expected + np.sqrt(dt) * drv
    np.testing.assert_allclose(state["c0_wind"]["wind"], expected,
                               atol=1e-10)


def test_ekf_picks_up_wind_state_per_craft():
    w, crafts = _build_world(n_crafts=2, sigma=2e-3)
    ekf = TargetNumpy(EKF(w))
    # 2 crafts × 13 rigid + 2 wind bubbles × 3 = 32 ambient.
    assert ekf.spec.ambient_dim == 32
    names = {s.name for s in ekf.spec.slots}
    assert {"c0_wind.wind", "c1_wind.wind"} <= names
    # Auto-Q assembles dt·σ² on each wind slot.
    L = np.asarray(ekf.ekf._L_fn(ekf.x, np.zeros(0), 0.01, 0.0))
    Q = L @ ekf.ekf._Sigma @ L.T
    # Both wind slots get dt·(2e-3)² = 4e-8 on diagonal.
    c0_wind_slot = ekf.spec.slot("c0_wind.wind")
    block = Q[c0_wind_slot.tangent_offset : c0_wind_slot.tangent_offset+3,
              c0_wind_slot.tangent_offset : c0_wind_slot.tangent_offset+3]
    np.testing.assert_allclose(block, 0.01 * (2e-3)**2 * np.eye(3),
                               atol=1e-15)


def test_state_dict_nests_wind_alongside_crafts():
    w, _ = _build_world(n_crafts=1)
    ekf = TargetNumpy(EKF(w))
    sd = ekf.state_dict()
    assert "c0" in sd and "c0_wind" in sd
    assert "wind" in sd["c0_wind"]


def test_unique_bubble_names_for_distinct_crafts():
    """Two bubbles with auto-generated names from distinct crafts
    don't collide (the default is `<craft.name>_wind`)."""
    w, crafts = _build_world(n_crafts=3)
    cw = TargetNumpy(w.compile())
    owners = set(cw.initial_state())
    assert {"c0", "c1", "c2", "c0_wind", "c1_wind", "c2_wind"} <= owners
