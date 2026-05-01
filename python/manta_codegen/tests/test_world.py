"""Phase 3a tests: World API — add_planet/add_field/add_craft, run().

Pure Python; verifies the World container records planets, fields, crafts
with their initial state, and the sim-loop config; verifies the back-compat
`world_from_craft` shim correctly hoists Craft.fields/planets/initial_state
into a synthetic World.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from manta_codegen.core import Craft, World, world_from_craft
from manta_codegen.parts.structure.point_mass import PointMass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_world_add_craft_default_world_frame() -> None:
    body = PointMass("body", mass=1.0)
    c = Craft("ex_world")
    c.root.add(body)
    w = World()
    w.add_craft(c, pos=(1.0, 2.0, 3.0), vel=(0.0, 0.5, 0.0))
    _check(len(w.crafts) == 1, "craft not recorded")
    e = w.crafts[0]
    _check(e.craft is c, "wrong craft handle")
    _check(e.on is None, "default frame must be world (on=None)")
    _check(e.position == (1.0, 2.0, 3.0), f"unexpected position {e.position}")
    _check(e.vel_linear == (0.0, 0.5, 0.0), f"unexpected velocity {e.vel_linear}")


def test_world_run_sets_dt_and_rate_mult() -> None:
    w = World().run(dt=0.005, sim_rate_mult=200.0)
    _check(w.dt == 0.005, "dt not set")
    _check(w.sim_rate_mult == 200.0, "sim_rate_mult not set")


def test_world_add_craft_rejects_unregistered_planet() -> None:
    from manta_codegen.core import PlanetDescriptor

    class FakePlanet(PlanetDescriptor):
        cpp_class  = "manta::FakePlanet"
        cpp_header = "manta/fake.hpp"

    p = FakePlanet()
    c = Craft("ex")
    c.root.add(PointMass("body", mass=1.0))
    w = World()
    try:
        w.add_craft(c, on=p)   # planet not registered first
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_world_chaining_returns_world() -> None:
    body = PointMass("body", mass=1.0)
    c = Craft("ex")
    c.root.add(body)
    w = (World()
         .add_craft(c, pos=(0, 0, 1))
         .run(dt=0.001))
    _check(len(w.crafts) == 1, "chained add_craft didn't record")
    _check(w.dt == 0.001, "chained run didn't set dt")


def test_world_from_craft_legacy_shim() -> None:
    """A Craft with sim_config + initial_state should round-trip through the
    backward-compat shim into a synthetic World with matching fields."""
    body = PointMass("body", mass=1.0)
    c = (Craft("legacy")
         .sim_config(dt=0.002, sim_rate_mult=10.0)
         .initial_state(position=(1, 2, 3),
                        vel_linear=(0, 0.5, 0)))
    c.root.add(body)
    w = world_from_craft(c)
    _check(w.dt == 0.002, "dt didn't propagate")
    _check(w.sim_rate_mult == 10.0, "rate mult didn't propagate")
    _check(len(w.crafts) == 1, "craft missing")
    e = w.crafts[0]
    _check(e.position == (1.0, 2.0, 3.0), "initial position lost")
    _check(e.vel_linear == (0.0, 0.5, 0.0), "initial velocity lost")
    _check(e.on is None, "legacy crafts have no planet anchor")


if __name__ == "__main__":
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
