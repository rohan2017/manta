"""Coupling ABC.

A `Coupling` joins two crafts so their dynamics are evaluated in the
same compiled tick. Concrete subclasses (e.g. `Tether`) implement
`compute_wrenches_sym(ctx_a, ctx_b)` and World.compile routes coupled
components through `compile_world_tick`.

This module is the canonical home of the ABC. Import via:
    from manta.couplings import Coupling   # or
    from manta import Coupling
"""

from __future__ import annotations


class Coupling:
    """Abstract base for inter-craft constraints.

    Concrete subclasses (Tether, ContactConstraint, RigidLatch, …)
    declare two craft endpoints and produce extra wrench/state terms in
    the tick graph for the connected component. The presence of a
    Coupling forces both crafts into the same compile unit.
    """

    @property
    def craft_a(self):
        raise NotImplementedError

    @property
    def craft_b(self):
        raise NotImplementedError
