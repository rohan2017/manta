"""Coupling ABC.

A `Coupling` joins two crafts so their dynamics are evaluated in the
same compiled tick. Concrete subclasses (e.g. `Tether`) declare two craft
endpoints and implement `compute_wrenches_sym(ctx_a, ctx_b)`; Sim(world)
routes coupled components through `compile_world_tick`.

This module is the canonical home of the ABC. Import via:
    from manta.couplings import Coupling   # or
    from manta import Coupling
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Coupling(ABC):
    """Abstract base for inter-craft constraints.

    A concrete subclass (e.g. `Tether`) declares two craft endpoints
    (`craft_a` / `craft_b`) and produces the extra wrench terms they
    exchange (`compute_wrenches_sym`) in the tick graph for the connected
    component. The presence of a Coupling forces both crafts into the same
    compile unit.
    """

    def __init__(self, name: str) -> None:
        from ..ir.module import check_name

        self.name = check_name(name, who=type(self).__name__)
        self._world = None

    @property
    @abstractmethod
    def craft_a(self):
        """The first coupled craft."""

    @property
    @abstractmethod
    def craft_b(self):
        """The second coupled craft."""

    @abstractmethod
    def compute_wrenches_sym(self, ctx_a, ctx_b):
        """The wrench pair `(wrench_on_a, wrench_on_b)` this coupling
        applies, given each craft's `TickContext`. Frames: each wrench is
        in its own craft's `CraftFrame`, at that craft's origin."""
