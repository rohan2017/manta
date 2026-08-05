"""FieldSource — base for parts that add a disturbance to a field.

A subclass declares which Field class it contributes to
(`emits_field`) and how to build its craft-anchored `Disturbance`
(`make_disturbance`). The world, at compile time, walks every craft for
FieldSource parts and adds each one's disturbance to that field — see
`World.finalize()`. A source does NOT provide the field: the
target field must already be registered on the world (a `GravitySource`
requires a `GravityField`), else compilation raises. The disturbance
follows the carrying craft's pose via the active trace, so no field
plumbing changes per source.
"""

from __future__ import annotations

import numpy as np

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from ...ir.wrench import Wrench
from .._declarations import PartUpdate
from ..base import Part, RootPart


def cumulative_offset(part) -> np.ndarray:
    """The part's rest-pose position in its craft's body frame: the sum of
    `mount_offset` along the parent chain up to (excluding) the craft
    root.
    Joint angles are ignored — a source rides its mount's REST offset
    (fine for the typical root-mounted source; documented)."""
    off = np.zeros(3)
    cur = part
    while cur is not None and not isinstance(cur, RootPart):
        off = off + np.asarray(cur.mount_offset, dtype=float)
        cur = cur.parent
    return off


class FieldSource(Part):
    """Base for a part that adds a disturbance to a world field.

    Subclasses set the class attribute `emits_field` (the `Field`
    subclass the disturbance belongs to — which must be registered on the
    world) and implement `make_disturbance(craft, offset_body)` returning
    the craft-anchored `Disturbance`. A field source contributes no
    wrench."""

    #: The Field subclass this source contributes a disturbance to. The
    #: field must be registered on the world. Subclasses override.
    emits_field: type | None = None

    def make_disturbance(self, craft, offset_body):
        """Build the craft-anchored Disturbance this source emits.
        `offset_body` is the source's rest position in the craft body
        frame (from `cumulative_offset`)."""
        raise NotImplementedError(
            f"{type(self).__name__}: must implement make_disturbance().")

    def update(self, ctx) -> PartUpdate:
        # A field source is a pure emitter — no force on its carrier.
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero, torque=zero), new_state={})
