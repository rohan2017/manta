"""OpticalSource — a part that emits a semantic ellipsoid for cameras.

Bolt it to a vehicle and any `Camera` on another craft can see it: the
source emits a `BodySemanticEllipsoid` (a coarse 3-D extent + class
label) that rides this craft's pose, and the camera projects that quadric
to an image-frame bounding box. This is what lets vehicles see each other
in the sim — the optical analogue of a transponder.
"""

from __future__ import annotations

from ...fields.optical import BodySemanticEllipsoid, OpticalField
from ..base import Parameter
from .base import FieldSource


class OpticalSource(FieldSource):
    """Emits a semantic ellipsoid that rides the carrying craft.

    Parameters:
        semi_axes — (a, b, c) half-extents of the bounding ellipsoid along
                    the craft's body axes, m. Roughly half the vehicle's
                    length/width/height.
        label     — integer class id the camera reports with each box.
    """

    emits_field = OpticalField

    semi_axes: tuple = Parameter((1.0, 1.0, 1.0))
    label:     int   = Parameter(0)

    def make_disturbance(self, craft, offset_body):
        return BodySemanticEllipsoid(
            craft, offset_body, semi_axes=tuple(self.semi_axes),
            label=int(self.label), name=f"{craft.name}_{self.name}")
