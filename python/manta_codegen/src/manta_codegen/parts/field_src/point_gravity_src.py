"""PointGravitySrc — adds an inverse-square gravity disturbance to GravityField."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...fields import GravityField


class PointGravitySrc(PartDescriptor):
    """A part that *contributes* an inverse-square gravity disturbance to the
    registered GravityField, centered at the part's scene-frame position. Use
    this for self-gravitating bodies (asteroids, space stations, exotic
    gravity generators).

    The part has no mass or MOI of its own; only `grav_mass` (in kg) sets the
    strength via μ = G · grav_mass.

    By default the disturbance is added per-tick (lifetime=1) so the source
    tracks part motion. Set `persistent=True` for a static source that's
    added once and never re-added — saves the per-tick allocation.

    Required fields: GravityField.
    Telemetry: none.
    """

    cpp_class_template = "manta::parts::PointGravitySrcT"
    cpp_header         = "manta/parts/field_src/point_gravity_src.hpp"
    requires_fields    = [GravityField]

    def __init__(self,
                 name: str,
                 grav_mass: float,
                 persistent: bool = False,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.grav_mass  = float(grav_mass)
        self.persistent = bool(persistent)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return f'"{self.name}", {scalar}({_f(self.grav_mass)})'

    def emit_post_construction(self, scalar: str = "manta::MFloat") -> list[str]:
        out = list(super().emit_post_construction(scalar))
        if self.persistent:
            out.append(f"{self.name}_->set_persistent(true);")
        return out
