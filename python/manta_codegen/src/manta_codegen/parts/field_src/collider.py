"""Collider — adds a contact disturbance to a registered CollisionField
and applies the reaction wrench back on its part each tick.

v1 exposes the single-sphere constructor only (the most common case for
craft body protection). Multi-shape colliders are still callable from
C++ but aren't wired through codegen here yet — add when needed.
"""

from __future__ import annotations

from ..._format import cpp_float as _f, cpp_mfloat as _mf
from ...core import PartDescriptor
from ...fields import CollisionField


class Collider(PartDescriptor):
    """Single-sphere collider at the part's origin.

    Required fields: CollisionField.
    Telemetry: none.
    """

    cpp_class_template = "manta::parts::ColliderT"
    cpp_header         = "manta/parts/field_src/collider.hpp"
    requires_fields    = [CollisionField]
    scalar_templated   = True

    def __init__(self,
                 name: str,
                 radius: float,
                 k_normal: float = 1.0e6,
                 d_normal: float = 5.0e3,
                 mu_static: float = 0.7,
                 mu_kinetic: float = 0.5,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if radius <= 0.0:
            raise ValueError("Collider.radius must be > 0")
        self.radius     = float(radius)
        self.k_normal   = float(k_normal)
        self.d_normal   = float(d_normal)
        self.mu_static  = float(mu_static)
        self.mu_kinetic = float(mu_kinetic)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return (f'"{self.name}", '
                f'{_mf(self.radius)}, '
                f'{_mf(self.k_normal)}, {_mf(self.d_normal)}, '
                f'{_mf(self.mu_static)}, {_mf(self.mu_kinetic)}')
