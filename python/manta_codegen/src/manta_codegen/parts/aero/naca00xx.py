"""Naca00xx — symmetric NACA 4-digit airfoil descriptor.

A rectangular planform (chord × span) discretized into N spanwise segments.
Each segment is treated as a 2-D airfoil section ("blade-element"); the
per-segment lift/drag is integrated over the wing.

Required field: FluidField (provides ρ and the ambient velocity at each
sample point).

Codegen note: only the geometry / sample-count knobs go through the
descriptor constructor. The post-stall blend and the empirical
coefficients (Cl_α, Cd0, α_stall, Cd_α²) live as setter methods on the
C++ class — tune them at runtime via `craft.<name>().set_Cl_alpha(...)`
etc. if you need to.
"""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...fields.fluid_field import FluidField


class Naca00xx(PartDescriptor):
    cpp_header         = "manta/parts/aero/naca00xx.hpp"
    cpp_class_template = "manta::parts::Naca00xxT"
    requires_fields    = [FluidField]
    scalar_templated   = True

    def __init__(self,
                 name: str,
                 chord: float,
                 span: float,
                 thickness_ratio: float,
                 n_sample_points: int = 8,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if chord <= 0.0:
            raise ValueError("Naca00xx.chord must be > 0")
        if span <= 0.0:
            raise ValueError("Naca00xx.span must be > 0")
        if not (0.0 < thickness_ratio < 1.0):
            raise ValueError("Naca00xx.thickness_ratio must be in (0, 1)")
        if n_sample_points < 1:
            raise ValueError("Naca00xx.n_sample_points must be >= 1")
        self.chord            = float(chord)
        self.span             = float(span)
        self.thickness_ratio  = float(thickness_ratio)
        self.n_sample_points  = int(n_sample_points)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return (f'"{self.name}", '
                f'manta::MFloat({_f(self.chord)}), '
                f'manta::MFloat({_f(self.span)}), '
                f'manta::MFloat({_f(self.thickness_ratio)}), '
                f'{self.n_sample_points}')
