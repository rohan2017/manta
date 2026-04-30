"""Hull — sampled-points buoyancy."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import PartDescriptor
from ...fields import GravityField  # placeholder until FluidField descriptor lands


class Hull(PartDescriptor):
    """A buoyant body modeled by `sample_points` of equal volume V/N each.
    Each tick: query the registered FluidField at every sample, contribute
    F_i = -ρ_i · (V/N) · g_part applied AT the sample point.

    If the registered FluidField is an OceanAtmosField (runtime detected via
    dynamic_cast in C++), each sample's effective density is smoothstep-blended
    between water and air across `surface_smoothing` (m).

    Required fields: FluidField, GravityField (registration enforced at
    runtime; the codegen emits the corresponding feature-test macros so
    static_assert-based compile-time checks become possible later).
    """

    cpp_class          = "manta::parts::Hull"
    cpp_class_template = "manta::parts::HullT"
    cpp_header         = "manta/parts/structure/hull.hpp"
    # Note: FluidField descriptor doesn't exist yet. Add when it lands.
    requires_fields    = [GravityField]

    def __init__(self,
                 name: str,
                 volume: float,
                 sample_points: list[tuple[float, float, float]],
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if not sample_points:
            raise ValueError("Hull requires at least one sample point")
        self.volume = float(volume)
        self.sample_points = [tuple(float(x) for x in p) for p in sample_points]

    def emit_constructor_args(self, scalar: str = "manta::Real") -> str:
        pts = ", ".join(
            f"manta::geom::Vec3<manta::PartFrame, {scalar}>{{"
            f"{scalar}({_f(x)}), {scalar}({_f(y)}), {scalar}({_f(z)})}}"
            for (x, y, z) in self.sample_points
        )
        return (f'"{self.name}", {scalar}({_f(self.volume)}), '
                f'std::vector<manta::geom::Vec3<manta::PartFrame, {scalar}>>{{{pts}}}')
