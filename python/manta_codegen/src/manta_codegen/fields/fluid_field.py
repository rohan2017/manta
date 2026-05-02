"""FluidField — superposition of fluid disturbances."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class FluidField(FieldDescriptor):
    """Single concrete fluid field aggregating uniform-gas and uniform-
    incompressible disturbances. For complex bounded disturbances (in_influence
    predicates) prefer building them in C++ — the codegen exposes only the
    simple uniform shapes.
    """

    cpp_class     = "manta::fields::FluidField"
    cpp_header    = "manta/fields/fluid_field.hpp"
    feature_macro = "MANTA_HAS_FLUID_FIELD"

    def __init__(self,
                 density: float | None = None,
                 velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """Convenience: passing `density` registers a single uniform-incompressible
        disturbance — matches the old `UniformFluidField` shape."""
        super().__init__()
        self._disturbances: list[dict] = []
        if density is not None:
            self.add_uniform_incompressible(density, velocity)

    def add_uniform_incompressible(self,
                                   density: float,
                                   velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
                                   ) -> "FluidField":
        self._disturbances.append({
            "kind":     "incompressible",
            "density":  float(density),
            "velocity": tuple(float(x) for x in velocity),
        })
        return self

    def add_uniform_gas(self,
                        R: float,
                        temperature: float,
                        pressure: float,
                        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
                        ) -> "FluidField":
        self._disturbances.append({
            "kind":        "gas",
            "R":           float(R),
            "temperature": float(temperature),
            "pressure":    float(pressure),
            "velocity":    tuple(float(x) for x in velocity),
        })
        return self

    def emit_construction(self, var_name: str) -> str:
        return f"{self.cpp_class} {var_name}{{}};"

    def emit_extra_setup(self, var_name: str) -> list[str]:
        out: list[str] = []
        scene = "manta::geom::Vec3<manta::SceneFrame>"
        for d in self._disturbances:
            vx, vy, vz = d["velocity"]
            v_expr = f"{scene}{{{_f(vx)}, {_f(vy)}, {_f(vz)}}}"
            if d["kind"] == "incompressible":
                expr = (f"{self.cpp_class}::Disturbance::uniform_incompressible("
                        f"manta::Real({_f(d['density'])}), {v_expr})")
            elif d["kind"] == "gas":
                expr = (f"{self.cpp_class}::Disturbance::uniform_gas("
                        f"manta::Real({_f(d['R'])}), "
                        f"manta::Real({_f(d['temperature'])}), "
                        f"manta::Real({_f(d['pressure'])}), {v_expr})")
            else:
                raise RuntimeError(f"unknown disturbance kind {d['kind']!r}")
            out.append(f"{var_name}.add({expr}, manta::fields::PERSISTENT);")
        return out
