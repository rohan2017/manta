"""MagField — superposition of magnetic disturbances."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class MagField(FieldDescriptor):
    """Single concrete magnetic field aggregating dipole and uniform
    disturbances. State is the magnetic flux density vector B in Tesla.
    """

    cpp_class     = "manta::fields::MagField"
    cpp_header    = "manta/fields/mag_field.hpp"
    feature_macro = "MANTA_HAS_MAG_FIELD"

    def __init__(self) -> None:
        super().__init__()
        self._disturbances: list[dict] = []

    def add_uniform(self, b: tuple[float, float, float]) -> "MagField":
        self._disturbances.append({"kind": "uniform",
                                   "b":    tuple(float(x) for x in b)})
        return self

    def add_dipole(self,
                   moment: tuple[float, float, float],
                   origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "MagField":
        self._disturbances.append({
            "kind":   "dipole",
            "moment": tuple(float(x) for x in moment),
            "origin": tuple(float(x) for x in origin),
        })
        return self

    def emit_construction(self, var_name: str) -> str:
        return f"{self.cpp_class} {var_name}{{}};"

    def emit_extra_setup(self, var_name: str) -> list[str]:
        out: list[str] = []
        scene = "manta::geom::Vec3<manta::SceneFrame>"
        for d in self._disturbances:
            if d["kind"] == "uniform":
                bx, by, bz = d["b"]
                expr = (f"{self.cpp_class}::Disturbance::uniform("
                        f"{scene}{{{_f(bx)}, {_f(by)}, {_f(bz)}}})")
            elif d["kind"] == "dipole":
                ox, oy, oz = d["origin"]
                mx, my, mz = d["moment"]
                expr = (f"{self.cpp_class}::Disturbance::dipole("
                        f"{scene}{{{_f(ox)}, {_f(oy)}, {_f(oz)}}}, "
                        f"{scene}{{{_f(mx)}, {_f(my)}, {_f(mz)}}})")
            else:
                raise RuntimeError(f"unknown disturbance kind {d['kind']!r}")
            out.append(f"{var_name}.add({expr}, manta::fields::PERSISTENT);")
        return out
