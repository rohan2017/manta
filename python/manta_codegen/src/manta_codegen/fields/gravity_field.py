"""GravityField — superposition of gravity disturbances."""

from __future__ import annotations

from .._format import cpp_float as _f
from ..core import FieldDescriptor


class GravityField(FieldDescriptor):
    """Single concrete gravity field that aggregates a list of additive
    disturbances. Each disturbance is one of:
      * uniform(g)              — constant g vector everywhere
      * point_mass(mu, origin)  — inverse-square gravity from a point mass
      * point_mass_j2(mu, origin, j2, eq_radius, polar_axis)
                                — point mass + J2 oblateness perturbation

    The codegen emits the C++ `manta::fields::GravityField` and follows it
    with one `add(Disturbance::..., PERSISTENT)` call per disturbance.

    Convenience: passing a `g` tuple to the constructor is equivalent to
    a single uniform disturbance — matches the old ergonomic shape so most
    examples need only mechanical changes.
    """

    cpp_class     = "manta::fields::GravityField"
    cpp_header    = "manta/fields/gravity_field.hpp"
    feature_macro = "MANTA_HAS_GRAVITY_FIELD"

    def __init__(self,
                 g: tuple[float, float, float] | None = None) -> None:
        super().__init__()
        self._disturbances: list[dict] = []
        if g is not None:
            self.add_uniform(g)

    def add_uniform(self, g: tuple[float, float, float]) -> "GravityField":
        self._disturbances.append({"kind": "uniform",
                                   "g": tuple(float(x) for x in g)})
        return self

    def add_point_mass(self,
                       mu: float,
                       origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                       j2: float | None = None,
                       eq_radius: float | None = None,
                       polar_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> "GravityField":
        if j2 is not None and eq_radius is None:
            raise ValueError("GravityField.add_point_mass: j2 requires eq_radius")
        self._disturbances.append({
            "kind": "point_mass_j2" if j2 is not None else "point_mass",
            "mu":         float(mu),
            "origin":     tuple(float(x) for x in origin),
            "j2":         None if j2 is None else float(j2),
            "eq_radius":  None if eq_radius is None else float(eq_radius),
            "polar_axis": tuple(float(x) for x in polar_axis),
        })
        return self

    def emit_construction(self, var_name: str) -> str:
        return f"{self.cpp_class} {var_name}{{}};"

    def emit_extra_setup(self, var_name: str) -> list[str]:
        out: list[str] = []
        scene = "manta::geom::Vec3<manta::SceneFrame>"
        for d in self._disturbances:
            if d["kind"] == "uniform":
                gx, gy, gz = d["g"]
                expr = (f"{self.cpp_class}::Disturbance::uniform("
                        f"{scene}{{{_f(gx)}, {_f(gy)}, {_f(gz)}}})")
            elif d["kind"] == "point_mass":
                ox, oy, oz = d["origin"]
                expr = (f"{self.cpp_class}::Disturbance::point_mass("
                        f"{scene}{{{_f(ox)}, {_f(oy)}, {_f(oz)}}}, "
                        f"manta::MFloat({_f(d['mu'])}))")
            elif d["kind"] == "point_mass_j2":
                ox, oy, oz = d["origin"]
                ax, ay, az = d["polar_axis"]
                expr = (f"{self.cpp_class}::Disturbance::point_mass_j2("
                        f"{scene}{{{_f(ox)}, {_f(oy)}, {_f(oz)}}}, "
                        f"manta::MFloat({_f(d['mu'])}), "
                        f"manta::MFloat({_f(d['j2'])}), "
                        f"manta::MFloat({_f(d['eq_radius'])}), "
                        f"{scene}{{{_f(ax)}, {_f(ay)}, {_f(az)}}})")
            else:
                raise RuntimeError(f"unknown disturbance kind {d['kind']!r}")
            out.append(f"{var_name}.add({expr}, manta::fields::PERSISTENT);")
        return out
