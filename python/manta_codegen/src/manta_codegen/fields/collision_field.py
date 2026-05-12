"""CollisionField — superposition of contact disturbances.

Two stock disturbance kinds in v1:
  * single_sphere(center, radius, k, d, mu_s, mu_k)  — a fixed solid
    ball at scene-frame `center`. The reaction wrench is bilateral —
    any Collider part on the field gets pushed away on overlap.
  * infinite_plane(point, normal, k, d, mu_s, mu_k) — an infinite
    half-space. Use this for ground planes.

Multi-shape Colliders (the per-part disturbance) carry USER_TAG (no
replication) — the geometry doesn't fit the 96-byte Params budget.
"""

from __future__ import annotations

from .._format import cpp_float as _f, cpp_mfloat as _mf
from ..core import FieldDescriptor


class CollisionField(FieldDescriptor):
    cpp_class     = "manta::fields::CollisionField"
    cpp_header    = "manta/fields/collision_field.hpp"
    feature_macro = "MANTA_HAS_COLLISION_FIELD"

    def __init__(self) -> None:
        super().__init__()
        self._disturbances: list[dict] = []

    def add_single_sphere(self,
                          center: tuple[float, float, float],
                          radius: float,
                          k: float = 1.0e6,
                          d: float = 5.0e3,
                          mu_static: float = 0.7,
                          mu_kinetic: float = 0.5) -> "CollisionField":
        self._disturbances.append({
            "kind": "single_sphere",
            "center": tuple(float(x) for x in center),
            "radius": float(radius),
            "k": float(k), "d": float(d),
            "mu_s": float(mu_static), "mu_k": float(mu_kinetic),
        })
        return self

    def add_infinite_plane(self,
                           point: tuple[float, float, float] = (0.0, 0.0, 0.0),
                           normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
                           k: float = 1.0e6,
                           d: float = 5.0e3,
                           mu_static: float = 0.7,
                           mu_kinetic: float = 0.5) -> "CollisionField":
        self._disturbances.append({
            "kind": "infinite_plane",
            "point": tuple(float(x) for x in point),
            "normal": tuple(float(x) for x in normal),
            "k": float(k), "d": float(d),
            "mu_s": float(mu_static), "mu_k": float(mu_kinetic),
        })
        return self

    def emit_construction(self, var_name: str) -> str:
        return f"{self.cpp_class} {var_name}{{}};"

    def emit_extra_setup(self, var_name: str) -> list[str]:
        out: list[str] = []
        scene = "manta::geom::Vec3<manta::SceneFrame>"
        for d in self._disturbances:
            if d["kind"] == "single_sphere":
                cx, cy, cz = d["center"]
                expr = (f"{self.cpp_class}::Disturbance::single_sphere("
                        f"{scene}{{{_f(cx)}, {_f(cy)}, {_f(cz)}}}, "
                        f"{_mf(d['radius'])}, {_mf(d['k'])}, {_mf(d['d'])}, "
                        f"{_mf(d['mu_s'])}, {_mf(d['mu_k'])})")
            elif d["kind"] == "infinite_plane":
                px, py, pz = d["point"]
                nx, ny, nz = d["normal"]
                expr = (f"{self.cpp_class}::Disturbance::infinite_plane("
                        f"{scene}{{{_f(px)}, {_f(py)}, {_f(pz)}}}, "
                        f"{scene}{{{_f(nx)}, {_f(ny)}, {_f(nz)}}}, "
                        f"{_mf(d['k'])}, {_mf(d['d'])}, "
                        f"{_mf(d['mu_s'])}, {_mf(d['mu_k'])})")
            else:
                raise RuntimeError(f"unknown disturbance kind {d['kind']!r}")
            out.append(f"{var_name}.add({expr}, manta::fields::PERSISTENT);")
        return out
