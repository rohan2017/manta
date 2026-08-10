"""C++ backend type registry — maps `Manifold.kind` to Eigen types.

This is the ONLY place in the C++ backend that knows about specific
manifold kinds. To add a new manifold (e.g., SE3, R6, a 2D rotation),
add an entry here keyed on the manifold's `kind` string — no churn
elsewhere in the C++ codegen.

Rust / JAX / TF backends maintain their own analogous registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ...ir.manifold import Manifold


@dataclass(frozen=True)
class CppType:
    """How a manifold kind maps onto C++ source.

    Attrs:
      decl    — C++ type name (e.g., ``"double"``, ``"Eigen::Vector3d"``).
      zero    — C++ default-value expression for an uninitialized slot
                (e.g., ``"0.0"``, ``"Eigen::Vector3d::Zero()"``).
      literal — function that turns a Python value into a C++ initializer
                expression matching `decl` (e.g., a Python float into
                ``"0.5"``, a 3-tuple into
                ``"Eigen::Vector3d(0.5, 1.0, 2.0)"``).
    """
    decl:    str
    zero:    str
    literal: Callable[[object], str]


def _scalar_literal(v) -> str:
    return repr(float(v))


def _vec3_literal(v) -> str:
    a, b, c = (float(x) for x in v)
    return f"Eigen::Vector3d({a!r}, {b!r}, {c!r})"


def _quat_literal(v) -> str:
    w, x, y, z = (float(c) for c in v)
    return f"Eigen::Vector4d({w!r}, {x!r}, {y!r}, {z!r})"


_REGISTRY: dict[str, CppType] = {
    "scalar": CppType(
        decl="double",
        zero="0.0",
        literal=_scalar_literal,
    ),
    "vec": CppType(
        decl="Eigen::Vector3d",
        zero="Eigen::Vector3d::Zero()",
        literal=_vec3_literal,
    ),
    "quat": CppType(
        decl="Eigen::Vector4d",
        zero="Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)",
        literal=_quat_literal,
    ),
}


def _vecn_literal(n):
    def render(value):
        import numpy as _np
        vals = ", ".join(f"{float(x)}" for x in
                         _np.asarray(value, dtype=float).reshape(-1))
        return (f"(Eigen::Matrix<double, {n}, 1>() << {vals}).finished()")
    return render


def cpp_type_for(manifold: Manifold) -> CppType:
    """Look up the C++ mapping for a Manifold instance, keyed on
    `manifold.kind`. Raises NotImplementedError if the kind is not
    registered — at which point the fix is to add an entry above.

    The parameterized-dimension kind ("vecn") is CONSTRUCTED here from
    `storage_shape` rather than enumerated in the registry — one kind,
    every n (the same dimension-is-data principle as RnManifold
    itself)."""
    if manifold.kind == "vecn":
        n = int(manifold.storage_shape[0])
        return CppType(
            decl=f"Eigen::Matrix<double, {n}, 1>",
            zero=f"Eigen::Matrix<double, {n}, 1>::Zero()",
            literal=_vecn_literal(n),
        )
    try:
        return _REGISTRY[manifold.kind]
    except KeyError:
        raise NotImplementedError(
            f"cpp backend: no type mapping for manifold kind "
            f"{manifold.kind!r} (manifold={manifold!r}). Add an entry "
            f"to manta/codegen/cpp/types.py::_REGISTRY.")
