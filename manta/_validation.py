"""Small numeric validators shared by authoring and runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np


def finite_real(value: Any, name: str = "value") -> float:
    """Coerce a real scalar to ``float``, rejecting bool and non-finite values.

    Manta deliberately carries its own copy of this boundary rather than
    importing it: the modeling library depends on numpy and CasADi only.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a real number, not {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def require_finite(value: Any, *, name: str) -> Any:
    """Return a real numeric value/container after strict finite checking.

    Strings and booleans are deliberately not coerced.  This is the common
    numeric boundary used before values enter symbolic model construction;
    accepting ``True`` as one kilogram or ``"1.0"`` as a gain hides the
    mistake until much later in the pipeline.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real numeric data, not bool")
    if isinstance(value, Real):
        finite_real(value, name)
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf":
            raise TypeError(
                f"{name} must contain real numeric data, got dtype {value.dtype}")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            require_finite(item, name=name)
    elif isinstance(value, Mapping):
        for item in value.values():
            require_finite(item, name=name)
    else:
        raise TypeError(
            f"{name} must be real numeric data, got {type(value).__name__}")
    return value


def require_positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    """Return a finite float that is positive (or non-negative)."""
    require_finite(value, name=name)
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    number = finite_real(value, name)
    valid = number >= 0.0 if allow_zero else number > 0.0
    if not valid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}, got {number!r}")
    return number
