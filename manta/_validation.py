"""Small numeric validators shared by authoring and runtime boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any

import numpy as np


def require_finite(value: Any, *, name: str) -> Any:
    """Return ``value`` after rejecting any non-finite numeric component."""
    if isinstance(value, Number) or isinstance(value, np.ndarray):
        try:
            finite = np.isfinite(value)
        except TypeError:
            return value
        if not bool(np.all(finite)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            require_finite(item, name=name)
    elif isinstance(value, Mapping):
        for item in value.values():
            require_finite(item, name=name)
    return value


def require_positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    """Return a finite float that is positive (or non-negative)."""
    number = float(value)
    require_finite(number, name=name)
    valid = number >= 0.0 if allow_zero else number > 0.0
    if not valid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}, got {number!r}")
    return number
