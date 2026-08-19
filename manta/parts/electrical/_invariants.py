"""Electrical model invariants and shared symbolic limit expressions."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import casadi as ca
import numpy as np

from manta._validation import finite_real as _finite_real


def finite_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real SI scalar, got {value!r}")
    return _finite_real(value, name)


def positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    number = finite_scalar(value, name=name)
    if number < 0.0 or (number == 0.0 and not allow_zero):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}, got {number!r}")
    return number


def positive_or_inf(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real SI scalar, got {value!r}")
    number = float(value)
    if math.isnan(number) or number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def unit_interval(value: Any, *, name: str) -> float:
    number = finite_scalar(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number!r}")
    return number


def bounded_positive(value: ca.MX, limit: float) -> ca.MX:
    positive_value = ca.fmax(value, 0.0)
    return positive_value if math.isinf(limit) else ca.fmin(positive_value, limit)


def c1_gate(value: ca.MX, low: float, high: float) -> ca.MX:
    if high == low:
        return ca.if_else(value >= high, 1.0, 0.0)
    x = ca.fmin(ca.fmax((value - low) / (high - low), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)
