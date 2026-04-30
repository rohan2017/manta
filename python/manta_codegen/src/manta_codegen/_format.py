"""Tiny formatting helpers used by descriptors and emitters.

Kept at the package root so descriptor modules (which sit above `emit/`) can
import without circular dependencies.
"""

from __future__ import annotations


def cpp_float(x: float | int) -> str:
    """Format a Python number as a C++ float literal with the `f` suffix.

    Always emits a decimal point — integer zeros become "0.0f" rather than
    "0f" (which is not valid C++ syntax).
    """
    s = f"{float(x):.7g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"
