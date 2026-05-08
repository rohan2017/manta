"""Tiny formatting helpers used by descriptors and emitters.

Kept at the package root so descriptor modules (which sit above `emit/`) can
import without circular dependencies.
"""

from __future__ import annotations


def cpp_float(x: float | int) -> str:
    """Format a Python number as a C++ `float` literal (with `f` suffix).

    Use this for values going into a hard-coded `float` field — sensor
    σ, white-noise stddev, anything that's `float` regardless of the
    library's MFloat configuration.

    For values whose target type is `MFloat` (which can be either
    float or double depending on `MANTA_DOUBLE_PRECISION`), use
    `cpp_mfloat()` instead — it emits a suffix-less literal that the
    C++ compiler picks the correct width for.

    Always emits a decimal point — integer zeros become "0.0f" rather
    than "0f" (which is not valid C++ syntax).
    """
    s = f"{float(x):.7g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def cpp_mfloat(x: float | int) -> str:
    """Format a Python number as a `manta::MFloat(<literal>)` expression.

    The inner literal has NO `f` suffix — the compiler picks float vs.
    double based on the active `MANTA_DOUBLE_PRECISION` setting. This
    avoids the silent float→double precision loss that an `f`-suffixed
    literal would cause when the library is built in double precision.

    Returns the full `manta::MFloat(...)` expression (callers don't
    need to wrap it themselves).
    """
    s = f"{float(x):.17g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return f"manta::MFloat({s})"
