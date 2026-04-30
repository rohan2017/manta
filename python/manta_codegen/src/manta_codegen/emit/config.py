"""Emit <name>_config.h — feature-test macros for registered fields.

This file is force-included into every TU of the craft target via the CMake
fragment, so all TUs see a consistent macro state (no ODR risk from
`#ifdef`-conditional inline definitions).

Required-field assertions in part headers (e.g. `static_assert(MANTA_HAS_FOO,
"...")`) read from this file. Compile-time augmentations in part `update()`
bodies (e.g. `#ifdef MANTA_HAS_OCEAN_ATMOS_FIELD`) read from it too.
"""

from __future__ import annotations

from ..core import Craft
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD


def emit_config_h(craft: Craft) -> str:
    lines: list[str] = [
        GENERATED_BANNER,
        CPP_INCLUDE_GUARD,
        "",
        "// Feature-test macros for fields registered with this craft. Force-included",
        "// into every TU of the craft target via the generated CMake fragment.",
        "",
    ]
    if not craft.fields:
        lines.append("// (No fields registered for this craft.)")
        lines.append("")
        return "\n".join(lines)

    for f in craft.fields:
        if f.feature_macro:
            lines.append(f"#define {f.feature_macro} 1")
    lines.append("")
    return "\n".join(lines)
