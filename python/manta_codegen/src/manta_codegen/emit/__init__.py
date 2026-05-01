"""Top-level emit() entry point.

Outputs (all written under `out_dir`, intended to be committed to the user's repo):
    <name>.hpp                  Typed Craft subclass + accessors.
    <name>.cpp                  Constructor body + part wiring.
    <name>_config.h             Feature-test macros for registered fields.
    <name>_telemetry.hpp        Per-craft telemetry struct + JSON encode.
    <name>_main.cpp             Sim main with Zenoh I/O. (workflow="binary" only)
    <name>.cmake                CMake fragment to include from user's CMakeLists.

This module is intentionally a thin orchestrator; each emitter lives in its own file.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core import Craft, World
from .craft import emit_craft_hpp, emit_craft_cpp
from .config import emit_config_h
from .telemetry import emit_telemetry_hpp
from .main import emit_main_cpp
from .cmake import emit_cmake_fragment


def emit(world: World,
         out_dir: str | os.PathLike,
         workflow: str = "library") -> None:
    """Render `world` to a directory of C++/CMake artifacts.

    workflow="library": emits Craft type, telemetry, config.h, CMake fragment.
                        User provides their own main.cpp.
    workflow="binary":  also emits a sim main with Zenoh I/O wired through
                        `craft.bindings`.

    Existing files are overwritten. The output is intended to live in the user's
    project tree (committed to git), not in the build directory.
    """
    if workflow not in ("library", "binary"):
        raise ValueError(
            f"workflow must be 'library' or 'binary', got {workflow!r}")

    # Per-craft emitters take the primary craft; emit_main_cpp consumes the
    # World for dt / sim_rate_mult / initial state.
    if not world.crafts:
        raise RuntimeError("emit(): World has no crafts")
    craft = world.crafts[0].craft

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = craft.name
    files: dict[str, str] = {
        f"{name}.hpp":            emit_craft_hpp(craft),
        f"{name}.cpp":            emit_craft_cpp(craft),
        f"{name}_config.h":       emit_config_h(world),
        f"{name}.cmake":          emit_cmake_fragment(craft, workflow=workflow),
    }
    # Telemetry only applies to sim-side workflows; an estimator-side craft
    # doesn't publish its sensor outputs (it consumes them).
    if workflow in ("library", "binary"):
        files[f"{name}_telemetry.hpp"] = emit_telemetry_hpp(craft)
    if workflow == "binary":
        files[f"{name}_main.cpp"] = emit_main_cpp(craft, world=world)

    for filename, contents in files.items():
        (out / filename).write_text(contents)


__all__ = ["emit"]
