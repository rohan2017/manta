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

    if not world.crafts:
        raise RuntimeError("emit(): World has no crafts")

    # Identify unique Craft objects (by Python identity). A multi-craft
    # world that adds the same Craft instance multiple times shares a single
    # set of generated files and bindings; multiple distinct Craft objects
    # each get their own .hpp / .cpp / _telemetry / _config.
    seen: set[int] = set()
    unique_crafts: list[Craft] = []
    for entry in world.crafts:
        if id(entry.craft) not in seen:
            seen.add(id(entry.craft))
            unique_crafts.append(entry.craft)

    multi = len(unique_crafts) > 1
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    # Per-craft files — one set per unique Craft.
    for craft in unique_crafts:
        n = craft.name
        files[f"{n}.hpp"] = emit_craft_hpp(craft)
        files[f"{n}.cpp"] = emit_craft_cpp(craft)
        if workflow in ("library", "binary"):
            files[f"{n}_telemetry.hpp"] = emit_telemetry_hpp(craft)

    # World-level files (single-craft worlds preserve the legacy naming
    # where the cmake fragment / config.h are named after the craft).
    world_name = world.name
    files[f"{world_name}_config.h"] = emit_config_h(world)
    files[f"{world_name}.cmake"]    = emit_cmake_fragment(
        world, workflow=workflow, multi=multi)
    if workflow == "binary":
        files[f"{world_name}_main.cpp"] = emit_main_cpp(world)

    for filename, contents in files.items():
        (out / filename).write_text(contents)


__all__ = ["emit"]
