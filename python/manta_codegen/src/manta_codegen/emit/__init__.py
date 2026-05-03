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


def emit_config(cfg, base_out, workflow: str = "library", flat_target=None) -> None:
    """Emit a MantaConfig to disk. One subdirectory per Target under `base_out`.

    `flat_target` is a backward-compat hook used by the CLI for single-Target
    configs: when set, codegen writes directly into `base_out / flat_target.name`
    using the existing single-world emit path.

    Phase A: only Targets with exactly one World are supported. Multi-world
    Targets and EKF-bearing Targets land in subsequent commits.
    """
    from ..core import World
    from ..manifest import MantaConfig

    if not isinstance(cfg, MantaConfig):
        raise TypeError(f"emit_config: expected MantaConfig, got {type(cfg).__name__}")

    base = Path(base_out)
    targets = cfg.targets

    # Phase-A guard: every Target must have exactly one World in `drives`,
    # and no other Driveables (EKF, etc.) yet.
    for t in targets:
        worlds = [d for d in t.drives if isinstance(d, World)]
        non_worlds = [d for d in t.drives if not isinstance(d, World)]
        if len(worlds) != 1 or non_worlds:
            raise NotImplementedError(
                f"emit_config: Target {t.name!r} has "
                f"{len(worlds)} worlds and {len(non_worlds)} other drives; "
                f"only single-World Targets are supported in this build "
                f"(multi-world + EKF support is in flight).")

    # Validate no cross-target connect/binding spans.
    target_for_world: dict[int, str] = {}
    for t in targets:
        for d in t.drives:
            if isinstance(d, World):
                target_for_world[id(d)] = t.name

    def world_target_name(w):
        try:
            return target_for_world[id(w)]
        except KeyError:
            return None

    def craft_target_name(c):
        w = getattr(c, "_world", None)
        return world_target_name(w) if w is not None else None

    for t in targets:
        for d in t.drives:
            if isinstance(d, World):
                for binding in d.bindings + []:
                    pass   # bindings are already world-local; OK.
                for conn in d.connections:
                    src_t = craft_target_name(conn.source.craft_ref)
                    snk_t = craft_target_name(conn.sink.craft_ref)
                    if src_t is None or snk_t is None:
                        raise RuntimeError(
                            f"emit_config: connect() endpoint not in any Target "
                            f"({conn.source.name} → {conn.sink.name})")
                    if src_t != snk_t:
                        raise RuntimeError(
                            f"emit_config: connect({conn.source.name} → {conn.sink.name}) "
                            f"crosses targets ({src_t!r} → {snk_t!r}); "
                            f"use publish/subscribe across binaries instead.")

    # Phase-A delegates each Target to the existing single-World emit.
    for t in targets:
        world = next(d for d in t.drives if isinstance(d, World))
        # Cadence is now Target-level; copy onto the world for the legacy
        # emitters that still read world.dt / world.sim_rate_mult.
        world.dt = t.dt
        world.sim_rate_mult = t.sim_rate_mult
        target_dir = base / t.name
        target_dir.mkdir(parents=True, exist_ok=True)
        emit(world, out_dir=target_dir, workflow=workflow)


__all__ = ["emit", "emit_config"]
