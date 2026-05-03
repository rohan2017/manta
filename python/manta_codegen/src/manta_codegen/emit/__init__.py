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


def emit_config(cfg, base_out, workflow: str = "library") -> None:
    """Emit a MantaConfig to disk. One subdirectory per Target under `base_out`.

    Supported Target shapes:
      * exactly one World in `drives` — emits the standard sim main.
      * exactly one EKF in `drives`   — emits an EKF-driven main wrapping
                                        the EKF's internal world.
      * one World + one EKF in `drives` — emits an in-process two-world
                                          main where sim outputs feed the
                                          EKF via cross-world connect().
    """
    from ..core import World
    from ..manifest import MantaConfig
    from ..estimation.ekf import EKF

    if not isinstance(cfg, MantaConfig):
        raise TypeError(f"emit_config: expected MantaConfig, got {type(cfg).__name__}")

    base = Path(base_out)
    targets = cfg.targets

    def _classify(t):
        """Return ('world', World) | ('ekf', EKF) | ('sim+ekf', (World, EKF))."""
        worlds = [d for d in t.drives if isinstance(d, World)]
        ekfs   = [d for d in t.drives if isinstance(d, EKF)]
        other  = [d for d in t.drives
                  if not isinstance(d, World) and not isinstance(d, EKF)]
        if other:
            raise NotImplementedError(
                f"emit_config: Target {t.name!r} contains {len(other)} unsupported "
                f"drive type(s) ({[type(d).__name__ for d in other]}). "
                f"Supported: World, EKF.")
        if len(worlds) == 1 and not ekfs:
            return "world", worlds[0]
        if len(ekfs) == 1 and not worlds:
            return "ekf", ekfs[0]
        if len(worlds) == 1 and len(ekfs) == 1:
            return "sim+ekf", (worlds[0], ekfs[0])
        raise NotImplementedError(
            f"emit_config: Target {t.name!r} has {len(worlds)} worlds and "
            f"{len(ekfs)} EKFs; supported shapes are 1×World, 1×EKF, "
            f"or 1×World+1×EKF (multiple sim worlds or multiple EKFs are "
            f"not yet supported).")

    classified = [(t, *_classify(t)) for t in targets]

    # Validate no cross-target connect/binding spans. Sim+EKF targets cover
    # both their sim world and the EKF's internal est world.
    target_for_world: dict[int, str] = {}
    for t, kind, drv in classified:
        if kind == "world":
            target_for_world[id(drv)] = t.name
        elif kind == "ekf":
            target_for_world[id(drv.world)] = t.name
        else:  # "sim+ekf"
            sim_w, ekf = drv
            target_for_world[id(sim_w)]    = t.name
            target_for_world[id(ekf.world)] = t.name

    def world_target_name(w):
        try:
            return target_for_world[id(w)]
        except KeyError:
            return None

    def craft_target_name(c):
        w = getattr(c, "_world", None)
        return world_target_name(w) if w is not None else None

    for t, kind, drv in classified:
        if kind == "world":
            worlds = [drv]
        elif kind == "ekf":
            worlds = [drv.world]
        else:
            sim_w, ekf = drv
            worlds = [sim_w, ekf.world]
        for w in worlds:
            for conn in w.connections:
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

    for t, kind, drv in classified:
        target_dir = base / t.name
        target_dir.mkdir(parents=True, exist_ok=True)
        if kind == "world":
            world = drv
            world.dt = t.dt
            world.sim_rate_mult = t.sim_rate_mult
            emit(world, out_dir=target_dir, workflow=workflow)
        elif kind == "ekf":
            ekf = drv
            world = ekf.world
            world.dt = t.dt
            world.sim_rate_mult = t.sim_rate_mult
            _emit_ekf_target(t, ekf, world, target_dir, workflow)
        else:  # "sim+ekf"
            sim_w, ekf = drv
            sim_w.dt = t.dt
            sim_w.sim_rate_mult = t.sim_rate_mult
            ekf.world.dt = t.dt
            ekf.world.sim_rate_mult = t.sim_rate_mult
            _emit_sim_plus_ekf_target(t, sim_w, ekf, target_dir, workflow)


def _emit_ekf_target(target, ekf, world, out_dir: Path,
                     workflow: str) -> None:
    """Emit per-craft + main for an EKF-driven Target. Reuses the standard
    craft/telemetry/config/cmake emitters; substitutes the main with the
    EKF-driven variant. workflow="library" omits the main entirely."""
    from .ekf_main import emit_ekf_main_cpp

    if workflow not in ("library", "binary"):
        raise ValueError(
            f"workflow must be 'library' or 'binary', got {workflow!r}")
    if not world.crafts:
        raise RuntimeError("emit_config (EKF target): wrapped world has no crafts")

    seen: set[int] = set()
    unique_crafts: list[Craft] = []
    for entry in world.crafts:
        if id(entry.craft) not in seen:
            seen.add(id(entry.craft))
            unique_crafts.append(entry.craft)

    multi = len(unique_crafts) > 1
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for craft in unique_crafts:
        n = craft.name
        files[f"{n}.hpp"] = emit_craft_hpp(craft)
        files[f"{n}.cpp"] = emit_craft_cpp(craft)
        files[f"{n}_telemetry.hpp"] = emit_telemetry_hpp(craft)

    world_name = world.name
    files[f"{world_name}_config.h"] = emit_config_h(world)
    files[f"{world_name}.cmake"]    = emit_cmake_fragment(
        world, workflow=workflow, multi=multi)
    if workflow == "binary":
        files[f"{world_name}_main.cpp"] = emit_ekf_main_cpp(target, ekf)

    for filename, contents in files.items():
        (out_dir / filename).write_text(contents)


def _emit_sim_plus_ekf_target(target, sim_world: World, ekf, out_dir: Path,
                              workflow: str) -> None:
    """Emit per-craft + main for a Target whose drives = [sim_world, ekf].

    File layout:
      * One per-craft fileset (.hpp/.cpp/_telemetry.hpp) per unique Craft
        across BOTH worlds. The sim and est crafts have different names so
        they end up in different .hpp files.
      * One world-level config.h + cmake fragment, named after the target.
      * The cmake fragment's `add_executable(...)` target is also named
        after the target.
      * One unified _main.cpp from emit_sim_plus_ekf_main_cpp().
    """
    from .sim_ekf_main import emit_sim_plus_ekf_main_cpp

    if workflow not in ("library", "binary"):
        raise ValueError(
            f"workflow must be 'library' or 'binary', got {workflow!r}")
    if not sim_world.crafts:
        raise RuntimeError("emit_config (sim+ekf): sim world has no crafts")
    if not ekf.world.crafts:
        raise RuntimeError("emit_config (sim+ekf): EKF wrapped world has no crafts")

    # Union the per-craft fileset across both worlds — no name collisions
    # are possible because sim and est crafts are required to have
    # distinct names (otherwise their generated .hpp/.cpp would overwrite
    # each other).
    seen: set[int] = set()
    unique_crafts: list[Craft] = []
    seen_names: set[str] = set()
    for entry in list(sim_world.crafts) + list(ekf.world.crafts):
        if id(entry.craft) in seen:
            continue
        seen.add(id(entry.craft))
        if entry.craft.name in seen_names:
            raise RuntimeError(
                f"emit_config (sim+ekf): two distinct crafts share the name "
                f"{entry.craft.name!r}; rename one so generated files don't "
                f"collide.")
        seen_names.add(entry.craft.name)
        unique_crafts.append(entry.craft)

    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for craft in unique_crafts:
        n = craft.name
        files[f"{n}.hpp"] = emit_craft_hpp(craft)
        files[f"{n}.cpp"] = emit_craft_cpp(craft)
        files[f"{n}_telemetry.hpp"] = emit_telemetry_hpp(craft)

    # Target-level files use the target name (not a world name) so the
    # cmake target / config.h are addressable from the user's CMakeLists.
    # The two worlds share the same feature-test macro set; we union their
    # registered fields by passing a synthetic world to emit_config_h.
    union = _union_world_for_config(sim_world, ekf.world, target.name)
    files[f"{target.name}_config.h"]   = emit_config_h(union)
    files[f"{target.name}.cmake"]      = emit_cmake_fragment(
        union, workflow=workflow, multi=len(unique_crafts) > 1)
    if workflow == "binary":
        files[f"{target.name}_main.cpp"] = emit_sim_plus_ekf_main_cpp(
            target, sim_world, ekf)

    for filename, contents in files.items():
        (out_dir / filename).write_text(contents)


def _union_world_for_config(a: World, b: World, name: str) -> World:
    """Build a synthetic World whose `name`, `crafts`, `fields`, `planets`,
    and `tethers` are the union of `a` and `b`. Used to drive
    emit_config_h / emit_cmake_fragment for sim+ekf targets — those
    emitters only read these attributes, never tick the world."""
    u = World(name)
    seen_crafts: set[int] = set()
    for entry in list(a.crafts) + list(b.crafts):
        if id(entry.craft) in seen_crafts:
            continue
        seen_crafts.add(id(entry.craft))
        u.crafts.append(entry)
    seen_fields: set[int] = set()
    for f in list(a.fields) + list(b.fields):
        if id(f) in seen_fields:
            continue
        seen_fields.add(id(f))
        u.fields.append(f)
    u.planets.extend(a.planets)
    u.planets.extend(b.planets)
    u.tethers.extend(a.tethers)
    u.tethers.extend(b.tethers)
    return u


__all__ = ["emit", "emit_config"]
