"""`manta-codegen` CLI: load a craft Python module and emit C++/CMake artifacts.

Usage:
    manta-codegen path/to/craft.py [--out OUT_DIR] [--workflow library|binary]

The craft module must expose one of (in priority order):
  * `make_config()` returning a `MantaConfig`     (preferred, multi-target).
  * `MANTA_CONFIG` bound to a `MantaConfig`.
  * `make_world()` returning a `World`            (legacy, single-target).
  * `WORLD` bound to a `World`.

`make_config` is the multi-target shape — produces one C++ main per Target.
`make_world` is the original single-target shortcut, kept for backward
compat; the codegen wraps the returned World in a one-Target MantaConfig.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .core import World
from .emit import emit, emit_config
from .manifest import MantaConfig, Target


def _load_user_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("user_craft_spec", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_craft_spec"] = module
    spec.loader.exec_module(module)
    return module


def _resolve_config(module, workflow: str) -> MantaConfig:
    """Pull a MantaConfig out of the user module. Wraps a legacy-style
    `make_world()` / `WORLD` into a single-Target config for codegen."""
    if hasattr(module, "make_config"):
        cfg = module.make_config()
    elif hasattr(module, "MANTA_CONFIG"):
        cfg = module.MANTA_CONFIG
    elif hasattr(module, "make_world"):
        world = module.make_world()
        cfg = _wrap_world(world, workflow)
    elif hasattr(module, "WORLD"):
        cfg = _wrap_world(module.WORLD, workflow)
    else:
        raise RuntimeError(
            "user module must define make_config() / MANTA_CONFIG / "
            "make_world() / WORLD.")
    if not isinstance(cfg, MantaConfig):
        raise TypeError(
            f"resolved config must be a MantaConfig, got {type(cfg).__name__}.")
    return cfg


def _wrap_world(world: World, workflow: str) -> MantaConfig:
    if not isinstance(world, World):
        raise TypeError(f"make_world / WORLD must yield a World, got {type(world).__name__}.")
    if not world.crafts:
        raise RuntimeError("make_world: World has no crafts.")
    target_name = world.crafts[0].craft.name
    return MantaConfig(targets=[Target(
        name=target_name,
        drives=[world],
        dt=getattr(world, "dt", 0.001),
        sim_rate_mult=getattr(world, "sim_rate_mult", 1.0),
    )])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manta-codegen")
    parser.add_argument("craft_py", help="Python file describing the system.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <craft_py_dir>/generated).")
    parser.add_argument("--workflow",
                        choices=("library", "binary"),
                        default="library")
    args = parser.parse_args(argv)

    craft_py = Path(args.craft_py).resolve()
    if not craft_py.exists():
        parser.error(f"file not found: {craft_py}")

    module = _load_user_module(craft_py)
    cfg = _resolve_config(module, args.workflow)

    if args.out:
        base_out = Path(args.out).resolve()
    else:
        base_out = craft_py.parent / "generated"

    if len(cfg.targets) == 1:
        # Legacy-shape compatibility: when there's a single Target, write
        # directly under base_out using the target's name as the subfolder
        # so existing build wiring (`generated/<craft>/`) keeps working.
        target = cfg.targets[0]
        out_dir = base_out if base_out.name == target.name else base_out / target.name
        emit_config(cfg, base_out=out_dir.parent, workflow=args.workflow,
                    flat_target=target)
        print(f"manta-codegen: wrote {target.name} ({args.workflow}) → {out_dir}")
    else:
        # Multi-target: one subdirectory per Target.
        emit_config(cfg, base_out=base_out, workflow=args.workflow)
        for t in cfg.targets:
            print(f"manta-codegen: wrote {t.name} ({args.workflow}) → {base_out / t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
