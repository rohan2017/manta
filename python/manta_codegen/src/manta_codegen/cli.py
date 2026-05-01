"""`manta-codegen` CLI: load a craft Python module and emit C++/CMake artifacts.

Usage:
    manta-codegen path/to/craft.py [--out OUT_DIR] [--workflow library|binary]

The craft module must expose either:
  * A top-level `WORLD` variable bound to a `manta_codegen.World`, OR
  * A factory function `make_world()` that returns a World.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .core import Craft, World
from .emit import emit


def _load_world(module_path: Path) -> World:
    spec = importlib.util.spec_from_file_location("user_craft_spec", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_craft_spec"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "WORLD"):
        obj = module.WORLD
    elif hasattr(module, "make_world"):
        obj = module.make_world()
    else:
        raise RuntimeError(
            f"{module_path} must define WORLD or make_world() returning a World.")

    if not isinstance(obj, World):
        raise TypeError(f"Loaded object must be a World (got {type(obj).__name__}).")
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manta-codegen")
    parser.add_argument("craft_py", help="Python file describing the craft.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <craft_py_dir>/generated/<craft_name>).")
    parser.add_argument("--workflow",
                        choices=("library", "binary"),
                        default="library")
    args = parser.parse_args(argv)

    craft_py = Path(args.craft_py).resolve()
    if not craft_py.exists():
        parser.error(f"file not found: {craft_py}")

    world = _load_world(craft_py)
    if not world.crafts:
        parser.error(f"{craft_py}: World has no crafts")
    primary_craft_name = world.crafts[0].craft.name

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        out_dir = craft_py.parent / "generated" / primary_craft_name

    emit(world, out_dir=out_dir, workflow=args.workflow)
    print(f"manta-codegen: wrote {primary_craft_name} ({args.workflow}) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
