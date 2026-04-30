"""`manta-codegen` CLI: load a craft Python module and emit C++/CMake artifacts.

Usage:
    manta-codegen path/to/craft.py [--out OUT_DIR] [--workflow library|binary]

The craft module must expose either:
  * A top-level `CRAFT` variable bound to a `manta_codegen.Craft`, OR
  * A factory function named `make_craft()` that returns a Craft.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .core import Craft
from .emit import emit


def _load_craft(module_path: Path) -> Craft:
    spec = importlib.util.spec_from_file_location("user_craft_spec", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_craft_spec"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "CRAFT"):
        craft = module.CRAFT
    elif hasattr(module, "make_craft"):
        craft = module.make_craft()
    else:
        raise RuntimeError(
            f"{module_path} must define `CRAFT` or `make_craft()` returning a Craft.")
    if not isinstance(craft, Craft):
        raise TypeError(f"Loaded object is not a Craft (got {type(craft).__name__}).")
    return craft


def _parse_topics(s: str) -> dict[str, str]:
    """Parse 'name1=topic1,name2=topic2' into {name1: topic1, ...}."""
    out: dict[str, str] = {}
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"--topics entry {entry!r} must be name=topic")
        name, topic = entry.split("=", 1)
        out[name.strip()] = topic.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manta-codegen")
    parser.add_argument("craft_py", help="Python file describing the craft.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <craft_py_dir>/generated/<craft_name>).")
    parser.add_argument("--workflow",
                        choices=("library", "binary", "real_data"),
                        default="library")
    parser.add_argument("--topics", type=_parse_topics, default=None,
                        help="For --workflow real_data: comma-separated "
                             "part=topic pairs, e.g. "
                             "'imu=robot/imu/cooked,dvl=robot/dvl/vel'. "
                             "Each part must exist in the craft and have a "
                             "measurement decoder (sensor parts do; non-sensor "
                             "parts will fail at codegen time).")
    args = parser.parse_args(argv)

    craft_py = Path(args.craft_py).resolve()
    if not craft_py.exists():
        parser.error(f"file not found: {craft_py}")

    if args.workflow == "real_data" and not args.topics:
        parser.error("--workflow real_data requires --topics")

    craft = _load_craft(craft_py)

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        out_dir = craft_py.parent / "generated" / craft.name

    emit(craft, out_dir=out_dir, workflow=args.workflow, topics=args.topics)
    print(f"manta-codegen: wrote {craft.name} ({args.workflow}) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
