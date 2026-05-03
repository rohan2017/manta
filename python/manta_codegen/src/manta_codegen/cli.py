"""`manta-codegen` CLI: load a craft Python module and emit C++/CMake artifacts.

Usage:
    manta-codegen path/to/craft.py [--out OUT_DIR] [--workflow library|binary]

The craft module must expose one of:
  * `make_config()` returning a `MantaConfig`  (preferred).
  * `MANTA_CONFIG` bound to a `MantaConfig`.

The codegen produces one C++ main per Target inside the config.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .emit import emit_config
from .manifest import MantaConfig


def _load_user_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("user_craft_spec", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_craft_spec"] = module
    spec.loader.exec_module(module)
    return module


def _resolve_config(module) -> MantaConfig:
    if hasattr(module, "make_config"):
        cfg = module.make_config()
    elif hasattr(module, "MANTA_CONFIG"):
        cfg = module.MANTA_CONFIG
    else:
        raise RuntimeError(
            "user module must define make_config() returning a MantaConfig "
            "or bind MANTA_CONFIG.")
    if not isinstance(cfg, MantaConfig):
        raise TypeError(
            f"resolved config must be a MantaConfig, got {type(cfg).__name__}.")
    return cfg


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
    cfg = _resolve_config(module)

    if args.out:
        base_out = Path(args.out).resolve()
    else:
        base_out = craft_py.parent / "generated"

    if len(cfg.targets) == 1:
        # Single-Target shape: write directly under base_out using the
        # target's name as the subfolder so existing build wiring
        # (`generated/<target>/`) keeps working without arg shuffling.
        target = cfg.targets[0]
        out_dir = base_out if base_out.name == target.name else base_out / target.name
        emit_config(cfg, base_out=out_dir.parent, workflow=args.workflow)
        print(f"manta-codegen: wrote {target.name} ({args.workflow}) → {out_dir}")
    else:
        emit_config(cfg, base_out=base_out, workflow=args.workflow)
        for t in cfg.targets:
            print(f"manta-codegen: wrote {t.name} ({args.workflow}) → {base_out / t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
