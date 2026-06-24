"""Generate the WASM bundles for the homepage example carousel.

Emits the C ABI + kernels + JS runtime + descriptor for each demo vehicle
into its web/ subdirectory, then (if emcc is on PATH) compiles each to
`<name>.mjs` + `<name>.wasm`. Run from the repo root:

    source ~/emsdk/emsdk_env.sh        # put emcc on PATH
    .venv/bin/python web/build_demos.py

The quad model is web/demos.py:build_quad; the airplane is the real
examples/vehicles/airplane.py:build_world (current Cessna 172).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

from manta import Sim, TargetWasm  # noqa: E402


def emit(build_world, out: str, class_name: str) -> None:
    world = build_world()
    target = ROOT / out
    target.mkdir(parents=True, exist_ok=True)
    TargetWasm(Sim(world), str(target), class_name=class_name)
    print(f"  emitted {out}/  ({class_name})")


def compile_bundle(out: str) -> bool:
    build = ROOT / out / "build.sh"
    if not shutil.which("emcc"):
        print(f"  emcc not found — skipping compile of {out} "
              f"(run `{build}` once Emscripten is on PATH)")
        return False
    subprocess.run(["sh", str(build)], check=True, cwd=str(ROOT / out))
    return True


def main() -> None:
    from demos import build_quad
    from examples.vehicles.airplane import build_world as build_airplane

    print("generating WASM bundles…")
    emit(build_quad, "web/quad", "Quad")
    emit(build_airplane, "web/airplane", "Airplane")

    print("compiling…")
    for out in ("web/quad", "web/airplane"):
        compile_bundle(out)
    print("done.")


if __name__ == "__main__":
    main()
