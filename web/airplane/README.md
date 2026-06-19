# Airplane WASM bundle (homepage live demo)

The live flight on the manta homepage runs the airframe from
`examples/vehicles/airplane.py` compiled to WebAssembly by manta's
`TargetWasm` backend. `demo.js` is the custom Three.js renderer + control loop
(authored by hand, not generated); everything else here is generated.

## Regenerate

From the repo root, with the [Emscripten SDK](https://emscripten.org) active
(`source /path/to/emsdk/emsdk_env.sh`):

```sh
# 1. emit the C ABI + kernels + JS runtime + build.sh
.venv/bin/python - <<'PY'
from manta import Sim, TargetWasm
from examples.vehicles.airplane import build_world
TargetWasm(Sim(build_world()), "web/airplane", class_name="Airplane")
PY

# 2. compile to airplane.mjs + airplane.wasm
cd web/airplane && ./build.sh
```

## Files

| File | Source | Served? |
| --- | --- | --- |
| `demo.js` | hand-written renderer + control loop | yes |
| `airplane.js` | generated ES-module runtime (`Runtime` + `Sim`) | yes |
| `airplane.mjs` / `airplane.wasm` | `emcc` output | yes |
| `airplane.descriptor.json` | generated state/IO layout | yes (fetched by the runtime) |
| `build.sh` | generated build script | no (regeneration only) |
| `airplane_kernels.c/.h`, `airplane_abi.c` | generated build inputs | **git-ignored** — re-emitted by step 1 |

The control loop in `demo.js` mirrors `airplane.py` exactly (dt=0.002, 10
substeps/frame, the same servo gains and scripted circuit), so the in-browser
flight is numerically identical to `TargetNumpy`.
