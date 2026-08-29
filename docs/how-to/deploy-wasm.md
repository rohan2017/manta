# Run a model in the browser (WASM)

[`TargetWasm`][manta.TargetWasm] lowers any Module to a browser bundle: the
same CasADi flat-C kernels the C++ backend uses, behind a thin flat-double
**C ABI** exported to WebAssembly, plus a JSON descriptor, a JS runtime, and
an Emscripten build script.

```python
from manta import TargetWasm
TargetWasm(Sim(w), "out", class_name="Drone")
# → out/{drone_kernels.c/h, drone_abi.c, drone.descriptor.json,
#        drone.js, build.sh}
```

The math path is identical to [`TargetCpp`][manta.TargetCpp] — `densify` +
the shared kernel emitter — so the numbers match every other backend exactly.
WASM adds only the marshalling glue.

The generated `Filter` view owns logical model time just like
`TargetNumpy` and generated C++ filters. `predict(dt)` uses the held time and
advances it only after a successful kernel call; `predict(dt, {t})` explicitly
resynchronizes the clock to `t + dt`. `update(..., {t})` may override sample
time but never advances the clock. Both `dt` and explicit time are validated.

Use `filter.checkpoint()` and `filter.restore(checkpoint)` to move an owned
`{x, P, time, artifactId}` restart point. Restore validates artifact identity,
shape, finite values, covariance symmetry/positive-semidefiniteness, and time
before changing live state. `filter.reset()` restores model defaults and time
zero.

## Build

`build.sh` invokes [Emscripten](https://emscripten.org) (`emcc`) to compile
the kernels + C ABI into `drone.mjs` + `drone.wasm`:

```sh
cd out && ./build.sh        # needs the emsdk on PATH
```

## Use it

`drone.js` is an ES module with no manta-specific logic — the embedded
descriptor drives all marshalling. It mirrors the numpy `Sim` surface:

```js
import { load } from "./drone.js";

const sim = (await load()).sim();
sim.state[/* position z */ 2] = 5.0;
for (let i = 0; i < 200; i++) sim.step(0.005, { "t.throttle": 1.5 * 9.81 });

console.log(sim.slot("position"), sim.reading("gps.position"));
```

`Runtime.call(method, args)` is the generic layer beneath `Sim` — it packs a
`{slotName: Float64Array | number}` map into the WASM heap per the
descriptor's per-entry layout and unpacks every output slot. Any Module's
entry points (a filter's `predict`/`update`, an LQR's `control`) are reachable
through it.

## What gets emitted

| File | Role |
| --- | --- |
| `<base>_kernels.c/.h` | CasADi flat-C math (shared with the C++ backend) |
| `<base>_abi.c` | flat-double C ABI (`int <base>_<method>(const double* in, double* out)`), `EMSCRIPTEN_KEEPALIVE`-exported |
| `<base>.descriptor.json` | state/IO layout the JS marshals from |
| `<base>.js` | ES-module runtime (`Runtime` + `Sim`) |
| `build.sh` | `emcc` → `<base>.mjs` + `<base>.wasm` |

The C ABI compiles with a plain C compiler too (the `EMSCRIPTEN_KEEPALIVE`
guard is a no-op off Emscripten), so the ABI can be round-tripped natively.

## Source material

- Reference: [Targets](../reference/targets.md)
- Code: `manta/codegen/wasm/`
- Concepts: [Codegen and backends](../explanation/codegen.md)
