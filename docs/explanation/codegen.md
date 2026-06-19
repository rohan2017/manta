# Codegen and backends

!!! note "Draft"
    This page is scaffolded. The outline below marks what it should cover.

A transform emits a typed `Module` — a state layout plus named CasADi
kernels plus typed entry points. A `Target*` lowers that Module to a
backend. Backends contain **no per-transform code**: each implements one
generic lowering of a Module.

## To cover

- **The Module IR** — what a Module is (state fields, kernels, entry
  points) and why it isn't directly callable.
- **`TargetNumpy`** — one `NumpyRuntime` engine, four thin views whose
  surface is *derived from the Module's shape* (sim / filter / regulator
  / recurrence).
- **`TargetCpp`** — a typed Eigen class over flat-C kernels + CMake; the
  ABI is generated from one record so the C++ surface mirrors numpy.
- **`TargetWasm`** — the *same* flat-C kernels behind a thin flat-double
  C ABI + a JS runtime + an Emscripten build, so a Module runs in the
  browser. Reuses the C++ backend's math path verbatim; adds only the
  marshalling glue.
- **`TargetJax`** — a jitted `lax.scan` rollout (flat crafts).
- **Identical loops across backends** — the property that makes
  "develop in Python, deploy in C++" work.

## Source material

- Reference: [Targets](../reference/targets.md)
- Code: `manta/codegen/target.py`, `manta/codegen/numpy/`,
  `manta/codegen/cpp/`, `manta/codegen/wasm/`, `manta/codegen/jax/`
- How-to: [Deploy a model to C++](../how-to/deploy-cpp.md),
  [Run a model in the browser (WASM)](../how-to/deploy-wasm.md)
