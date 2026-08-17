# How-to guides

Task-focused recipes. Unlike the [tutorials](../tutorials/index.md), these
assume you already know the basics and just want to get a specific thing
done.

- **[Deploy a model to C++](deploy-cpp.md)** — lower a `Sim`/`EKF`/`LQR`
  to an Eigen C++ library for embedded use.
- **[Run a model in the browser (WASM)](deploy-wasm.md)** — lower the same
  Module to a WebAssembly + JS bundle via Emscripten.
- **[Fit parameters from a log](fit-parameters.md)** — system-ID against
  recorded controls + measurements.
- **[Model a displacement hull](displacement-hull.md)** — calibrate a
  distributed, surface-piercing hydrostatic and drag model.
- **[Write a custom Part](custom-part.md)** — add a new part with its own
  state, inputs, and wrench contribution.
- **[Write a custom Field or Disturbance](custom-field.md)** — add a new
  physical field or a new source on an existing one.
