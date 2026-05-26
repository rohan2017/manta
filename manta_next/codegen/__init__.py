"""C++ codegen internals.

The public entry point is `TargetCpp(cw, out_dir, class_name)` in
`manta_next.targets`; this package houses the pipeline pieces it
calls into:

  extract.py  — CompiledWorld → per-function ca.Function objects
  kernels.py  — ca.Function → flat C source
  wrapper.py  — typed Eigen-shaped C++ class
  cmake.py    — CMakeLists.txt fragment
  cpp.py      — `_emit_world_cpp()` orchestration (TargetCpp's body)

Math kernels come from CasADi's built-in `Function.generate()` (flat
C, CSE-optimized); the typed wrapper is a thin pack/unpack layer
over a typed `State` / `Inputs` struct.
"""
