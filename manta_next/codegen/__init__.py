"""C++ codegen for manta_next Crafts.

Public API:
  emit_cpp(craft, out_dir, class_name) — generate a self-contained C++
  static-library project that exposes the craft's predict, predict
  Jacobian, and per-Output measurement + measurement Jacobian as typed
  Eigen-shaped methods. Math kernels come from CasADi's built-in
  Function.generate() (flat C, CSE-optimized); the typed wrapper is a
  thin pack/unpack layer.

Architecture:
  manta_next/codegen/
    extract.py  — Craft → per-function ca.Function objects
    kernels.py  — ca.Function → flat C source
    wrapper.py  — typed Eigen-shaped C++ class
    cmake.py    — CMakeLists.txt fragment
    cpp.py      — top-level emit_cpp() orchestration
"""

def __getattr__(name):
    # Lazy attribute so individual sub-modules can be imported without
    # forcing the whole codegen pipeline through.
    if name == "emit_cpp":
        from .cpp import emit_cpp as _e
        return _e
    raise AttributeError(f"module 'manta_next.codegen' has no attribute {name!r}")


__all__ = ["emit_cpp"]
