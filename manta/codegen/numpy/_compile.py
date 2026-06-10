"""Optional compilation: codegen a Module's CasADi functions to C, build
them with `cc`, and call them as `external`s instead of interpreting the
MX graph. Bit-identical; ~8x per call (the interpretation overhead
dominates for these small systems). Cached on disk by the generated-
source hash so repeated runs skip the compiler.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any

import casadi as ca

# `ca.external` wrappers keyed by generated-source hash, so a second
# runtime over the same Module skips both gcc AND the dlopen. Guarded by
# a lock: TargetNumpy(compile=True) may be called from several threads.
_EXTERNAL_CACHE: dict[str, dict[str, Any]] = {}
_EXTERNAL_CACHE_LOCK = threading.Lock()
# Above this many MX nodes, codegen + gcc on the (huge symbolic) functions
# costs more than it ever saves — e.g. an EKF that linearizes the full coupled
# world tick is ~7k nodes -> hundreds of thousands of lines of C. Skip those
# *before* codegen (which is itself slow for them) and stay interpreted.
_MAX_INSTR = 3000


def _compiled_functions(functions: dict[str, Any]) -> dict[str, Any]:
    """Return externals keyed the same as `functions`, or `functions`
    unchanged if it's too big to bother / no C compiler is available."""
    if sum(f.n_instructions() for f in functions.values()) > _MAX_INSTR:
        return functions
    tmp = tempfile.mkdtemp()
    try:
        cg = ca.CodeGenerator("mod.c", {"with_header": True})
        for f in functions.values():
            cg.add(f)
        cg.generate(tmp + os.sep)
        src = open(os.path.join(tmp, "mod.c")).read()
        key = hashlib.sha1(src.encode()).hexdigest()[:16]
        with _EXTERNAL_CACHE_LOCK:
            if key in _EXTERNAL_CACHE:
                return _EXTERNAL_CACHE[key]
            cache_dir = os.path.join(tempfile.gettempdir(), "manta_compiled")
            os.makedirs(cache_dir, exist_ok=True)
            so_path = os.path.join(cache_dir, f"mod_{key}.so")
            try:
                if not os.path.exists(so_path):
                    # -O1, not -O3: these are huge straight-line numerical
                    # functions (symbolic Jacobians / Joseph updates), where
                    # -O2/3 compile time is super-linear for ~no runtime
                    # gain. Cap the compiler and fall back to the
                    # interpreter if it's slow. Build to a private temp
                    # path, then atomically publish — a concurrent process
                    # never dlopens a half-written .so.
                    so_tmp = os.path.join(tmp, f"mod_{key}.so")
                    subprocess.run(
                        ["cc", "-O1", "-fPIC", "-shared",
                         os.path.join(tmp, "mod.c"), "-o", so_tmp],
                        check=True, capture_output=True, timeout=30)
                    os.replace(so_tmp, so_path)
                out = {k: ca.external(f.name(), so_path)
                       for k, f in functions.items()}
            except (OSError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                out = functions              # no/slow compiler — interpret
            _EXTERNAL_CACHE[key] = out
            return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
