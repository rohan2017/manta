"""Explicit compilation of CasADi kernels to cached C externals.

``TargetNumpy(..., compile=True)`` is a contract, not a hint: every requested
function is returned as a CasADi ``External`` or this module raises an
actionable :class:`CompilationError`. Silently reverting to an interpreted MX
graph makes performance tests lie and can violate a controller's runtime
budget.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Literal

import casadi as ca

# `ca.external` wrappers keyed by generated source plus caller mapping, so a
# second runtime skips both gcc and dlopen without accidentally returning the
# first caller's dictionary keys. Guarded because compilation can be concurrent.
_EXTERNAL_CACHE: dict[str, dict[str, Any]] = {}
_EXTERNAL_CACHE_LOCK = threading.Lock()
# Above this many MX nodes, codegen + gcc on the (huge symbolic) functions
# costs more than it ever saves — e.g. an EKF that linearizes the full coupled
# world tick is ~7k nodes -> hundreds of thousands of lines of C. Skip those
# *before* codegen (which is itself slow for them) and stay interpreted.
_MAX_INSTR = 3000


class CompilationError(RuntimeError):
    """A requested native kernel could not be produced or loaded."""


def compile_functions(functions: dict[str, Any], *,
                      max_instructions: int | None = _MAX_INSTR,
                      optimization: Literal["balanced", "runtime"] =
                      "balanced") -> dict[str, Any]:
    """Compile an owned set of CasADi functions to cached C externals.

    This is the escape hatch for controller and estimator prototypes whose
    kernels have not yet been wrapped in a typed :class:`~manta.ir.Module`.
    Normal applications should prefer ``TargetNumpy(module, compile=True)``.

    The returned mapping has the same keys as ``functions``. Compilation is
    an explicit contract: an unavailable compiler, an oversized kernel, a
    compiler timeout, or a load failure raises :class:`CompilationError`.
    ``max_instructions=None`` opts out of the conservative generic size gate;
    use that only for deliberately looped/batched control kernels.
    """
    if not functions:
        raise ValueError("compile_functions requires at least one function")
    if optimization not in ("balanced", "runtime"):
        raise ValueError(
            "optimization must be 'balanced' or 'runtime'")
    return _compiled_functions(
        functions, max_instr=max_instructions, optimization=optimization,
    )


def _cache_dir() -> str:
    """The compiled-kernel cache directory — per-user, not the shared
    `$TMPDIR/manta_compiled`: these `.so`s get `dlopen`ed, and a fixed
    world-writable path would let any local user pre-place a library at
    a predictable location (and made the dir unusable for the second
    user to touch it). `$XDG_CACHE_HOME` when set, else `~/.cache`,
    else a uid-scoped tempdir."""
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        home = os.path.expanduser("~")
        base = (os.path.join(home, ".cache") if home != "~"
                else os.path.join(tempfile.gettempdir(),
                                  f"manta-{os.getuid()}"))
    return os.path.join(base, "manta", "compiled")


def _compiled_functions(functions: dict[str, Any], *,
                        max_instr: int | None = _MAX_INSTR,
                        optimization: Literal["balanced", "runtime"] =
                        "balanced") -> dict[str, Any]:
    """Return C externals keyed like ``functions`` or raise loudly.

    `max_instr` is the cost-benefit gate, counted in MX nodes. The
    default suits transforms whose big functions are UNROLLED symbolic
    monsters (an EKF's full-world Jacobian) — codegen there costs more than
    interpretation ever saves. A specialized runtime with deliberately
    bounded kernels may override the gate after measuring its compile/runtime
    tradeoff."""
    if max_instr is not None:
        instruction_count = sum(
            f.n_instructions() for f in functions.values())
        if instruction_count > max_instr:
            raise CompilationError(
                "native compilation refused: kernel has "
                f"{instruction_count} CasADi instructions, above the "
                f"configured limit {max_instr}")
    compiler = shutil.which("cc")
    if compiler is None:
        raise CompilationError(
            "native compilation requested but no 'cc' compiler is on PATH")
    try:
        tmp = tempfile.mkdtemp()
    except OSError as exc:
        raise CompilationError(
            f"cannot create native compilation workspace: {exc}") from exc
    try:
        try:
            cg = ca.CodeGenerator("mod.c", {"with_header": True})
            for f in functions.values():
                cg.add(f)
            cg.generate(tmp + os.sep)
            with open(os.path.join(tmp, "mod.c")) as fh:
                src = fh.read()
        except (OSError, RuntimeError) as exc:
            raise CompilationError(
                f"native CasADi code generation failed: {exc}") from exc
        compiler_flags = (
            ("-O3", "-march=native")
            if optimization == "runtime" else ("-O1",)
        )
        # Generated libraries are cached and may stay around across software
        # updates.  Optimization and target-architecture flags are therefore
        # part of the identity, not merely build-time details.
        source_key = hashlib.sha1(
            src.encode() + b"\0" + "\0".join(compiler_flags).encode()
            + b"\0" + platform.platform().encode()
        ).hexdigest()[:16]
        mapping_key = source_key + ":" + hashlib.sha1(
            "\0".join(
                f"{key}\0{function.name()}"
                for key, function in functions.items()
            ).encode()
        ).hexdigest()[:16]
        with _EXTERNAL_CACHE_LOCK:
            if mapping_key in _EXTERNAL_CACHE:
                return _EXTERNAL_CACHE[mapping_key]
            cache_dir = _cache_dir()
            try:
                os.makedirs(cache_dir, mode=0o700, exist_ok=True)
            except OSError as exc:
                raise CompilationError(
                    f"cannot create native-kernel cache {cache_dir!r}: "
                    f"{exc}. Set XDG_CACHE_HOME to a writable directory") \
                    from exc
            so_path = os.path.join(cache_dir, f"mod_{source_key}.so")
            try:
                if not os.path.exists(so_path):
                    # Generic transforms use -O1 because their huge
                    # straight-line symbolic Jacobians make -O2/3 compile
                    # time super-linear for little runtime gain. Deliberately
                    # bounded, repeatedly evaluated controller kernels may
                    # request the runtime profile instead.
                    # Build to a private temp
                    # path, then atomically publish — a concurrent process
                    # never dlopens a half-written .so.
                    so_tmp = os.path.join(tmp, f"mod_{source_key}.so")
                    subprocess.run(
                        [compiler, *compiler_flags, "-fPIC", "-shared",
                         os.path.join(tmp, "mod.c"), "-o", so_tmp],
                        check=True, capture_output=True,
                        timeout=120 if optimization == "runtime" else 30)
                    # `/tmp` and the user cache are commonly different
                    # filesystems in containers and WSL. First copy into a
                    # private file *inside* the cache, then rename there; the
                    # final publication remains atomic without relying on a
                    # cross-device rename.
                    publish_fd = -1
                    publish_tmp = ""
                    try:
                        publish_fd, publish_tmp = tempfile.mkstemp(
                            prefix=f".{source_key}-", suffix=".so",
                            dir=cache_dir,
                        )
                        os.close(publish_fd)
                        publish_fd = -1
                        shutil.copyfile(so_tmp, publish_tmp)
                        os.replace(publish_tmp, so_path)
                        publish_tmp = ""
                    except OSError as exc:
                        raise CompilationError(
                            f"cannot publish native kernel to {so_path!r}: "
                            f"{exc}. Set XDG_CACHE_HOME to a writable "
                            "directory") from exc
                    finally:
                        if publish_fd >= 0:
                            os.close(publish_fd)
                        if publish_tmp:
                            try:
                                os.unlink(publish_tmp)
                            except OSError:
                                pass
                out = {k: ca.external(f.name(), so_path)
                       for k, f in functions.items()}
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") \
                    if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                raise CompilationError(
                    f"native compilation failed with exit code "
                    f"{exc.returncode}: {stderr.strip() or 'no compiler output'}") \
                    from exc
            except subprocess.TimeoutExpired as exc:
                raise CompilationError(
                    f"native compilation exceeded the {exc.timeout}-second "
                    "compiler deadline") from exc
            except OSError as exc:
                raise CompilationError(
                    f"native compilation could not execute {compiler!r}: "
                    f"{exc}") from exc
            except CompilationError:
                raise
            except RuntimeError as exc:
                raise CompilationError(
                    f"compiled kernel {so_path!r} could not be loaded: "
                    f"{exc}") from exc
            _EXTERNAL_CACHE[mapping_key] = out
            return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
