"""Explicit compilation of CasADi kernels to cached C externals.

``TargetNumpy(..., compile=True)`` is a contract, not a hint: every requested
function is returned as a CasADi ``External`` or this module raises an
actionable :class:`CompilationError`. Silently reverting to an interpreted MX
graph makes performance tests lie and can violate a controller's runtime
budget.

This module is also the single owner of Manta's native-library cache
discipline. Every cached ``.so`` Manta ``dlopen``s — CasADi kernel externals,
the native filter-replay runner, and the HPIPM/OSQP QP bridges — is built and
published through :func:`build_native_library`, so all of them share one
directory layout, one atomic-publish protocol, one error vocabulary, and one
cache key. That key folds in :func:`toolchain_identity`: the compiler
(path, version, target), the Python/CasADi ABI the library is loaded into,
and — whenever the build targets the host CPU with ``-march=native`` — the
CPU's own identity, so a cache copied between machines or carried across a
compiler upgrade can never serve a stale or wrong-microarchitecture binary on
the live MPC/estimator path.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import casadi as ca

# `ca.external` wrappers keyed by generated source plus caller mapping, so a
# second runtime skips both gcc and dlopen without accidentally returning the
# first caller's dictionary keys. Guarded because compilation can be concurrent.
_EXTERNAL_CACHE: dict[str, dict[str, Any]] = {}
_EXTERNAL_CACHE_LOCK = threading.Lock()
# Above this many MX nodes, codegen + gcc on the (huge symbolic) functions
# usually costs more than it ever saves — e.g. an EKF that linearizes the
# full coupled world tick is ~7k nodes -> hundreds of thousands of lines of
# C. The gate refuses those *before* codegen (which is itself slow for them)
# with an error naming `max_instructions`, the parameter every public compile
# entry (`TargetNumpy`, `NumpyRuntime.compile_functions`, `compile_functions`)
# accepts to raise or disable it for a deliberately large simulation whose
# owner has declared a cold-build ceiling.
DEFAULT_MAX_INSTRUCTIONS = 3000
# Native code is a deployment/readiness artifact, not work for a real-time
# tick.  Give optimized CasADi kernels enough room to build, but never let a
# malformed or pathologically large generated translation unit wait forever.
DEFAULT_COMPILATION_TIMEOUT_S = 300.0
Optimization = Literal[
    "startup", "balanced", "runtime", "O0", "O1", "O2"
]


class CompilationError(RuntimeError):
    """A requested native kernel could not be produced or loaded."""


_NATIVE_LIBRARY_KEY_VERSION = b"manta-native-library-v2"
# /proc/cpuinfo fields that identify a microarchitecture (x86 and ARM
# spellings). Frequencies, cache sizes, and core indices are deliberately
# excluded: they vary per core and per boot without changing the ISA.
_CPU_IDENTITY_FIELDS = frozenset({
    "vendor_id", "cpu family", "model", "model name", "stepping", "flags",
    "features", "isa", "uarch", "cpu implementer", "cpu architecture",
    "cpu variant", "cpu part", "cpu revision", "hardware",
})


def _probe(command: list[str], *, timeout_s: float) -> str:
    """Run a short toolchain probe and return stdout+stderr, or raise."""
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True,
        timeout=timeout_s, stdin=subprocess.DEVNULL,
    )
    return completed.stdout + completed.stderr


def cpu_identity() -> str:
    """Identify the host CPU microarchitecture for ``-march=native`` builds.

    Reads the stable identity fields of ``/proc/cpuinfo`` (vendor, family,
    model, stepping, ISA feature flags; ARM implementer/part/variant) and
    falls back to ``platform.processor()`` where that file is unavailable.
    Raises :class:`CompilationError` when no identity can be determined at
    all: a native-targeted library must never be cached under an anonymous
    CPU.
    """
    identity = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            identity = "\n".join(sorted({
                line.strip()
                for line in stream
                if ":" in line
                and line.split(":", 1)[0].strip().lower()
                in _CPU_IDENTITY_FIELDS
            }))
    except OSError:
        identity = ""
    if not identity:
        identity = platform.processor() or platform.machine()
    if not identity:
        raise CompilationError(
            "cannot identify the host CPU for a -march=native build: "
            "/proc/cpuinfo is unavailable and platform.processor() is empty"
        )
    return identity


def toolchain_identity(
    compiler: str,
    compiler_flags: tuple[str, ...],
    *,
    timeout_s: float | None,
) -> bytes:
    """Return the identity that owns a native-library cache entry.

    The identity covers everything that changes the machine code or the ABI
    it is loaded into without changing the generated source:

    * the compiler — real path, ``--version`` banner, and ``-dumpmachine``
      target triple;
    * the operating system, machine, and Python/CasADi ABI (``SOABI``,
      interpreter version, CasADi version — the bundled headers and
      libraries the bridges compile against);
    * the host CPU (see :func:`cpu_identity`) plus the compiler's own
      resolution of ``-march=native`` whenever that flag is in
      ``compiler_flags``.

    Raises :class:`CompilationError` when the compiler cannot be probed.
    """
    probe_timeout = 5.0 if timeout_s is None else max(0.001, min(5.0, timeout_s))
    try:
        compiler_version = _probe([compiler, "--version"],
                                  timeout_s=probe_timeout)
        compiler_target = _probe([compiler, "-dumpmachine"],
                                 timeout_s=probe_timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CompilationError(
            f"cannot identify native compiler {compiler!r}: {exc}"
        ) from exc
    identity = [
        os.path.realpath(compiler),
        compiler_version.strip(),
        compiler_target.strip(),
        platform.platform(),
        platform.machine(),
        platform.python_implementation(),
        sys.version,
        str(sysconfig.get_config_var("SOABI")),
        ca.__version__,
    ]
    if "-march=native" in compiler_flags:
        identity.append(cpu_identity())
        # The compiler's own view of the host (gcc prints the resolved
        # `-march=`/`-mtune=` in the cc1 command line, clang prints
        # `-target-cpu`). Best effort: the cpuinfo identity above already
        # pins the microarchitecture when this probe is unsupported.
        try:
            resolved = _probe(
                [compiler, "-march=native", "-E", "-v", "-x", "c", "-"],
                timeout_s=probe_timeout,
            )
        except (OSError, subprocess.SubprocessError):
            resolved = ""
        identity.append("\n".join(sorted(
            token for line in resolved.splitlines() for token in line.split()
            if token.startswith(("-march=", "-mtune=", "-mcpu=",
                                 "-target-cpu", "-target-feature", "+"))
        )))
    return "\0".join(identity).encode()


def native_compiler(*, what: str) -> str:
    """Locate the system C compiler or raise an actionable error."""
    compiler = shutil.which("cc")
    if compiler is None:
        raise CompilationError(
            f"{what} requested but no 'cc' compiler is on PATH")
    return compiler


@dataclass(frozen=True)
class NativeLibrary:
    """One published shared library in Manta's native cache."""

    path: str
    key: str
    rebuilt: bool


def native_library_key(
    source: str,
    *,
    stem: str,
    compiler_flags: tuple[str, ...],
    link_args: tuple[str, ...],
    identity_salt: bytes,
    toolchain: bytes,
) -> str:
    """The cache key of a native library: source, build line, salt, toolchain."""
    digest = hashlib.sha256()
    for chunk in (
        _NATIVE_LIBRARY_KEY_VERSION, stem.encode(), source.encode(),
        "\0".join(compiler_flags).encode(), "\0".join(link_args).encode(),
        identity_salt, toolchain,
    ):
        digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def build_native_library(
    source: str,
    *,
    stem: str,
    what: str,
    compiler_flags: tuple[str, ...],
    link_args: tuple[str, ...] = (),
    identity_salt: bytes = b"",
    cache_subdir: str = "",
    timeout_s: float | None,
    compiler: str | None = None,
) -> NativeLibrary:
    """Compile one C translation unit to a cached, atomically published ``.so``.

    This is the single compile-and-publish path for every native library
    Manta loads. ``stem`` names the library family (``mod``, ``replay``,
    ``manta_hpipm``, ...), ``what`` labels errors, ``compiler_flags`` precede
    the source on the compiler command line and ``link_args`` follow it
    (include/library paths, ``-l``, rpath). ``identity_salt`` lets a caller
    fold extra build inputs (a header it compiles against) into the key.

    The cache entry lives at ``<cache>/<cache_subdir>/<stem>_<key>.so`` where
    the key covers the source, the full build line, the salt, and
    :func:`toolchain_identity`. An existing entry is reused without invoking
    the compiler. A new one is built in a private temporary directory and
    published by copying into the cache directory and ``os.replace``-ing
    onto its final name, so a concurrent process never ``dlopen``s a
    half-written file. Every failure — no compiler, unwritable cache,
    compiler error, deadline — raises :class:`CompilationError`.
    """
    if timeout_s is not None and (
        not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0
    ):
        raise CompilationError(
            f"{what} exceeded its code-generation and compiler deadline"
        )
    if compiler is None:
        compiler = native_compiler(what=what)
    toolchain = toolchain_identity(compiler, compiler_flags, timeout_s=timeout_s)
    key = native_library_key(
        source, stem=stem, compiler_flags=compiler_flags, link_args=link_args,
        identity_salt=identity_salt, toolchain=toolchain,
    )
    cache_dir = _cache_dir()
    if cache_subdir:
        cache_dir = os.path.join(cache_dir, cache_subdir)
    try:
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise CompilationError(
            f"cannot create {what} cache {cache_dir!r}: {exc}. "
            "Set XDG_CACHE_HOME to a writable directory") from exc
    library_path = os.path.join(cache_dir, f"{stem}_{key}.so")
    if os.path.exists(library_path):
        return NativeLibrary(path=library_path, key=key, rebuilt=False)
    try:
        workspace = tempfile.mkdtemp(prefix=f"manta-{stem}-")
    except OSError as exc:
        raise CompilationError(
            f"cannot create {what} compilation workspace: {exc}") from exc
    try:
        source_path = os.path.join(workspace, f"{stem}.c")
        private_path = os.path.join(workspace, f"{stem}_{key}.so")
        try:
            with open(source_path, "w", encoding="utf-8") as stream:
                stream.write(source)
            subprocess.run(
                [compiler, *compiler_flags, "-fPIC", "-shared", source_path,
                 *link_args, "-o", private_path],
                check=True, capture_output=True, timeout=timeout_s,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr.decode(errors="replace")
                      if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
            raise CompilationError(
                f"{what} compilation failed with exit code {exc.returncode}: "
                f"{stderr.strip() or 'no compiler output'}") from exc
        except subprocess.TimeoutExpired as exc:
            raise CompilationError(
                f"{what} compilation exceeded the {exc.timeout:g}-second "
                "remaining code-generation and compiler deadline") from exc
        except OSError as exc:
            raise CompilationError(
                f"{what} compilation could not execute {compiler!r}: {exc}"
            ) from exc
        # `/tmp` and the user cache are commonly different filesystems in
        # containers and WSL. Copy into a private file *inside* the cache,
        # then rename there; publication stays atomic without relying on a
        # cross-device rename.
        publish_fd = -1
        publish_path = ""
        try:
            publish_fd, publish_path = tempfile.mkstemp(
                prefix=f".{stem}_{key}-", suffix=".so", dir=cache_dir)
            os.close(publish_fd)
            publish_fd = -1
            shutil.copyfile(private_path, publish_path)
            os.replace(publish_path, library_path)
            publish_path = ""
        except OSError as exc:
            raise CompilationError(
                f"cannot publish {what} library to {library_path!r}: {exc}. "
                "Set XDG_CACHE_HOME to a writable directory") from exc
        finally:
            if publish_fd >= 0:
                os.close(publish_fd)
            if publish_path:
                try:
                    os.unlink(publish_path)
                except OSError:
                    pass
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return NativeLibrary(path=library_path, key=key, rebuilt=True)


def validate_max_instructions(value: int | None) -> int | None:
    """``None`` disables the instruction gate; otherwise a positive int."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "max_instructions must be a positive integer or None (no gate), "
            f"got {value!r}")
    return value


def compile_functions(functions: dict[str, Any], *,
                      max_instructions: int | None = DEFAULT_MAX_INSTRUCTIONS,
                      optimization: Optimization = "balanced",
                      timeout_s: float | None = DEFAULT_COMPILATION_TIMEOUT_S,
                      ) -> dict[str, Any]:
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
    validate_max_instructions(max_instructions)
    if optimization not in ("startup", "balanced", "runtime", "O0", "O1", "O2"):
        raise ValueError(
            "optimization must be startup/balanced/runtime or O0/O1/O2")
    if timeout_s is not None and (
        not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or not 0.0 < float(timeout_s)
    ):
        raise ValueError("timeout_s must be a positive finite number")
    return _compiled_functions(
        functions, max_instr=max_instructions, optimization=optimization,
        timeout_s=(None if timeout_s is None else float(timeout_s)),
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
                        max_instr: int | None = DEFAULT_MAX_INSTRUCTIONS,
                        optimization: Optimization = "balanced",
                        timeout_s: float | None = DEFAULT_COMPILATION_TIMEOUT_S,
                        ) -> dict[str, Any]:
    """Return C externals keyed like ``functions`` or raise loudly.

    `max_instr` is the cost-benefit gate, counted in MX nodes. The
    default suits transforms whose big functions are UNROLLED symbolic
    monsters (an EKF's full-world Jacobian) — codegen there costs more than
    interpretation ever saves. A specialized runtime with deliberately
    bounded kernels may override the gate after measuring its compile/runtime
    tradeoff."""
    if timeout_s is not None and (
        not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or not 0.0 < float(timeout_s)
    ):
        raise ValueError("timeout_s must be a positive finite number")
    deadline = (
        None if timeout_s is None else time.monotonic() + float(timeout_s)
    )
    if max_instr is not None:
        instruction_count = sum(
            f.n_instructions() for f in functions.values())
        if instruction_count > max_instr:
            raise CompilationError(
                "native compilation refused: kernel has "
                f"{instruction_count} CasADi instructions, above the "
                f"configured limit max_instructions={max_instr}. Raise "
                "max_instructions (TargetNumpy(..., max_instructions=N) or "
                "compile_functions(..., max_instructions=N); None disables "
                "the gate) after measuring the compile/runtime tradeoff")
    compiler = native_compiler(what="native compilation")
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    remaining_s = None if deadline is None else deadline - time.monotonic()
    if remaining_s is not None and remaining_s <= 0.0:
        raise CompilationError(
            f"native compilation exceeded the {float(timeout_s):g}-second "
            "code-generation and compiler deadline"
        )
    # Semantic profiles preserve existing callers. Full simulations may
    # request O0/O1/O2 explicitly; bounded, repeatedly evaluated robot
    # kernels request runtime.
    compiler_flags = {
        "startup": ("-O0",),
        "balanced": ("-O1",),
        "runtime": ("-O3", "-march=native"),
        "O0": ("-O0",),
        "O1": ("-O1",),
        "O2": ("-O2",),
    }[optimization]
    mapping = hashlib.sha1(
        "\0".join(
            f"{key}\0{function.name()}"
            for key, function in functions.items()
        ).encode()
    ).hexdigest()[:16]
    with _EXTERNAL_CACHE_LOCK:
        library = build_native_library(
            src, stem="mod", what="native compilation",
            compiler_flags=compiler_flags, timeout_s=remaining_s,
            compiler=compiler,
        )
        mapping_key = library.key + ":" + mapping
        if mapping_key in _EXTERNAL_CACHE:
            return _EXTERNAL_CACHE[mapping_key]
        try:
            out = {k: ca.external(f.name(), library.path)
                   for k, f in functions.items()}
        except RuntimeError as exc:
            raise CompilationError(
                f"compiled kernel {library.path!r} could not be loaded: "
                f"{exc}") from exc
        _EXTERNAL_CACHE[mapping_key] = out
        return out
