"""TargetNumpy — the native-Python backend.

ONE kernel engine + four thin typed views. `NumpyRuntime` (`_runtime`)
is the engine: the generic typed-arg gather → `ca.Function` call →
scatter over a Module's entry points (`call(method, values)`), plus the
shared port metadata helpers. Each view subclasses it and exposes
exactly the surface its Module shape implies — nothing else:

  * `NumpySim` (`_sim`)               — THREADED + an oracle ``step``
                        entry. Held state: `sim.state` (nested dict),
                        `step(dt)`/`step_n`, `outputs()`/`reading(name)`,
                        `attach_driver` (feeds the NOISE port).
  * `NumpyFilter` (`_filter`)         — HELD + a ``predict`` entry.
                        `predict`/`update` (by sensor *name* or a
                        caller-supplied `h(x)`), `x`/`P`, `reset`,
                        `state_dict`, `Q`.
  * `NumpyRecurrence` (`_recurrence`) — HELD + an OUTPUT port.
                        `step(dt, **inputs)`, `readouts()`, `reset`,
                        `state`.
  * `NumpyRegulator` (`_regulator`)   — THREADED + a ``control`` entry.
                        `u(x)`, `control(state_dict)`, `retarget`,
                        `reprogram`, `x_ref`, `gain`, `u_ff`.

`NoiseDriver` (`_noise`) drives the oracle's NOISE port; `_compile`
holds the optional cc-compiled-kernel path.

`TargetNumpy(x)` inspects the Module once and returns the matching view;
a Module that matches no view (e.g. the Sim's noiseless deploy bundle)
comes back as the bare engine — `call()` works on any entry point.

Everything is driven by the IR's types: slot names come from the manifold
spec, input/sensor/noise names + defaults/σ/rates from the Ports. The
backend never mentions a transform.
"""

from __future__ import annotations

from ...ir.module import ModuleKind
from ._compile import (
    DEFAULT_COMPILATION_TIMEOUT_S,
    DEFAULT_MAX_INSTRUCTIONS,
    CompilationError,
    compile_functions,
    validate_max_instructions,
)
from ._filter import FilterCheckpoint, NumpyFilter, UpdateResult
from ._filter_replay import (
    FilterReplayProgram,
    FilterReplayResult,
    NativeFilterReplay,
    ReplayBoundary,
    ReplayCheckpointResult,
    ReplayOperation,
    ReplayPredict,
    ReplayUpdate,
    TargetFilterReplay,
)
from ._noise import NoiseCheckpoint, NoiseDriver
from ._recurrence import NumpyRecurrence
from ._regulator import NumpyRegulator
from ._runtime import NumpyRuntime
from ._sim import NumpySim, SimCheckpoint

__all__ = [
    "CompilationError",
    "DEFAULT_COMPILATION_TIMEOUT_S",
    "DEFAULT_MAX_INSTRUCTIONS",
    "FilterCheckpoint",
    "FilterReplayProgram",
    "FilterReplayResult",
    "NativeFilterReplay",
    "NoiseCheckpoint",
    "NoiseDriver",
    "NumpyFilter",
    "NumpyRecurrence",
    "NumpyRegulator",
    "NumpyRuntime",
    "NumpySim",
    "ReplayBoundary",
    "ReplayCheckpointResult",
    "ReplayOperation",
    "ReplayPredict",
    "ReplayUpdate",
    "SimCheckpoint",
    "TargetFilterReplay",
    "TargetNumpy",
    "UpdateResult",
    "compile_functions",
]


def TargetNumpy(
    x,
    *,
    compile: bool = False,
    optimization: str | None = None,
    compile_timeout_s: float | None = DEFAULT_COMPILATION_TIMEOUT_S,
    max_instructions: int | None = DEFAULT_MAX_INSTRUCTIONS,
) -> NumpyRuntime:
    """Lower a typed `Module` — or any transform exposing `.module()`
    (`Sim`, `EKF`, `LQR`, a recurrence block) — to the matching
    native-Python view (sim / filter / recurrence / regulator), or the
    bare kernel engine when no view matches.

    `compile=True` builds the kernel's CasADi functions with optimized native
    code (O1 by default for full-simulation graphs, `-O3 -march=native` for
    other runtime models). A caller may explicitly select O0, O1, or O2 for a
    simulation whose compile/runtime tradeoff and cold-build ceiling are
    scenario-specific. It
    calls them as externals instead of interpreting the MX graph. Results are
    cached on disk; the default cold-build deadline is five minutes, and a
    simulation caller may replace or disable that deadline. It raises
    `CompilationError` if an external cannot be produced; explicit native
    execution never silently becomes interpretation. Pair with `NumpySim`'s
    `step_n` to fold substeps for a further amortization.

    `max_instructions` is the cost-benefit size gate, counted in CasADi
    instructions over every kernel (default `DEFAULT_MAX_INSTRUCTIONS`).
    Above it compilation is refused with an error naming this parameter. A
    deliberately large full-truth simulation whose owner has declared a
    finite cold-build ceiling raises it, or passes `None` to disable the
    gate entirely."""
    from ..target import as_module
    m = as_module(x, "TargetNumpy")
    if (
        optimization is not None
        or compile_timeout_s != DEFAULT_COMPILATION_TIMEOUT_S
    ):
        if not compile:
            raise ValueError(
                "optimization and compile_timeout_s require compile=True"
            )
        if m.kind is not ModuleKind.SIMULATOR:
            raise ValueError(
                "explicit TargetNumpy compilation policy is only for simulation"
            )
    if max_instructions != DEFAULT_MAX_INSTRUCTIONS and not compile:
        raise ValueError("max_instructions requires compile=True")
    validate_max_instructions(max_instructions)
    if optimization is not None:
        if optimization not in {"O0", "O1", "O2"}:
            raise ValueError("simulation optimization must be O0, O1, or O2")
    runtime = _select_view(m)(m)
    return (
        runtime._enable_compile(
            optimization=optimization,
            timeout_s=compile_timeout_s,
            max_instructions=max_instructions,
        )
        if compile else runtime
    )


def _select_view(m):
    """Select a runtime view from the Module's explicit capability."""
    return {
        ModuleKind.KERNEL: NumpyRuntime,
        ModuleKind.SIMULATOR: NumpySim,
        ModuleKind.FILTER: NumpyFilter,
        ModuleKind.RECURRENCE: NumpyRecurrence,
        ModuleKind.REGULATOR: NumpyRegulator,
    }[m.kind]
