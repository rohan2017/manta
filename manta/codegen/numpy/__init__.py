"""TargetNumpy — the native-Python backend.

ONE kernel engine + four thin typed views. `NumpyRuntime` (`_runtime`)
is the engine: the generic typed-arg gather → `ca.Function` call →
scatter over a Module's entry points (`call(method, values)`), plus the
shared port metadata helpers. Each view subclasses it and exposes
exactly the surface its Module shape implies — nothing else:

  * `NumpySim` (`_sim`)               — THREADED + an oracle ``step``
                        entry. Held state: `sim.state` (nested dict),
                        `step(dt)`/`step_n`, `outputs()`/`reading(name)`,
                        `attach_driver` (feeds the NOISE port),
                        `rate_gate`/`command_latch`.
  * `NumpyFilter` (`_filter`)         — HELD + a ``predict`` entry.
                        `predict`/`update` (by sensor *name* or a
                        caller-supplied `h(x)`), `x`/`P`, `reset`,
                        `state_dict`, `Q`.
  * `NumpyRecurrence` (`_recurrence`) — HELD + an OUTPUT port.
                        `step(dt, **inputs)`, `readouts()`, `reset`,
                        `state`.
  * `NumpyRegulator` (`_regulator`)   — THREADED + a ``control`` entry.
                        `u(x)`, `control(state_dict)`, `retarget`,
                        `x_ref`.

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

from ...ir.module import Hosting, Role
from ._filter import NumpyFilter
from ._noise import NoiseDriver
from ._recurrence import NumpyRecurrence
from ._regulator import NumpyRegulator
from ._runtime import NumpyRuntime
from ._sim import NumpySim

__all__ = [
    "NoiseDriver", "NumpyFilter", "NumpyRecurrence", "NumpyRegulator",
    "NumpyRuntime", "NumpySim", "TargetNumpy",
]


def TargetNumpy(x, *, compile: bool = False) -> NumpyRuntime:
    """Lower a typed `Module` — or any transform exposing `.module()`
    (`Sim`, `EKF`, `LQR`, a recurrence block) — to the matching
    native-Python view (sim / filter / recurrence / regulator), or the
    bare kernel engine when no view matches.

    `compile=True` builds the kernel's CasADi functions with `cc` and
    calls them as externals instead of interpreting the MX graph
    (bit-identical, ~8x per call; cached on disk). Pair with `NumpySim`'s
    `step_n` to fold substeps for a further amortization."""
    from ..target import as_module
    m = as_module(x, "TargetNumpy")
    runtime = _select_view(m)(m)
    return runtime._enable_compile() if compile else runtime


def _select_view(m):
    """Pick the matching view class from the Module's SHAPE — Hosting plus the
    Role/field signature, never an entry-point name (the C++ emitter's
    no-name-matching discipline, applied to view selection):

      * HELD + a matrix State field (the EKF covariance `P`)  → NumpyFilter
      * HELD + an OUTPUT port (a recurrence's readouts)       → NumpyRecurrence
      * a NOISE port (the simulation oracle's live draw)      → NumpySim
      * returns a CONTROL port (a control law's `u`)          → NumpyRegulator
      * anything else (e.g. the Sim's noiseless deploy bundle) → the engine
    """
    if m.hosting is Hosting.HELD:
        if m.matrix_fields:
            return NumpyFilter
        if m.sole_port(Role.OUTPUT) is not None:
            return NumpyRecurrence
        return NumpyRuntime
    if m.sole_port(Role.NOISE) is not None:
        return NumpySim
    if m.returns_role(Role.CONTROL):
        return NumpyRegulator
    return NumpyRuntime
