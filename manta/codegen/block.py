"""Block — the contract every lowerable IR satisfies.

A *block* is any IR a `Target` can lower to a runtime: the whole-world
analysis transforms (`Sim`, `EKF`, `LQR`) and the freestanding
signal-processing blocks (PID, Madgwick/Mahony, the IMU strapdown
integrator). Every block is now lowered through the ONE generic path —
`to_module(block)` (`manta.codegen.module_build`) → a backend's
`lower_module` — so the runtime shape lives in the `Module` IR, not here.

`RUNTIME_KIND` survives only as a coarse dispatch tag for the legacy
`Target.lower_block` entry point (see `manta.codegen.target`): both
backends' `lower_<kind>` handlers just forward to the generic Module path.
A new block needs **no backend code** — it declares a kind and produces a
Module via `to_module`.

Runtime kinds:

  * ``"evaluator"`` — `Sim`, `LQR`, and every `RecurrenceBlock`.
  * ``"ekf"``       — the stateful Kalman filter.

The kinds live here as string constants so the set stays closed and
greppable; `RUNTIME_KIND` on an IR class names which one it is.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

# --- The closed set of runtime kinds a Target switches on -------------------
KIND_EVALUATOR: str = "evaluator"
KIND_EKF:       str = "ekf"

ALL_KINDS: tuple[str, ...] = (KIND_EVALUATOR, KIND_EKF)


@runtime_checkable
class Block(Protocol):
    """Structural type for a lowerable IR.

    The only universal requirement is a `RUNTIME_KIND` class attribute
    naming which backend handler lowers it. The richer per-kind contract
    (e.g. the kernels + state spec + ports a ``recurrence`` block exposes)
    is documented on that kind's base class, not here — different kinds
    carry genuinely different payloads.
    """

    RUNTIME_KIND: ClassVar[str]


def block_kind(obj) -> str | None:
    """Return a block's `RUNTIME_KIND`, or `None` if it isn't a block.

    Reads the class attribute (kinds are per-class, not per-instance), so
    a plain object that merely happens to have an instance attribute named
    `RUNTIME_KIND` is not mistaken for a block.
    """
    return getattr(type(obj), "RUNTIME_KIND", None)
