"""Target — the backend ABC.

A `Target` lowers any IR *block* (`Sim`, `EKF`, `LQR`, and the
freestanding `recurrence` blocks — PID, Madgwick, the IMU integrator) to
a backend-specific artifact: a native-Python runtime, emitted C++, a
future Rust crate, etc.

`lower_block(block, **opts)` is the single entry point. It reads the
block's `RUNTIME_KIND` (see `manta.codegen.block`) and dispatches to the
backend's `lower_<kind>` method — so the dispatch is keyed on the small
*closed* set of runtime kinds, not on the *open* set of block classes. A
backend supports a kind iff it defines the matching `lower_<kind>`
handler; a block whose kind has no handler fails loudly and locally at the
call (a `NotImplementedError` naming the supported kinds) rather than
silently or deep in a stack.

That is what makes a half-finished backend obvious, and what makes adding
a block free: a new evaluator block (e.g. another filter) reuses the
existing `lower_evaluator` handler with zero backend edits.

The public `TargetNumpy(...)` / `TargetCpp(...)` callables are thin
wrappers over the concrete backends below. Adding a backend = one
`Target` subclass implementing the `lower_<kind>` handlers it supports.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from .block import block_kind


class Target(ABC):
    """Backend contract: lower a block to an artifact.

    Subclasses implement one `lower_<kind>` method per runtime kind they
    support (e.g. `lower_evaluator`, `lower_ekf`). Each handler takes the
    block plus backend-specific keyword options (e.g. `out_dir`,
    `class_name` for a file-emitting backend); `lower_block` forwards them.
    """

    name: str = "target"

    def lower_block(self, block, **opts) -> Any:
        """Lower one IR block, dispatching on its `RUNTIME_KIND`."""
        kind = block_kind(block)
        if kind is None:
            raise TypeError(
                f"{self.name}: {type(block).__name__} is not a lowerable "
                f"block (no RUNTIME_KIND). Expected a Sim, EKF, LQR, or a "
                f"recurrence block.")
        handler = getattr(self, f"lower_{kind}", None)
        if handler is None:
            supported = sorted(
                n[len("lower_"):] for n in dir(self)
                if n.startswith("lower_") and n != "lower_block"
                and callable(getattr(self, n)))
            raise NotImplementedError(
                f"{self.name}: no lowering for block kind {kind!r} "
                f"({type(block).__name__}). Supported kinds: {supported}.")
        return handler(block, **opts)
