"""Target — the backend ABC.

A `Target` lowers any IR transform (`Sim`, `EKF`, `LQR`) to a
backend-specific artifact: a native-Python runtime, emitted C++, a future
Rust crate, etc. Each backend implements the three `lower_*` hooks;
`lower()` holds the IR-type dispatch so it lives in exactly one place.

A backend that doesn't *yet* support a transform implements that hook to
raise `NotImplementedError` with a concrete message — so an unsupported
lowering fails loudly and locally (right at the call) instead of silently
or deep in a stack. That makes a half-finished backend obvious: the
missing piece is a hook that raises, not a gap you discover at runtime.

The public `TargetNumpy(...)` / `TargetCpp(...)` callables are thin
wrappers over the concrete backends below; their signatures are
unchanged. Adding a backend = one `Target` subclass implementing the
three hooks (+ a thin public wrapper if you want a bespoke signature).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Target(ABC):
    """Backend contract: lower a `Sim` / `EKF` / `LQR` IR to an artifact.

    Subclasses implement `lower_sim` / `lower_ekf` / `lower_lqr`. Each hook
    takes the IR plus backend-specific keyword options (e.g. `out_dir`,
    `class_name` for a file-emitting backend); `lower()` forwards them.
    """

    name: str = "target"

    @abstractmethod
    def lower_sim(self, sim, **opts) -> Any:
        """Lower a `Sim` IR (forward dynamics)."""

    @abstractmethod
    def lower_ekf(self, ekf, **opts) -> Any:
        """Lower an `EKF` IR (predict + measurement update)."""

    @abstractmethod
    def lower_lqr(self, lqr, **opts) -> Any:
        """Lower an `LQR` IR (baked feedback control law)."""

    def lower(self, ir, **opts) -> Any:
        """Dispatch on IR type to the matching `lower_*` hook."""
        from ..sim import Sim
        from ..estimation.ekf import EKF
        from ..control.lqr import LQR
        if isinstance(ir, Sim):
            return self.lower_sim(ir, **opts)
        if isinstance(ir, EKF):
            return self.lower_ekf(ir, **opts)
        if isinstance(ir, LQR):
            return self.lower_lqr(ir, **opts)
        raise TypeError(
            f"{self.name}: no handler for IR type {type(ir).__name__}. "
            f"Expected Sim, EKF, or LQR.")
