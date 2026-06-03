"""The backend contract.

A backend implements exactly two things: a way to run/translate a
`ca.Function`, and one generic lowering of a typed `Module`
(`manta.ir.module`). Nothing in a backend mentions a transform — `Sim`,
`EKF`, `LQR`, and the recurrence blocks all reach it as Modules.

`as_module` is the one entry-point helper: backends accept either a
`Module` or any transform exposing `.module()`.
"""

from __future__ import annotations

from ..ir.module import Module


def as_module(x, who: str) -> Module:
    """Return `x` as a `Module` (calling `x.module()` for a transform)."""
    if isinstance(x, Module):
        return x
    m = getattr(x, "module", None)
    if callable(m):
        mod = m()
        if isinstance(mod, Module):
            return mod
    raise TypeError(
        f"{who}: expected a Module or a transform with .module() "
        f"(Sim, EKF, LQR, a recurrence block), got {type(x).__name__}")
