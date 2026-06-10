"""Shared fitting declarations — `Prior` and `Window`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prior:
    """Gaussian prior on one fitted parameter.

    Args:
        sigma — 1-σ width. Scalar (isotropic across the parameter's
                components) or a per-component sequence. With `log=True`
                it is RELATIVE (log-space): `sigma=0.3` ≈ ±30%.
        mean  — prior mean. `None` (default) → the model's declared
                value.
        log   — fit `log(p)` instead of `p`. Scalar, strictly-positive
                parameters only (mass, a thrust magnitude along one
                axis). Keeps the parameter positive with no constraint
                and makes pure scale ambiguities linear.
                `NoiseFit` ignores this flag — noise σ is ALWAYS fit in
                log-space, and `sigma` there is always relative.
    """
    sigma: float | tuple = None
    mean: float | tuple | None = None
    log: bool = False


@dataclass(frozen=True)
class Window:
    """One fitting window: a short recorded rollout.

    Args:
        x0 — nested initial state dict (the `sim.state` shape:
             `{craft: {slot: value}}`). Slots omitted fall back to the
             world's initial state.
        u  — recorded controls: `{input name/suffix: scalar | (K,)}`.
             A scalar is held for the whole window; inputs omitted hold
             their model default.
        z  — recorded sensor readings: `{sensor name/suffix:
             (K, dim) | (K,)}`. Row k is the reading produced by step k
             (the step taken FROM state k). For `Fit`, only sensors
             present here enter the loss; for `NoiseFit`, every chosen
             sensor needs a trace.
        dt — fixed step, seconds.
        t0 — world-clock time of x0.
    """
    x0: dict
    u: dict = field(default_factory=dict)
    z: dict = field(default_factory=dict)
    dt: float = 0.01
    t0: float = 0.0
