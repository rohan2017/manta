"""NoiseDriver — stochastic source for a Module's NOISE port."""

from __future__ import annotations

import numpy as np


class NoiseDriver:
    """Draws the per-step samples that make an oracle Module's noise live.

    Binds to the NOISE port's fields (name, dim, σ); each `sample()` is an
    independent `N(0, σ²)` draw per active channel. Deliberately thin and
    swappable — kept out of the pure kernels — and simply omitted on a
    deploy target.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._channels: list[tuple[str, int, float]] = []

    def bind(self, fields) -> None:
        self._channels = [(f.name, f.dim, float(f.sigma or 0.0))
                          for f in fields]

    def sample(self) -> dict[str, np.ndarray]:
        return {name: self._rng.normal(0.0, sigma, dim)
                for name, dim, sigma in self._channels if sigma > 0.0}

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def __repr__(self) -> str:
        active = sum(1 for _, _, s in self._channels if s > 0.0)
        return f"<NoiseDriver seed={self._seed} active={active}>"
