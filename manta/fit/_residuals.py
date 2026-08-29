"""Dimension-generic residual statistics for fitted/reduced models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _owned_vector(value, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    owned = array.copy()
    owned.flags.writeable = False
    return owned


def _owned_covariance(value, *, dimension: int,
                      name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    owned = array.copy()
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class ResidualStatistics:
    """Bias and covariance evidence from independent residual sequences.

    ``instantaneous_covariance`` describes individual residual samples.
    ``white_equivalent_covariance`` is the Bartlett/Newey–West long-run
    covariance: using it as independent per-step noise reproduces the
    asymptotic integrated-error growth of the observed correlated sequence.
    Bias remains separate and is never folded into zero-mean covariance.
    """

    bias: NDArray[np.float64]
    instantaneous_covariance: NDArray[np.float64]
    white_equivalent_covariance: NDArray[np.float64]
    reference_dt_s: float
    correlation_lag_steps: int
    correlation_horizon_s: float
    samples: int
    windows: int
    effective_sample_size: NDArray[np.float64]

    def __post_init__(self) -> None:
        bias = _owned_vector(self.bias, name="ResidualStatistics.bias")
        dimension = bias.size
        object.__setattr__(self, "bias", bias)
        for name in ("instantaneous_covariance",
                     "white_equivalent_covariance"):
            object.__setattr__(self, name, _owned_covariance(
                getattr(self, name), dimension=dimension,
                name=f"ResidualStatistics.{name}"))
        effective = _owned_vector(
            self.effective_sample_size,
            name="ResidualStatistics.effective_sample_size")
        if effective.shape != bias.shape or np.any(effective < 1.0):
            raise ValueError("ResidualStatistics.effective_sample_size must "
                             "match bias and be >= 1")
        object.__setattr__(self, "effective_sample_size", effective)
        dt = float(self.reference_dt_s)
        horizon = float(self.correlation_horizon_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("ResidualStatistics.reference_dt_s must be "
                             "finite and positive")
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError("ResidualStatistics.correlation_horizon_s must "
                             "be finite and non-negative")
        if (isinstance(self.correlation_lag_steps, bool)
                or not isinstance(self.correlation_lag_steps, int)
                or self.correlation_lag_steps < 0):
            raise ValueError("ResidualStatistics.correlation_lag_steps must "
                             "be a non-negative integer")
        for name in ("samples", "windows"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 1):
                raise ValueError(f"ResidualStatistics.{name} must be a "
                                 "positive integer")
        object.__setattr__(self, "reference_dt_s", dt)
        object.__setattr__(self, "correlation_horizon_s", horizon)

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Estimator-ready white-equivalent covariance."""
        return self.white_equivalent_covariance


def _positive_semidefinite(matrix: NDArray[np.float64]
                           ) -> NDArray[np.float64]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return np.asarray(0.5 * (result + result.T), dtype=float)


def bartlett_hac_residual_statistics(
        sequences: Sequence[NDArray[np.float64]], *,
        reference_dt_s: float,
        correlation_horizon_s: float) -> ResidualStatistics:
    """Estimate bias, sample covariance, and Bartlett-HAC covariance.

    Each input is one independent ``(samples, dimension)`` sequence. Lagged
    products never cross sequence boundaries, so unrelated fitting windows do
    not acquire artificial adjacency. All sequences must share a dimension;
    the calculation is otherwise dimension-agnostic.
    """
    dt = float(reference_dt_s)
    horizon = float(correlation_horizon_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("reference_dt_s must be finite and positive")
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError("correlation_horizon_s must be finite and "
                         "non-negative")
    normalized: list[NDArray[np.float64]] = []
    dimension: int | None = None
    for index, sequence in enumerate(sequences):
        array = np.asarray(sequence, dtype=float)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(
                f"residual sequences[{index}] must have non-empty "
                "(samples, dimension) shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"residual sequences[{index}] must be finite")
        if dimension is None:
            dimension = array.shape[1]
        elif array.shape[1] != dimension:
            raise ValueError("residual sequences must share one dimension")
        normalized.append(array)
    if not normalized:
        raise ValueError("residual sequences may not be empty")

    residuals = np.vstack(normalized)
    bias = np.mean(residuals, axis=0)
    centered_sequences = [sequence - bias for sequence in normalized]
    centered = np.vstack(centered_sequences)
    sample_count = len(centered)
    instantaneous = centered.T @ centered / max(1, sample_count - 1)

    maximum_lag = min(len(sequence) - 1 for sequence in centered_sequences)
    lag_steps = min(maximum_lag, max(0, round(horizon / dt)))
    long_run = sum(
        (sequence.T @ sequence for sequence in centered_sequences),
        np.zeros_like(instantaneous),
    ) / sample_count
    for lag in range(1, lag_steps + 1):
        cross = sum(
            (sequence[lag:].T @ sequence[:-lag]
             for sequence in centered_sequences),
            np.zeros_like(instantaneous),
        ) / sample_count
        weight = 1.0 - lag / (lag_steps + 1.0)
        long_run += weight * (cross + cross.T)

    instantaneous = _positive_semidefinite(instantaneous)
    long_run = _positive_semidefinite(long_run)
    instantaneous_diagonal = np.diag(instantaneous)
    long_run_diagonal = np.diag(long_run)
    effective = np.full(int(dimension), float(sample_count))
    np.divide(
        sample_count * instantaneous_diagonal,
        long_run_diagonal,
        out=effective,
        where=long_run_diagonal > np.finfo(float).tiny,
    )
    effective = np.clip(effective, 1.0, float(sample_count))
    return ResidualStatistics(
        bias=bias,
        instantaneous_covariance=instantaneous,
        white_equivalent_covariance=long_run,
        reference_dt_s=dt,
        correlation_lag_steps=lag_steps,
        correlation_horizon_s=lag_steps * dt,
        samples=sample_count,
        windows=len(normalized),
        effective_sample_size=effective,
    )


__all__ = ["ResidualStatistics", "bartlett_hac_residual_statistics"]
