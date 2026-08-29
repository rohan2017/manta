"""Generic Bartlett-HAC evidence used by fitting and reduction pipelines."""

import numpy as np
import pytest

from manta import ResidualStatistics, bartlett_hac_residual_statistics


def test_correlated_residuals_expand_white_equivalent_covariance():
    rng = np.random.default_rng(7)
    count = 20_000
    white = rng.normal(size=(count, 3))
    correlated = np.zeros_like(white)
    correlated[0] = white[0]
    for index in range(1, count):
        correlated[index] = 0.8 * correlated[index - 1] + white[index]
    expected_bias = np.array((0.2, -0.1, 0.05))
    correlated += expected_bias

    statistics = bartlett_hac_residual_statistics(
        [correlated], reference_dt_s=0.02, correlation_horizon_s=0.5)

    assert isinstance(statistics, ResidualStatistics)
    np.testing.assert_allclose(statistics.bias, expected_bias, atol=0.05)
    ratio = (np.diag(statistics.white_equivalent_covariance)
             / np.diag(statistics.instantaneous_covariance))
    # AR(1) phi=.8 has asymptotic long-run ratio 9. The finite 25-lag
    # Bartlett window remains well above the independent-sample estimate.
    assert np.all((ratio > 6.0) & (ratio < 10.0))
    assert np.all(statistics.effective_sample_size < count / 6.0)
    assert statistics.correlation_lag_steps == 25
    assert statistics.correlation_horizon_s == pytest.approx(0.5)
    assert statistics.covariance is statistics.white_equivalent_covariance


@pytest.mark.parametrize("dimension", [1, 2, 7])
def test_statistics_are_dimension_generic_owned_and_psd(dimension):
    first = np.arange(12 * dimension, dtype=float).reshape(12, dimension)
    second = -first[:8]
    statistics = bartlett_hac_residual_statistics(
        [first, second], reference_dt_s=0.1, correlation_horizon_s=0.3)

    assert statistics.bias.shape == (dimension,)
    assert statistics.instantaneous_covariance.shape == (dimension, dimension)
    assert statistics.white_equivalent_covariance.shape == (dimension, dimension)
    assert statistics.samples == 20
    assert statistics.windows == 2
    assert np.linalg.eigvalsh(statistics.instantaneous_covariance).min() >= -1e-10
    assert np.linalg.eigvalsh(statistics.white_equivalent_covariance).min() >= -1e-10
    assert not statistics.bias.flags.writeable
    assert not statistics.white_equivalent_covariance.flags.writeable


def test_independent_windows_do_not_create_cross_boundary_lags():
    first = np.array(((1.0,), (-1.0,)))
    second = np.array(((1.0,), (-1.0,)))
    separate = bartlett_hac_residual_statistics(
        [first, second], reference_dt_s=1.0, correlation_horizon_s=1.0)
    concatenated = bartlett_hac_residual_statistics(
        [np.vstack((first, second))],
        reference_dt_s=1.0, correlation_horizon_s=1.0)

    # Concatenation invents one -1 -> +1 boundary transition; independent
    # windows correctly exclude it and therefore produce a different HAC.
    assert (separate.white_equivalent_covariance[0, 0]
            != concatenated.white_equivalent_covariance[0, 0])


@pytest.mark.parametrize(
    ("sequences", "kwargs", "message"),
    [
        ([], {}, "may not be empty"),
        ([np.empty((0, 2))], {}, "non-empty"),
        ([np.ones((2, 2)), np.ones((2, 3))], {}, "share one dimension"),
        ([np.array(((1.0, np.nan),))], {}, "finite"),
        ([np.ones((2, 2))], {"reference_dt_s": 0.0}, "positive"),
        ([np.ones((2, 2))], {"correlation_horizon_s": -1.0},
         "non-negative"),
    ],
)
def test_statistics_refuse_ambiguous_or_invalid_inputs(
        sequences, kwargs, message):
    options = {"reference_dt_s": 0.1, "correlation_horizon_s": 0.2,
               **kwargs}
    with pytest.raises(ValueError, match=message):
        bartlett_hac_residual_statistics(sequences, **options)
