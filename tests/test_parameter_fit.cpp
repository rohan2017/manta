// Unit tests for the ParameterFit wrapper. The two `tests/test_system_id.cpp`
// cases use the wrapper end-to-end on real Craft rollouts; this file
// exercises the wrapper's surface area (multi-param fits, bounds clamping,
// success / failure reporting) on a closed-form analytical model that's
// fast and predictable.
//
// The model:
//     y_i = a · x_i² + b · x_i + c     for i = 0..N-1
// Three parameters [a, b, c]. Truth: a=1.5, b=-2.0, c=0.5.

#include <array>
#include <cmath>
#include <vector>

#include <ceres/ceres.h>
#include <doctest/doctest.h>

#include "../include/manta/estimation/parameter_fit.hpp"

namespace {

struct QuadraticFitCost {
    std::vector<double> x;
    std::vector<double> y;

    template <class T>
    bool operator()(const T* p, T* residual) const {
        const T& a = p[0];
        const T& b = p[1];
        const T& c = p[2];
        for (size_t i = 0; i < x.size(); ++i) {
            T xi = T(x[i]);
            residual[i] = (a * xi * xi + b * xi + c) - T(y[i]);
        }
        return true;
    }
};

}  // namespace

TEST_CASE("ParameterFit: 3-param quadratic recovery, no bounds") {
    constexpr double TRUE_A = 1.5, TRUE_B = -2.0, TRUE_C = 0.5;
    constexpr int    N = 21;

    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = -1.0 + 0.1 * i;
        y[i] = TRUE_A * x[i] * x[i] + TRUE_B * x[i] + TRUE_C;
    }

    manta::estimation::ParameterFit<3> fit;
    auto result = fit.solve_dynamic(
        /*initial_guess=*/{0.0, 0.0, 0.0},
        new QuadraticFitCost{x, y},
        /*n_residuals=*/N);

    CHECK(result.success);
    CHECK(result.params[0] == doctest::Approx(TRUE_A).epsilon(1e-6));
    CHECK(result.params[1] == doctest::Approx(TRUE_B).epsilon(1e-6));
    CHECK(result.params[2] == doctest::Approx(TRUE_C).epsilon(1e-6));
}

TEST_CASE("ParameterFit: lower bound clamps a parameter that wants to go below it") {
    // Same truth as above, but constrain `a >= 2.0`. Truth `a = 1.5` is
    // unreachable, so the optimizer should hit the bound and stop there.
    constexpr int N = 21;
    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = -1.0 + 0.1 * i;
        y[i] = 1.5 * x[i] * x[i] - 2.0 * x[i] + 0.5;
    }

    manta::estimation::ParameterFit<3> fit;
    fit.set_lower_bound(0, 2.0);

    auto result = fit.solve_dynamic(
        /*initial_guess=*/{5.0, 0.0, 0.0},
        new QuadraticFitCost{x, y},
        /*n_residuals=*/N);

    CHECK(result.success);
    // Bound should be active. Exactly equal to the bound is the expected
    // behavior for a well-posed unimodal residual landscape.
    CHECK(result.params[0] >= 2.0 - 1e-9);
    CHECK(result.params[0] == doctest::Approx(2.0).epsilon(1e-3));
}

TEST_CASE("ParameterFit: static-residual-count solve<NRes> path") {
    // Compile-time residual count — Ceres dispatches a more efficient
    // autodiff inner loop. Verify the alternate API works.
    constexpr int N = 5;
    std::vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = -1.0 + 0.5 * i;
        y[i] = 1.5 * x[i] * x[i] - 2.0 * x[i] + 0.5;
    }

    manta::estimation::ParameterFit<3> fit;
    auto result = fit.solve<N>(
        /*initial_guess=*/{0.0, 0.0, 0.0},
        new QuadraticFitCost{x, y});

    CHECK(result.success);
    CHECK(result.params[0] == doctest::Approx(1.5).epsilon(1e-6));
    CHECK(result.params[1] == doctest::Approx(-2.0).epsilon(1e-6));
    CHECK(result.params[2] == doctest::Approx(0.5).epsilon(1e-6));
}

TEST_CASE("ParameterFit: brief_report is non-empty on a successful solve") {
    constexpr int N = 5;
    std::vector<double> x(N, 0.0), y(N, 0.0);
    for (int i = 0; i < N; ++i) { x[i] = i; y[i] = 2.0 * i + 1.0; }   // a=0, b=2, c=1

    manta::estimation::ParameterFit<3> fit;
    auto result = fit.solve_dynamic(
        /*initial_guess=*/{0.0, 0.0, 0.0},
        new QuadraticFitCost{x, y},
        /*n_residuals=*/N);

    CHECK(result.success);
    CHECK(!result.brief_report.empty());
}
