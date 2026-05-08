// Tests for the unscented Kalman filter (manta::estimation::UKF) and its
// craft-aware wrapper (manta::estimation::UKF).
//
// UKF mirrors EKF's API but uses sigma-point propagation instead of Jet
// autodiff Jacobians, so the user functors only need to accept double.

#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/estimation/ukf_kernel.hpp"
#include "../include/manta/core/craft.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::estimation;

// ---- Plain UKF on a 2-state constant-velocity model ----

namespace {
struct ConstVelProcess {
    Eigen::Matrix<double, 2, 1> operator()(const Eigen::Matrix<double, 2, 1>& x,
                                           double dt) const {
        Eigen::Matrix<double, 2, 1> y;
        y << x(0) + x(1) * dt, x(1);
        return y;
    }
};

struct PositionMeas1D {
    Eigen::Matrix<double, 1, 1> operator()(const Eigen::Matrix<double, 2, 1>& x) const {
        return Eigen::Matrix<double, 1, 1>{x(0)};
    }
};
}  // namespace

TEST_CASE("UKF: weights sum to 1 (mean) and match standard formulation") {
    UKFKernel<2, 1> ukf;          // default alpha=1e-3, beta=2, kappa=0
    auto wm = ukf.weights_mean();
    auto wc = ukf.weights_covariance();
    // Default α=1e-3 produces weights with magnitude ~1e6, so the analytical
    // sum-to-1 carries large floating-point cancellation. 1e-6 tolerance is
    // appropriate; tighter would test FP arithmetic, not the algorithm.
    CHECK(wm.sum() == doctest::Approx(1.0).epsilon(1e-6));
    // wc = wm + (1 - α² + β) at index 0 → sum_wc = sum_wm + (1 - α² + β)
    //                                              = 1 + 1 - α² + β ≈ 2 + β = 4.
    CHECK(wc.sum() == doctest::Approx(2.0 + 2.0).epsilon(1e-6));
}

TEST_CASE("UKF: linear system reproduces Kalman update (predict+update)") {
    // Constant-velocity 1D, perfectly linear. Sigma-point propagation should
    // produce the same answer as the linear KF.
    UKFKernel<2, 1> ukf(/*alpha=*/1.0, /*beta=*/0.0, /*kappa=*/2.0);
    Eigen::Matrix<double, 2, 1> x0; x0 << 0.0, 1.0;     // p=0, v=1
    Eigen::Matrix<double, 2, 2> P0 = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;
    ukf.set_state(x0);
    ukf.set_covariance(P0);

    const double dt = 0.1;
    Eigen::Matrix<double, 2, 2> Q = Eigen::Matrix<double, 2, 2>::Identity() * 1e-6;
    Eigen::Matrix<double, 1, 1> R; R(0, 0) = 1e-2;

    for (int k = 0; k < 50; ++k) ukf.predict(ConstVelProcess{}, dt, Q);

    // After 50 steps of dt=0.1 starting at (0,1), expected position ≈ 5.
    CHECK(ukf.state()(0) == doctest::Approx(5.0).epsilon(1e-6));
    CHECK(ukf.state()(1) == doctest::Approx(1.0).epsilon(1e-6));

    // Position measurement at z=4.5 should pull p toward 4.5 and reduce P.
    auto P_before = ukf.covariance()(0, 0);
    Eigen::Matrix<double, 1, 1> z; z << 4.5;
    ukf.update(PositionMeas1D{}, z, R);
    CHECK(ukf.state()(0) > 4.5);
    CHECK(ukf.state()(0) < 5.0);
    CHECK(ukf.covariance()(0, 0) < P_before);
}

TEST_CASE("UKF: covariance stays symmetric after update") {
    UKFKernel<2, 1> ukf;
    Eigen::Matrix<double, 2, 1> x0; x0 << 0.0, 1.0;
    Eigen::Matrix<double, 2, 2> P0;
    P0 << 0.5, 0.1, 0.1, 0.3;
    ukf.set_state(x0);
    ukf.set_covariance(P0);

    Eigen::Matrix<double, 1, 1> z; z << 0.0;
    Eigen::Matrix<double, 1, 1> R; R(0, 0) = 0.01;
    ukf.update(PositionMeas1D{}, z, R);

    auto P = ukf.covariance();
    CHECK(P(0, 1) == doctest::Approx(P(1, 0)).epsilon(1e-12));
}

// ---- UKF on a 13-state free-body craft ----

template <class Scalar>
class FreeBodyCraftU : public manta::CraftT<Scalar> {
public:
    FreeBodyCraftU() : manta::CraftT<Scalar>("ukf_test") {
        this->root().template add<manta::parts::MassT<Scalar>>("body", Scalar(1.0));
        this->root().compute_params();
    }
};

namespace {
struct PositionMeas3D_U {
    Eigen::Matrix<double, 3, 1> operator()(const Eigen::Matrix<double, 13, 1>& x) const {
        return x.segment<3>(0);
    }
};
}  // namespace

// Legacy UKF<NumCrafts, ...>-wrapper tests removed in Phase 6.
// UKFGeneric end-to-end coverage lives in test_generic_ukf.cpp.
