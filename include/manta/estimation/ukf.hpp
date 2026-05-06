#pragma once

// Pure-math Unscented Kalman Filter — header-only template using sigma-point
// propagation. This is the kernel; the manta-aware filter wrapper that knows
// about Crafts/Worlds is `manta::estimation::UKF` (declared in
// `world_ukf.hpp`). Unlike EKFKernel, UKFKernel does NOT require autodiff:
// the user's process and measurement functors only need to accept double
// scalars.
//
// Tradeoffs vs EKF:
//   * No Jacobians, so no Jet instantiation of user code → fewer compile-time
//     templates, faster builds, and parts that aren't Scalar-templated still
//     work (UKF only needs CraftT<double>).
//   * Captures nonlinearity to second order; EKF only to first order. Better
//     accuracy when local linearization breaks (large covariance, sharp
//     dynamics).
//   * 2*StateDim+1 process evaluations per predict; EKF does one evaluation
//     plus one Jet pass. Cost depends on which is cheaper for your craft.
//
// Tuning knobs (alpha, beta, kappa) follow the standard formulation:
//   * alpha controls sigma-point spread around the mean. Small (1e-3) keeps
//     points close.
//   * beta = 2 is optimal for Gaussian state distributions.
//   * kappa is a secondary scaling; default 0.
//
// Usage sketch — same functor signatures as EKF except the templated scalar
// is always double:
//
//   manta::estimation::UKFKernel<2, 1> ukf;
//   ukf.set_state(...); ukf.set_covariance(...);
//   ukf.predict(ConstVelProcess{}, dt, Q);
//   ukf.update(PositionMeas{},  z, R);

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace manta::estimation {

template <int StateDim, int MeasDim>
class UKFKernel {
public:
    static constexpr int NSigma = 2 * StateDim + 1;

    using StateVec     = Eigen::Matrix<double, StateDim, 1>;
    using StateCov     = Eigen::Matrix<double, StateDim, StateDim>;
    using MeasVec      = Eigen::Matrix<double, MeasDim,  1>;
    using MeasCov      = Eigen::Matrix<double, MeasDim,  MeasDim>;
    using SigmaMatrix  = Eigen::Matrix<double, StateDim, NSigma>;
    using MeasSigma    = Eigen::Matrix<double, MeasDim,  NSigma>;
    using WeightVec    = Eigen::Matrix<double, NSigma,   1>;

    explicit UKFKernel(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0) noexcept
        : x_(StateVec::Zero())
        , P_(StateCov::Identity())
        , alpha_(alpha), beta_(beta), kappa_(kappa) {
        recompute_weights();
    }

    void set_state(const StateVec& x)      noexcept { x_ = x; }
    void set_covariance(const StateCov& P) noexcept { P_ = P; }

    const StateVec& state()      const noexcept { return x_; }
    const StateCov& covariance() const noexcept { return P_; }

    // Predict: propagate sigma points through f, recover mean & covariance.
    template <class ProcessF>
    void predict(const ProcessF& f, double dt, const StateCov& Q) {
        SigmaMatrix chi = sigma_points(x_, P_);

        // Push each column through the process.
        SigmaMatrix chi_pred;
        for (int i = 0; i < NSigma; ++i) {
            StateVec xi = chi.col(i);
            chi_pred.col(i) = f(xi, dt);
        }

        // Recover mean.
        StateVec x_new = StateVec::Zero();
        for (int i = 0; i < NSigma; ++i) x_new += wm_(i) * chi_pred.col(i);

        // Recover covariance.
        StateCov P_new = Q;
        for (int i = 0; i < NSigma; ++i) {
            StateVec d = chi_pred.col(i) - x_new;
            P_new += wc_(i) * (d * d.transpose());
        }

        x_ = x_new;
        P_ = 0.5 * (P_new + P_new.transpose());   // symmetrize
    }

    // Update: propagate sigma points through h, build cross-covariance.
    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        update_n<MeasDim>(h, z, R);
    }

    // Update step with a per-call measurement dimension N. Lets a single
    // UKF instance absorb measurements of varying width — the standard
    // sequential per-sensor update pattern. Mathematically equivalent to a
    // fused update when per-sensor R blocks are independent.
    template <int N, class MeasureH>
    void update_n(const MeasureH& h,
                  const Eigen::Matrix<double, N, 1>& z,
                  const Eigen::Matrix<double, N, N>& R) {
        SigmaMatrix chi = sigma_points(x_, P_);

        Eigen::Matrix<double, N, NSigma> zeta;
        for (int i = 0; i < NSigma; ++i) {
            StateVec xi = chi.col(i);
            zeta.col(i) = h(xi);
        }

        Eigen::Matrix<double, N, 1> z_pred = Eigen::Matrix<double, N, 1>::Zero();
        for (int i = 0; i < NSigma; ++i) z_pred += wm_(i) * zeta.col(i);

        Eigen::Matrix<double, N, N>        Pzz = R;
        Eigen::Matrix<double, StateDim, N> Pxz =
            Eigen::Matrix<double, StateDim, N>::Zero();
        for (int i = 0; i < NSigma; ++i) {
            Eigen::Matrix<double, N, 1> dz = zeta.col(i) - z_pred;
            StateVec                    dx = chi.col(i)  - x_;
            Pzz += wc_(i) * (dz * dz.transpose());
            Pxz += wc_(i) * (dx * dz.transpose());
        }

        Eigen::Matrix<double, StateDim, N> K = Pxz * Pzz.inverse();
        x_ = x_ + K * (z - z_pred);
        StateCov P_new = P_ - K * Pzz * K.transpose();
        P_ = 0.5 * (P_new + P_new.transpose());   // symmetrize
    }

    // Sigma points: chi_0 = x; chi_i = x + L_i; chi_{i+N} = x - L_i,
    // where L = sqrt((N + lambda) * P) via Cholesky.
    SigmaMatrix sigma_points(const StateVec& x, const StateCov& P) const {
        const double scale = static_cast<double>(StateDim) + lambda_;
        Eigen::LLT<StateCov> llt(scale * P);
        StateCov L = llt.matrixL();   // lower-triangular: L L^T = scale*P

        SigmaMatrix chi;
        chi.col(0) = x;
        for (int i = 0; i < StateDim; ++i) {
            chi.col(1 + i)            = x + L.col(i);
            chi.col(1 + StateDim + i) = x - L.col(i);
        }
        return chi;
    }

    const WeightVec& weights_mean()       const noexcept { return wm_; }
    const WeightVec& weights_covariance() const noexcept { return wc_; }

private:
    void recompute_weights() noexcept {
        lambda_ = alpha_ * alpha_ * (StateDim + kappa_) - StateDim;
        const double denom = static_cast<double>(StateDim) + lambda_;
        const double w_other = 1.0 / (2.0 * denom);
        wm_(0) = lambda_ / denom;
        wc_(0) = wm_(0) + (1.0 - alpha_ * alpha_ + beta_);
        for (int i = 1; i < NSigma; ++i) {
            wm_(i) = w_other;
            wc_(i) = w_other;
        }
    }

    StateVec  x_;
    StateCov  P_;
    double    alpha_, beta_, kappa_;
    double    lambda_ = 0.0;
    WeightVec wm_;
    WeightVec wc_;
};

} // namespace manta::estimation
