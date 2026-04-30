#pragma once

// Extended Kalman Filter — header-only template using Ceres Jets for forward-
// mode autodiff Jacobians.
//
// Usage sketch:
//
//   struct ConstVelProcess {
//     template <class S>
//     Eigen::Matrix<S, 2, 1> operator()(const Eigen::Matrix<S, 2, 1>& x,
//                                       double dt) const {
//       Eigen::Matrix<S, 2, 1> y;
//       y << x(0) + x(1) * S(dt),
//            x(1);
//       return y;
//     }
//   };
//
//   struct PositionMeas {
//     template <class S>
//     Eigen::Matrix<S, 1, 1> operator()(const Eigen::Matrix<S, 2, 1>& x) const {
//       return Eigen::Matrix<S, 1, 1>{x(0)};
//     }
//   };
//
//   manta::estimation::EKF<2, 1> ekf;
//   ekf.set_state(...);
//   ekf.set_covariance(...);
//   ekf.predict(ConstVelProcess{}, dt, Q);
//   ekf.update(PositionMeas{},  z, R);
//
// The user-supplied functors must be templated on the scalar so the same
// callable accepts double (for the value step) and ceres::Jet<double, N>
// (for the Jacobian step). MeasDim is fixed per filter instance.
//
// All math is double-precision internally. Single-precision (float) is not
// the right call for filter covariance updates — numerical conditioning
// matters, and the cost of double over float is negligible compared to the
// autodiff overhead.

#include <Eigen/Core>
#include <Eigen/LU>
#include <ceres/jet.h>

namespace manta::estimation {

template <int StateDim, int MeasDim>
class EKF {
public:
    using StateVec       = Eigen::Matrix<double, StateDim, 1>;
    using StateCov       = Eigen::Matrix<double, StateDim, StateDim>;
    using MeasVec        = Eigen::Matrix<double, MeasDim, 1>;
    using MeasCov        = Eigen::Matrix<double, MeasDim,  MeasDim>;
    using MeasJacobian   = Eigen::Matrix<double, MeasDim,  StateDim>;

    EKF() : x_(StateVec::Zero()), P_(StateCov::Identity()) {}

    void set_state(const StateVec& x)    noexcept { x_ = x; }
    void set_covariance(const StateCov& P) noexcept { P_ = P; }

    const StateVec& state()      const noexcept { return x_; }
    const StateCov& covariance() const noexcept { return P_; }

    // Predict step: x_new = f(x, dt), P_new = F P F^T + Q.
    // F (the process Jacobian df/dx) is derived automatically from `f` via
    // Ceres Jets. The user's functor must be a template on the scalar type.
    template <class ProcessF>
    void predict(const ProcessF& f, double dt, const StateCov& Q) {
        // Compute the Jacobian at the INPUT state (before advancing x).
        using Jet = ceres::Jet<double, StateDim>;
        Eigen::Matrix<Jet, StateDim, 1> x_jet;
        for (int i = 0; i < StateDim; ++i) {
            x_jet(i) = Jet(x_(i), i);   // value = x, derivatives = e_i
        }
        Eigen::Matrix<Jet, StateDim, 1> y_jet = f(x_jet, dt);

        StateVec x_new;
        StateCov F;
        for (int i = 0; i < StateDim; ++i) {
            x_new(i) = y_jet(i).a;
            for (int j = 0; j < StateDim; ++j) {
                F(i, j) = y_jet(i).v[j];
            }
        }
        x_ = x_new;
        P_ = F * P_ * F.transpose() + Q;
    }

    // Update step: innovation y = z - h(x), S = H P H^T + R, K = P H^T S^-1,
    // x += K y, P = (I - K H) P. H = dh/dx via Ceres Jets.
    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        using Jet = ceres::Jet<double, StateDim>;
        Eigen::Matrix<Jet, StateDim, 1> x_jet;
        for (int i = 0; i < StateDim; ++i) {
            x_jet(i) = Jet(x_(i), i);
        }
        Eigen::Matrix<Jet, MeasDim, 1> z_jet = h(x_jet);

        MeasVec      z_pred;
        MeasJacobian H;
        for (int i = 0; i < MeasDim; ++i) {
            z_pred(i) = z_jet(i).a;
            for (int j = 0; j < StateDim; ++j) {
                H(i, j) = z_jet(i).v[j];
            }
        }

        MeasVec y = z - z_pred;
        MeasCov S = H * P_ * H.transpose() + R;
        Eigen::Matrix<double, StateDim, MeasDim> K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        P_ = (StateCov::Identity() - K * H) * P_;
        // Symmetrize P to fight numerical drift.
        P_ = 0.5 * (P_ + P_.transpose().eval());
    }

private:
    StateVec x_;
    StateCov P_;
};

} // namespace manta::estimation
