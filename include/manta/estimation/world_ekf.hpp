#pragma once

// WorldEKF — an EKF wired directly against a user's templated Craft class.
//
// The user authors a craft once as a class template:
//
//     template <class Scalar>
//     class MyCraft : public manta::CraftT<Scalar> {
//     public:
//         MyCraft() : manta::CraftT<Scalar>("my_craft") {
//             this->root().template add<manta::parts::PointMassT<Scalar>>(
//                 "body", Scalar(1.0));
//             this->root().compute_params();
//         }
//     };
//
// Then constructs the EKF:
//
//     manta::estimation::WorldEKF<MyCraft, 13, 3> ekf;
//     ekf.predict(dt);
//     ekf.update(z, R);
//
// Internally WorldEKF holds two instances: one with double scalars (the
// value step), one with `ceres::Jet<double, StateDim>` (the Jacobian step).
// Both are default-constructed (the user's craft template must be default-
// constructible, which is the natural case when parts are added in the
// constructor).
//
// State layout: 13-DOF rigid-body state from `CraftT::RigidState`
// (= [px py pz | qw qx qy qz | vx vy vz | wx wy wz]).
//
// Measurement step: the user supplies a templated functor h<S>(craft) that
// returns the predicted measurement. Typically this reads sensor parts'
// last_*() values, which the user updates via set_measurement(...) on the
// est-side craft from the data source.

#include <cmath>
#include <Eigen/Core>
#include <ceres/jet.h>

#include "../core/craft.hpp"
#include "ekf.hpp"

namespace manta::estimation {

template <template <class> class CraftTpl, int MeasDim>
class WorldEKF {
public:
    static constexpr int StateDim = CraftT<double>::kRigidStateDim;
    using Jet      = ceres::Jet<double, StateDim>;
    using CraftR   = CraftTpl<double>;
    using CraftJ   = CraftTpl<Jet>;
    using StateVec = Eigen::Matrix<double, StateDim, 1>;
    using StateCov = Eigen::Matrix<double, StateDim, StateDim>;
    using MeasVec  = Eigen::Matrix<double, MeasDim,  1>;
    using MeasCov  = Eigen::Matrix<double, MeasDim,  MeasDim>;

    WorldEKF() = default;

    // Live access to the underlying crafts. The "real" instance is what the
    // user feeds sensor measurements into via `est.imu().set_measurement(...)`
    // (typical pattern). The "jet" instance is internal scaffolding for
    // Jacobian extraction; the user normally doesn't touch it.
    CraftR&       craft()       noexcept { return craft_real_; }
    const CraftR& craft() const noexcept { return craft_real_; }

    // Register a field on BOTH internal crafts. Use this for any field the
    // est-side parts (e.g. GravityPart) query in their update() — it must
    // be visible to the value-step craft AND the Jacobian-step craft.
    template <typename FieldT>
    void register_field(FieldT& f) {
        craft_real_.register_field(f);
        craft_jet_.register_field(f);
    }

    void set_state(const StateVec& x)      noexcept { ekf_.set_state(x); }
    void set_covariance(const StateCov& P) noexcept { ekf_.set_covariance(P); }

    const StateVec& state()      const noexcept { return ekf_.state(); }
    const StateCov& covariance() const noexcept { return ekf_.covariance(); }

    // Predict step: integrate the craft's dynamics forward by dt with the
    // current EKF state as the input.
    //   - Value step: set state on craft_real_, call evaluate(x, dt) for x_new.
    //   - Jacobian step: set state on craft_jet_ with seeded Jets, call
    //     evaluate, read partials off the output Jets.
    void predict(double dt, const StateCov& Q) {
        // Build the process functor that the EKF expects: f(x, dt) → x_new.
        // We capture a reference to the jet craft so the EKF's predict()
        // can drive it with Jet scalars.
        auto& jet_craft = craft_jet_;
        auto& real_craft = craft_real_;
        auto f = [&jet_craft, &real_craft, dt](const auto& x_arg, double /*dt_*/) {
            using S = typename std::decay_t<decltype(x_arg)>::Scalar;
            // Dispatch on whether S is double (value path) or Jet (Jacobian path).
            if constexpr (std::is_same_v<S, double>) {
                return real_craft.evaluate(x_arg, S(dt));
            } else {
                // S == Jet: route through craft_jet_.
                typename CraftJ::RigidState x_jet;
                for (int i = 0; i < StateDim; ++i) x_jet(i) = x_arg(i);
                auto y_jet = jet_craft.evaluate(x_jet, S(dt));
                Eigen::Matrix<S, StateDim, 1> y;
                for (int i = 0; i < StateDim; ++i) y(i) = y_jet(i);
                return y;
            }
        };
        ekf_.predict(f, dt, Q);
    }

    // Update step: user supplies a templated measurement functor.
    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        ekf_.update(h, z, R);
    }

    // Per-sensor update: lets a single WorldEKF absorb measurements of
    // varying width N (e.g. DVL=3, IMU=6, Mag=3) without instantiating
    // separate filters. Codegen drives this from
    //   if (ekf.craft().sensor().consume_fresh()) ekf.update_n<N>(h, z, R);
    // patterns. Mathematically equivalent to a single fused update when
    // sensors' R blocks are independent.
    template <int N, class MeasureH>
    void update_n(const MeasureH& h,
                  const Eigen::Matrix<double, N, 1>& z,
                  const Eigen::Matrix<double, N, N>& R) {
        ekf_.template update_n<N>(h, z, R);
    }

    // Codegen-friendly accessors for the rigid-state slices. Output
    // BoundSignals on the Python EKF descriptor read these via
    //     ekf_<id>.position()(i)  /  .orientation()(i)  /  .vel_linear()(i)
    // / .vel_angular()(i) / .full_state()(i)
    // and the corresponding `_stddev()` variants.
    Eigen::Matrix<double, 3, 1> position() const noexcept {
        return ekf_.state().template segment<3>(0);
    }
    Eigen::Matrix<double, 4, 1> orientation() const noexcept {
        return ekf_.state().template segment<4>(3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear() const noexcept {
        return ekf_.state().template segment<3>(7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular() const noexcept {
        return ekf_.state().template segment<3>(10);
    }
    const StateVec& full_state() const noexcept { return ekf_.state(); }

    Eigen::Matrix<double, 3, 1> position_stddev() const noexcept {
        return diag_stddev_segment<3>(0);
    }
    Eigen::Matrix<double, 4, 1> orientation_stddev() const noexcept {
        return diag_stddev_segment<4>(3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev() const noexcept {
        return diag_stddev_segment<3>(7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev() const noexcept {
        return diag_stddev_segment<3>(10);
    }

private:
    template <int N>
    Eigen::Matrix<double, N, 1> diag_stddev_segment(int start) const noexcept {
        Eigen::Matrix<double, N, 1> out;
        const auto& P = ekf_.covariance();
        for (int i = 0; i < N; ++i) {
            const double v = P(start + i, start + i);
            out(i) = std::sqrt(v > 0.0 ? v : 0.0);
        }
        return out;
    }

    CraftR              craft_real_;
    CraftJ              craft_jet_;
    EKF<StateDim, MeasDim> ekf_;
};

} // namespace manta::estimation
