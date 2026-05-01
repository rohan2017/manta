#pragma once

// CraftUKF — a UKF wired directly against a user's Craft.
//
// Compared to CraftEKF, CraftUKF has a notable structural advantage: it
// does NOT require the craft to be Scalar-templated, because the unscented
// transform doesn't compute Jacobians via Jets. The user can pass either
// a templated `MyCraftT<double>` (same authoring style as CraftEKF) or a
// plain non-templated `manta::Craft` subclass — both work as long as the
// craft exposes `evaluate(state, dt)` returning a 13-DOF rigid state.
//
// API mirrors CraftEKF as closely as possible so callers can swap
// estimators by changing the type:
//
//   manta::estimation::CraftUKF<MyCraft, 3> ukf;            // 3-DoF measurement
//   ukf.predict(dt, Q);
//   ukf.update(h, z, R);
//
// Internally CraftUKF holds a single CraftR (default-constructed) and a
// UKF<13, MeasDim>. The 2N+1 sigma points are propagated by repeatedly
// calling the same craft's evaluate() — cheap if evaluate() is cheap.
//
// State layout: 13-DOF rigid-body state from `CraftT::RigidState`.
//
// Caveat: the rigid state lives on R^7 × R^6 (with a quaternion in slots
// 3–6), not pure R^13. We do straight R^13 sigma-point arithmetic and rely
// on `CraftT::set_rigid_state` to renormalize the quaternion at each
// evaluate() call. For small covariances and reasonable sigma spreads this
// is fine; it's the same compromise EKF makes.

#include <type_traits>

#include <Eigen/Core>

#include "../core/craft.hpp"
#include "ukf.hpp"

namespace manta::estimation {

// Generic form: CraftUKF takes the concrete craft type directly. Works with
// both templated (MyCraftT<double>) and non-templated (PlainCraft) crafts.
template <class CraftType, int MeasDim>
class CraftUKFOf {
public:
    static constexpr int StateDim = CraftType::kRigidStateDim;
    using CraftR   = CraftType;
    using StateVec = Eigen::Matrix<double, StateDim, 1>;
    using StateCov = Eigen::Matrix<double, StateDim, StateDim>;
    using MeasVec  = Eigen::Matrix<double, MeasDim,  1>;
    using MeasCov  = Eigen::Matrix<double, MeasDim,  MeasDim>;

    explicit CraftUKFOf(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : ukf_(alpha, beta, kappa) {}

    CraftR&       craft()       noexcept { return craft_; }
    const CraftR& craft() const noexcept { return craft_; }

    // Register a field on the underlying craft. UKF only has one craft
    // instance (no Jet shadow) so this is a thin pass-through.
    template <typename FieldT>
    void register_field(FieldT& f) { craft_.register_field(f); }

    void set_state(const StateVec& x)      noexcept { ukf_.set_state(x); }
    void set_covariance(const StateCov& P) noexcept { ukf_.set_covariance(P); }

    const StateVec& state()      const noexcept { return ukf_.state(); }
    const StateCov& covariance() const noexcept { return ukf_.covariance(); }

    // Propagate each sigma point through craft.evaluate(x, dt). If the craft's
    // internal Scalar isn't `double` (e.g. Craft = CraftT<float>), cast at the
    // boundary — UKF works in double for numerical conditioning regardless.
    void predict(double dt, const StateCov& Q) {
        auto& craft = craft_;
        using CraftScalar = typename CraftR::RigidState::Scalar;
        auto f = [&craft, dt](const StateVec& x_in, double /*dt_*/) -> StateVec {
            if constexpr (std::is_same_v<CraftScalar, double>) {
                return craft.evaluate(x_in, dt);
            } else {
                Eigen::Matrix<CraftScalar, StateDim, 1> x_native = x_in.template cast<CraftScalar>();
                auto y_native = craft.evaluate(x_native, CraftScalar(dt));
                return y_native.template cast<double>();
            }
        };
        ukf_.predict(f, dt, Q);
    }

    // Measurement step: user-supplied functor h(x) → predicted measurement.
    // Unlike EKF, this does NOT need to be templated on Scalar — pure double.
    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        ukf_.update(h, z, R);
    }

private:
    CraftR              craft_;
    UKF<StateDim, MeasDim> ukf_;
};

// Convenience alias: same shape as CraftEKF<MyCraftTemplate, MeasDim> for
// templated craft types. Picks the double instantiation.
template <template <class> class CraftTpl, int MeasDim>
using CraftUKF = CraftUKFOf<CraftTpl<double>, MeasDim>;

} // namespace manta::estimation
