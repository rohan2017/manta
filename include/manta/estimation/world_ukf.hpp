#pragma once

// `manta::estimation::UKF` — a UKF wired against a user's
// `manta::WorldT<double>` (the sim/Real World). Estimates the joint state
// of every Craft in the World via sigma-point propagation. State layout
// matches the manta-aware EKF:
//
//   [c0.p (3) | c0.q (4) | c0.v (3) | c0.ω (3) |
//    c1.p (3) | c1.q (4) | c1.v (3) | c1.ω (3) | ... ]
//
//   StateDim = 13 * NumCrafts
//
// Compared to the manta-aware EKF, UKF has a notable structural advantage:
// it does NOT need a Jet-shadow World. Every sigma point is a vector of
// `double`s, propagated through the same Real `WorldT<double>`. The
// craft scalar can be either `double` (templated craft) or a non-
// templated `manta::Craft` — both work as long as `evaluate(state, dt)`
// is available.
//
// Trade-offs vs EKF:
//   * No autodiff, no Jet path — fewer compile-time templates, faster
//     builds, parts that aren't Scalar-templated still work.
//   * 2*StateDim+1 forward evaluations of the entire World per predict;
//     EKF does one Jet-instantiated update. For NumCrafts crafts each
//     with ~5 parts, the ratio depends on which is cheaper; both scale
//     well to roughly 10 crafts.
//   * Captures nonlinearity to second order; EKF only to first order.
//
// State layout, quaternion handling, and the bind-based API mirror the
// manta-aware EKF — see world_ekf.hpp for the design notes that apply to
// both.

#include <array>
#include <cmath>

#include <Eigen/Core>

#include "../core/craft.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "ukf.hpp"

namespace manta::estimation {

template <int NumCrafts, int MeasDim>
class UKF {
    static_assert(NumCrafts >= 1, "manta::estimation::UKF needs at least one craft");

public:
    static constexpr int kCrafts        = NumCrafts;
    static constexpr int kRigidStateDim = 13;
    static constexpr int kStateDim      = NumCrafts * kRigidStateDim;

    using WorldR   = WorldT<double>;
    using CraftR   = CraftT<double>;

    using StateVec = Eigen::Matrix<double, kStateDim, 1>;
    using StateCov = Eigen::Matrix<double, kStateDim, kStateDim>;
    using MeasVec  = Eigen::Matrix<double, MeasDim,   1>;
    using MeasCov  = Eigen::Matrix<double, MeasDim,   MeasDim>;

    explicit UKF(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : ukf_(alpha, beta, kappa) {}

    // Bind to the Real World + craft pointers in slot order matching the
    // state vector. Unlike the EKF, no Jet shadow is needed — sigma
    // propagation runs through the same Real world.
    void bind(WorldR& w_real,
              std::array<CraftR*, NumCrafts> real_crafts) noexcept {
        w_real_      = &w_real;
        crafts_real_ = real_crafts;
    }

    void set_state(const StateVec& x)        noexcept { ukf_.set_state(x); }
    void set_covariance(const StateCov& P)   noexcept { ukf_.set_covariance(P); }

    const StateVec& state()      const noexcept { return ukf_.state(); }
    const StateCov& covariance() const noexcept { return ukf_.covariance(); }

    // Propagate each sigma point through the entire Real World's update.
    // Saves/restores craft state around each propagation so the World
    // ends in the post-mean-state configuration after predict returns.
    void predict(double dt, const StateCov& Q) {
        if (!w_real_) return;
        w_real_->clock().set_dt(static_cast<float>(dt));

        // Save initial craft states so we can restore between sigma points.
        std::array<typename CraftR::RigidState, NumCrafts> saved;
        for (int k = 0; k < NumCrafts; ++k) {
            saved[k] = crafts_real_[k]->get_rigid_state();
        }

        auto f = [this, &saved](const StateVec& x_in, double /*dt*/) -> StateVec {
            // Reset every craft from its saved pre-predict state, then
            // overwrite with this sigma point's values. This avoids
            // residual cross-contamination between sigma evaluations
            // (otherwise integrators reading stale state from the
            // previous sigma would leak across).
            for (int k = 0; k < NumCrafts; ++k) {
                crafts_real_[k]->set_rigid_state(saved[k]);
                typename CraftR::RigidState xk =
                    x_in.template segment<kRigidStateDim>(kRigidStateDim * k);
                crafts_real_[k]->set_rigid_state(xk);
            }
            w_real_->step();
            StateVec y;
            for (int k = 0; k < NumCrafts; ++k) {
                y.template segment<kRigidStateDim>(kRigidStateDim * k) =
                    crafts_real_[k]->get_rigid_state();
            }
            return y;
        };
        ukf_.predict(f, dt, Q);

        // Mirror the post-predict mean state back into the Real crafts so
        // downstream sensor reads, telemetry, and `set_measurement` calls
        // see the latest belief.
        for (int k = 0; k < NumCrafts; ++k) {
            typename CraftR::RigidState xk =
                ukf_.state().template segment<kRigidStateDim>(kRigidStateDim * k);
            crafts_real_[k]->set_rigid_state(xk);
        }
    }

    template <int N, class MeasureH>
    void update_n(const MeasureH& h,
                  const Eigen::Matrix<double, N, 1>& z,
                  const Eigen::Matrix<double, N, N>& R) {
        ukf_.template update_n<N>(h, z, R);
        // Mirror updated mean state.
        for (int k = 0; k < NumCrafts; ++k) {
            if (!crafts_real_[k]) continue;
            typename CraftR::RigidState xk =
                ukf_.state().template segment<kRigidStateDim>(kRigidStateDim * k);
            crafts_real_[k]->set_rigid_state(xk);
        }
    }

    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        update_n<MeasDim>(h, z, R);
    }

    // ---- Per-craft slice accessors (mirror EKF) ----
    Eigen::Matrix<double, 3, 1> position(int craft_idx = 0) const noexcept {
        return ukf_.state().template segment<3>(craft_idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 4, 1> orientation(int craft_idx = 0) const noexcept {
        return ukf_.state().template segment<4>(craft_idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear(int craft_idx = 0) const noexcept {
        return ukf_.state().template segment<3>(craft_idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular(int craft_idx = 0) const noexcept {
        return ukf_.state().template segment<3>(craft_idx * kRigidStateDim + 10);
    }
    const StateVec& full_state() const noexcept { return ukf_.state(); }

    Eigen::Matrix<double, 3, 1> position_stddev(int craft_idx = 0) const noexcept {
        return diag_stddev_segment<3>(craft_idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 4, 1> orientation_stddev(int craft_idx = 0) const noexcept {
        return diag_stddev_segment<4>(craft_idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev(int craft_idx = 0) const noexcept {
        return diag_stddev_segment<3>(craft_idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev(int craft_idx = 0) const noexcept {
        return diag_stddev_segment<3>(craft_idx * kRigidStateDim + 10);
    }

    CraftR&       craft(int idx = 0)       noexcept { return *crafts_real_[idx]; }
    const CraftR& craft(int idx = 0) const noexcept { return *crafts_real_[idx]; }

private:
    template <int N>
    Eigen::Matrix<double, N, 1> diag_stddev_segment(int start) const noexcept {
        Eigen::Matrix<double, N, 1> out;
        const auto& P = ukf_.covariance();
        for (int i = 0; i < N; ++i) {
            const double v = P(start + i, start + i);
            out(i) = std::sqrt(v > 0.0 ? v : 0.0);
        }
        return out;
    }

    WorldR*                          w_real_      = nullptr;
    std::array<CraftR*, NumCrafts>   crafts_real_{};
    UKFKernel<kStateDim, MeasDim>    ukf_;
};

} // namespace manta::estimation
