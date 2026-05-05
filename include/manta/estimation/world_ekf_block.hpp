#pragma once

// WorldEKFBlockDecomposed — block-decomposed EKF for swarms of decoupled
// crafts.
//
// For NumCrafts ≥ ~5 the monolithic `WorldEKF<NumCrafts, MeasDim>`
// becomes expensive: every Jet operation tracks `13·NumCrafts` partial
// derivatives, so a Jet world pass costs O(NumCrafts²) work per part-
// tick. For 20 crafts that's 400× the single-craft cost.
//
// When the crafts don't physically couple — no tether, contact, fluid
// coupling, or shared sensor — the process Jacobian F is block
// diagonal:
//
//     ∂x_i_post/∂x_j_pre = 0   for i ≠ j
//
// and we can compute each block independently. This class runs
// NumCrafts Jet passes per tick, each with `Jet<double, 13>` (one
// craft's worth of partials):
//
//   for k in 0..NumCrafts:
//     seed craft k's state with identity-derivative Jets
//     seed every other craft j ≠ k with value-only Jets (zero derivs)
//     w_jet.evaluate()
//     for each measurement on craft k:
//         read h_jet from craft k's sensors → store h_pre + H[meas, k-cols]
//     w_jet.advance_only(dt)
//     read craft k's post-integrate state → x_k_post + F[k-rows, k-cols]
//
// The off-diagonal F blocks stay at zero. Cost: NumCrafts ×
// O(parts · 13²) instead of 1 × O(parts · (13·NumCrafts)²) — linear
// in NumCrafts instead of quadratic.
//
// Coupling caveat: this only produces the correct Jacobian when crafts
// are dynamically independent. When craft j is "inactive" in pass k,
// its state has zero Jet derivatives, so any contribution to craft k's
// dynamics through cross-craft physics goes uncomputed (the Jacobian
// row would be zero where it should be non-zero). Tethers, contact,
// fluid drag from a neighbor's wake — all unmodeled here. Use the
// monolithic WorldEKF for those cases.
//
// Usage mirrors WorldEKF but with explicit per-craft brackets:
//
//   ekf.begin_step(dt, Q);
//   for (int k = 0; k < NumCrafts; ++k) {
//       ekf.begin_craft(k);                                     // seed + evaluate
//       if (sensor_k.consume_fresh())
//           ekf.add_update<N>(h_at_pre, z, R);                  // reads Jet h
//       ekf.end_craft();                                         // advance, capture F_kk
//   }
//   ekf.end_step();                                              // assemble, apply, mirror

#include <array>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <ceres/jet.h>

#include "../core/craft.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "ekf.hpp"

namespace manta::estimation {

template <int NumCrafts, int MeasDim>
class WorldEKFBlockDecomposed {
    static_assert(NumCrafts >= 1, "WorldEKFBlockDecomposed needs at least one craft");

public:
    static constexpr int kCrafts        = NumCrafts;
    static constexpr int kRigidStateDim = 13;
    static constexpr int kStateDim      = NumCrafts * kRigidStateDim;

    // Jet width = 13 (per-craft) — the whole point of block-decomposing.
    using Jet      = ceres::Jet<double, kRigidStateDim>;
    using WorldJ   = WorldT<Jet>;
    using CraftR   = CraftT<double>;
    using CraftJ   = CraftT<Jet>;

    using StateVec = Eigen::Matrix<double, kStateDim, 1>;
    using StateCov = Eigen::Matrix<double, kStateDim, kStateDim>;
    using MeasVec  = Eigen::Matrix<double, MeasDim,   1>;
    using MeasCov  = Eigen::Matrix<double, MeasDim,   MeasDim>;

    WorldEKFBlockDecomposed() = default;

    void bind(WorldJ& w_jet,
              std::array<CraftR*, NumCrafts> real_crafts,
              std::array<CraftJ*, NumCrafts> jet_crafts) noexcept {
        w_jet_       = &w_jet;
        crafts_real_ = real_crafts;
        crafts_jet_  = jet_crafts;
    }

    void set_state(const StateVec& x)        noexcept { x_ = x; }
    void set_covariance(const StateCov& P)   noexcept { P_ = P; }

    const StateVec& state()      const noexcept { return x_; }
    const StateCov& covariance() const noexcept { return P_; }

    // ---- Per-craft slice accessors (mirror WorldEKF) ----
    Eigen::Matrix<double, 3, 1> position(int idx = 0) const noexcept {
        return x_.template segment<3>(idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 4, 1> orientation(int idx = 0) const noexcept {
        return x_.template segment<4>(idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear(int idx = 0) const noexcept {
        return x_.template segment<3>(idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular(int idx = 0) const noexcept {
        return x_.template segment<3>(idx * kRigidStateDim + 10);
    }
    const StateVec& full_state() const noexcept { return x_; }

    Eigen::Matrix<double, 3, 1> position_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 4, 1> orientation_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<4>(idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidStateDim + 10);
    }

    CraftR&       craft(int idx = 0)       noexcept { return *crafts_real_[idx]; }
    const CraftR& craft(int idx = 0) const noexcept { return *crafts_real_[idx]; }
    CraftJ&       craft_jet(int idx = 0)       noexcept { return *crafts_jet_[idx]; }
    const CraftJ& craft_jet(int idx = 0) const noexcept { return *crafts_jet_[idx]; }

    // ---- Fused predict/update lifecycle ----

    void begin_step(double dt, const StateCov& Q) {
        if (!w_jet_) return;
        w_jet_->clock().set_dt(static_cast<float>(dt));
        dt_pending_ = dt;
        Q_pending_  = Q;
        queue_.clear();
        F_.setZero();
        x_post_.setZero();
        active_craft_ = -1;
    }

    // Begin Jet pass for craft `k`: seed craft k's state with identity-
    // derivative Jets, every other craft j ≠ k with value-only Jets
    // (zero derivatives), and run `w_jet.evaluate()`. After this call
    // the Jet sensor caches are populated at x_pre and the active
    // craft's H block can be read by `add_update`.
    void begin_craft(int k) {
        if (!w_jet_) return;
        active_craft_ = k;
        for (int j = 0; j < NumCrafts; ++j) {
            typename CraftJ::RigidState xk;
            for (int i = 0; i < kRigidStateDim; ++i) {
                if (j == k) {
                    xk(i) = Jet(x_(kRigidStateDim*j + i), i);
                } else {
                    xk(i) = Jet(x_(kRigidStateDim*j + i));
                }
            }
            crafts_jet_[j]->set_rigid_state(xk);
        }
        w_jet_->evaluate();
    }

    // Read h(x_pre) + H block for the currently active craft. Called
    // once per fresh sensor on craft `active_craft_`. The H matrix
    // captured here is sparse: only the 13 columns at
    // [13·active_craft_ .. 13·(active_craft_+1)) are nonzero, since
    // the inactive crafts' Jet states had zero derivatives.
    template <int N, class HReader>
    void add_update(const HReader& h,
                    const Eigen::Matrix<double, N, 1>& z,
                    const Eigen::Matrix<double, N, N>& R) {
        const int k = active_craft_;
        Eigen::Matrix<Jet, N, 1> h_jet = h(*this);

        Eigen::Matrix<double, N, 1>        h_pre;
        Eigen::Matrix<double, N, kStateDim> H =
            Eigen::Matrix<double, N, kStateDim>::Zero();
        for (int i = 0; i < N; ++i) {
            h_pre(i) = h_jet(i).a;
            for (int j = 0; j < kRigidStateDim; ++j) {
                H(i, kRigidStateDim*k + j) = h_jet(i).v[j];
            }
        }
        Eigen::Matrix<double, N, 1> y_innov = z - h_pre;

        queue_.push_back([H, y_innov, R](StateVec& x, StateCov& P) {
            Eigen::Matrix<double, N, N> S = H * P * H.transpose() + R;
            Eigen::Matrix<double, kStateDim, N> K =
                P * H.transpose() * S.inverse();
            x = x + K * y_innov;
            P = (StateCov::Identity() - K * H) * P;
            P = 0.5 * (P + P.transpose().eval());
        });
    }

    // End the active craft's Jet pass: advance the Jet world to x_post
    // and read this craft's F-block + post-integrate state slice.
    // Other crafts also advance (they share the world) but their
    // derivatives are zero, so we ignore their Jet output here — each
    // craft's slot in F is filled by ITS OWN pass.
    void end_craft() {
        if (!w_jet_) return;
        const int k = active_craft_;
        w_jet_->advance_only(static_cast<Jet>(dt_pending_));

        auto xk_jet = crafts_jet_[k]->get_rigid_state();
        for (int i = 0; i < kRigidStateDim; ++i) {
            x_post_(kRigidStateDim*k + i) = xk_jet(i).a;
            for (int j = 0; j < kRigidStateDim; ++j) {
                F_(kRigidStateDim*k + i, kRigidStateDim*k + j) = xk_jet(i).v[j];
            }
        }
        active_craft_ = -1;
    }

    void end_step() {
        if (!w_jet_) return;

        // Block-diagonal F → P_pre = F P F^T + Q.
        StateCov P = F_ * P_ * F_.transpose() + Q_pending_;
        StateVec x = x_post_;

        for (auto& apply : queue_) apply(x, P);
        queue_.clear();

        // Renormalize each craft's quaternion (see WorldEKF for the
        // rationale; same convention).
        for (int k = 0; k < NumCrafts; ++k) {
            const int qw = kRigidStateDim*k + 3;
            const double n2 =
                x(qw)*x(qw) + x(qw+1)*x(qw+1)
              + x(qw+2)*x(qw+2) + x(qw+3)*x(qw+3);
            if (n2 > 1e-24) {
                const double inv_n = 1.0 / std::sqrt(n2);
                x(qw)   *= inv_n;
                x(qw+1) *= inv_n;
                x(qw+2) *= inv_n;
                x(qw+3) *= inv_n;
            }
        }

        x_ = x;
        P_ = P;

        for (int k = 0; k < NumCrafts; ++k) {
            if (!crafts_real_[k]) continue;
            typename CraftR::RigidState xk =
                x.template segment<kRigidStateDim>(kRigidStateDim * k);
            crafts_real_[k]->set_rigid_state(xk);
        }
    }

private:
    template <int N>
    Eigen::Matrix<double, N, 1> diag_stddev_segment(int start) const noexcept {
        Eigen::Matrix<double, N, 1> out;
        for (int i = 0; i < N; ++i) {
            const double v = P_(start + i, start + i);
            out(i) = std::sqrt(v > 0.0 ? v : 0.0);
        }
        return out;
    }

    WorldJ*                          w_jet_       = nullptr;
    std::array<CraftR*, NumCrafts>   crafts_real_{};
    std::array<CraftJ*, NumCrafts>   crafts_jet_ {};
    StateVec                         x_           = StateVec::Zero();
    StateCov                         P_           = StateCov::Identity();

    // Per-tick fused state (lives between begin_step and end_step).
    double                                                       dt_pending_ = 0.0;
    StateCov                                                     Q_pending_  = StateCov::Zero();
    std::vector<std::function<void(StateVec&, StateCov&)>>       queue_;
    StateCov                                                     F_          = StateCov::Zero();
    StateVec                                                     x_post_     = StateVec::Zero();
    int                                                          active_craft_ = -1;
};

} // namespace manta::estimation
