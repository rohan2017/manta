#pragma once

// `manta::estimation::BlockDecomposedEKF` — block-decomposed ESKF for
// swarms of decoupled crafts.
//
// For NumCrafts ≥ ~5 the monolithic `EKF<NumCrafts, MeasDim>` becomes
// expensive: every Jet operation tracks 12·NumCrafts partial derivatives,
// so a Jet world pass costs O(NumCrafts²) per part-tick.
//
// When the crafts don't physically couple — no tether, contact, fluid
// coupling, or shared sensor — the process Jacobian F is block diagonal:
//
//     ∂x_i_post/∂x_j_pre = 0   for i ≠ j
//
// and we can compute each block independently. This class runs NumCrafts
// Jet passes per tick, each with `Jet<double, 12>` (one craft's worth of
// tangent partials):
//
//   for k in 0..NumCrafts:
//     seed craft k's state via boxplus(ref_k, δ_jet)         (12-DOF tangent)
//     seed every other craft j ≠ k with value-only Jets      (zero derivs)
//     w_jet.kinematic_and_aggregate()
//     for each measurement on craft k:
//         read h_jet from craft k's sensors → store h_pre + H[meas, k-cols]
//     w_jet.integrate(dt)
//     read craft k's post state via boxminus → x_k_ref_post + F[k-rows, k-cols]
//
// The off-diagonal F blocks stay at zero. Cost: NumCrafts × O(parts · 12²)
// instead of 1 × O(parts · (12·NumCrafts)²) — linear in NumCrafts instead
// of quadratic.
//
// Coupling caveat: this only produces the correct Jacobian when crafts
// are dynamically independent. When craft j is "inactive" in pass k, its
// state has zero Jet derivatives, so any contribution to craft k's
// dynamics through cross-craft physics goes uncomputed. Tethers, contact,
// fluid drag from a neighbor's wake — all unmodeled here. Use the
// monolithic EKF for those cases.
//
// State conventions match `EKF`: 13·N ambient reference, 12·N tangent
// covariance. See ekf.hpp for the boxplus / boxminus discussion.
//
// Usage with explicit per-craft brackets:
//
//   ekf.begin_step(dt, Q);
//   for (int k = 0; k < NumCrafts; ++k) {
//       ekf.begin_craft(k);                                 // seed + evaluate
//       if (sensor_k.consume_fresh())
//           ekf.add_update<N>(h_at_pre, z, R);              // reads Jet h
//       ekf.end_craft();                                    // advance, capture F_kk
//   }
//   ekf.end_step();                                         // assemble, apply, mirror

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
#include "../geom/kinematic_link.hpp"   // angle_axis_to_quat

namespace manta::estimation {

template <int NumCrafts, int MeasDim>
class BlockDecomposedEKF {
    static_assert(NumCrafts >= 1, "BlockDecomposedEKF needs at least one craft");

public:
    static constexpr int kCrafts        = NumCrafts;
    static constexpr int kRigidStateDim = 13;
    static constexpr int kRigidTangent  = 12;
    static constexpr int kStateDim      = NumCrafts * kRigidStateDim;
    static constexpr int kTangentDim    = NumCrafts * kRigidTangent;

    // Jet width = 12 (per-craft) — the whole point of block-decomposing.
    using Jet      = ceres::Jet<double, kRigidTangent>;
    using WorldJ   = WorldT<Jet>;
    using CraftR   = CraftT<double>;
    using CraftJ   = CraftT<Jet>;

    using StateVec   = Eigen::Matrix<double, kStateDim,   1>;            // 13·N ambient
    using StateCov   = Eigen::Matrix<double, kTangentDim, kTangentDim>;  // 12·N × 12·N
    using TangentVec = Eigen::Matrix<double, kTangentDim, 1>;
    using MeasVec    = Eigen::Matrix<double, MeasDim,     1>;
    using MeasCov    = Eigen::Matrix<double, MeasDim,     MeasDim>;

    BlockDecomposedEKF() : x_ref_(StateVec::Zero()), P_(StateCov::Identity()) {
        for (int k = 0; k < NumCrafts; ++k) x_ref_(13 * k + 3) = 1.0;
    }

    void bind(WorldJ& w_jet,
              std::array<CraftR*, NumCrafts> real_crafts,
              std::array<CraftJ*, NumCrafts> jet_crafts) noexcept {
        w_jet_       = &w_jet;
        crafts_real_ = real_crafts;
        crafts_jet_  = jet_crafts;
    }

    void set_state(const StateVec& x) noexcept {
        x_ref_ = x;
        renormalize_quats(x_ref_);
        mirror_to_real();
    }
    void set_covariance(const StateCov& P) noexcept { P_ = P; }

    const StateVec& state()      const noexcept { return x_ref_; }
    const StateCov& covariance() const noexcept { return P_; }

    // ---- Per-craft slice accessors (mirror EKF) ----
    Eigen::Matrix<double, 3, 1> position(int idx = 0) const noexcept {
        return x_ref_.template segment<3>(idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 4, 1> orientation(int idx = 0) const noexcept {
        return x_ref_.template segment<4>(idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear(int idx = 0) const noexcept {
        return x_ref_.template segment<3>(idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular(int idx = 0) const noexcept {
        return x_ref_.template segment<3>(idx * kRigidStateDim + 10);
    }
    const StateVec& full_state() const noexcept { return x_ref_; }

    Eigen::Matrix<double, 3, 1> position_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 0);
    }
    Eigen::Matrix<double, 3, 1> orientation_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 6);
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 9);
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
        x_ref_post_ = x_ref_;     // value-side default; per-craft passes overwrite
        active_craft_ = -1;
    }

    // Begin Jet pass for craft k: seed craft k's state via boxplus(ref, δ_jet)
    // with identity-derivative Jets in the per-craft tangent slots; seed
    // every other craft j ≠ k with value-only Jets (zero derivatives).
    void begin_craft(int k) {
        if (!w_jet_) return;
        active_craft_ = k;
        for (int j = 0; j < NumCrafts; ++j) {
            if (j == k) seed_craft_jets_with_tangent(j);
            else        seed_craft_jets_value_only(j);
        }
        w_jet_->kinematic_and_aggregate();
    }

    // Read h(x_ref) + H block for the currently active craft. The H
    // matrix captured here is sparse: only the 12 columns at
    // [12·k .. 12·(k+1)) are nonzero, since inactive crafts had zero
    // derivatives.
    template <int N, class HReader>
    void add_update(const HReader& h,
                    const Eigen::Matrix<double, N, 1>& z,
                    const Eigen::Matrix<double, N, N>& R) {
        const int k = active_craft_;
        Eigen::Matrix<Jet, N, 1> h_jet = h(*this);

        Eigen::Matrix<double, N, 1>            h_pre;
        Eigen::Matrix<double, N, kTangentDim>  H =
            Eigen::Matrix<double, N, kTangentDim>::Zero();
        for (int i = 0; i < N; ++i) {
            h_pre(i) = h_jet(i).a;
            for (int j = 0; j < kRigidTangent; ++j) {
                H(i, kRigidTangent * k + j) = h_jet(i).v[j];
            }
        }
        Eigen::Matrix<double, N, 1> y_innov = z - h_pre;

        queue_.push_back([H, y_innov, R](TangentVec& delta_acc, StateCov& P) {
            Eigen::Matrix<double, N, N>           S = H * P * H.transpose() + R;
            Eigen::Matrix<double, kTangentDim, N> K = P * H.transpose() * S.inverse();
            delta_acc += K * y_innov;
            P = (StateCov::Identity() - K * H) * P;
            P = 0.5 * (P + P.transpose().eval());
        });
    }

    // End the active craft's Jet pass: advance the Jet world and extract
    // (x_k_ref_post, F_kk) for this craft via boxminus. Other crafts also
    // advance under the shared Jet world but their Jet derivatives are
    // zero — each craft's F slot is filled by its own pass.
    void end_craft() {
        if (!w_jet_) return;
        const int k = active_craft_;
        w_jet_->integrate(static_cast<Jet>(dt_pending_));
        extract_F_block_and_ref(k);
        active_craft_ = -1;
    }

    void end_step() {
        if (!w_jet_) return;

        // x_ref_post_ came from the Jet world's integrator, which
        // already normalized each craft's q.
        x_ref_ = x_ref_post_;

        // Block-diagonal F → P_pre = F P F^T + Q.
        StateCov P = F_ * P_ * F_.transpose() + Q_pending_;
        P = 0.5 * (P + P.transpose().eval());

        TangentVec delta_acc = TangentVec::Zero();
        for (auto& apply : queue_) apply(delta_acc, P);
        queue_.clear();

        inject_delta(delta_acc);

        P_ = P;
        mirror_to_real();
    }

private:
    // Seed craft j's Jet state from x_ref using the boxplus(ref, δ_jet)
    // pattern with identity-derivative Jets in slots [0..11] of this Jet
    // width (per-craft, since Jet width = 12 here).
    void seed_craft_jets_with_tangent(int j) {
        const int ref_off = 13 * j;
        typename CraftJ::RigidState xk;

        for (int i = 0; i < 3; ++i) {
            xk(i) = Jet(x_ref_(ref_off + i), i);
        }

        const double qwr = x_ref_(ref_off + 3);
        const double qxr = x_ref_(ref_off + 4);
        const double qyr = x_ref_(ref_off + 5);
        const double qzr = x_ref_(ref_off + 6);

        Jet exp_w(1.0);
        Jet exp_x(0.0); exp_x.v[3] = 0.5;
        Jet exp_y(0.0); exp_y.v[4] = 0.5;
        Jet exp_z(0.0); exp_z.v[5] = 0.5;

        xk(3) = qwr * exp_w - qxr * exp_x - qyr * exp_y - qzr * exp_z;
        xk(4) = qwr * exp_x + qxr * exp_w + qyr * exp_z - qzr * exp_y;
        xk(5) = qwr * exp_y - qxr * exp_z + qyr * exp_w + qzr * exp_x;
        xk(6) = qwr * exp_z + qxr * exp_y - qyr * exp_x + qzr * exp_w;

        for (int i = 0; i < 3; ++i) xk(7  + i) = Jet(x_ref_(ref_off + 7  + i), 6 + i);
        for (int i = 0; i < 3; ++i) xk(10 + i) = Jet(x_ref_(ref_off + 10 + i), 9 + i);

        crafts_jet_[j]->set_rigid_state(xk);
    }

    // Seed craft j's Jet state with value-only Jets (zero derivatives).
    void seed_craft_jets_value_only(int j) {
        const int ref_off = 13 * j;
        typename CraftJ::RigidState xk;
        for (int i = 0; i < kRigidStateDim; ++i) {
            xk(i) = Jet(x_ref_(ref_off + i));
        }
        crafts_jet_[j]->set_rigid_state(xk);
    }

    // Extract F_kk and x_k_ref_post from the active craft's Jet state.
    void extract_F_block_and_ref(int k) {
        auto rk = crafts_jet_[k]->get_rigid_state();
        const int ref_off = 13 * k;
        const int tan_off = 12 * k;

        // Reference (post) — `.a` channel only.
        for (int i = 0; i < 13; ++i) x_ref_post_(ref_off + i) = rk(i).a;

        // Position rows.
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < kRigidTangent; ++j) {
                F_(tan_off + i, tan_off + j) = rk(i).v[j];
            }
        }

        // Orientation rows: 2·imag(q_ref_post.conj ⊗ q_jet_post).
        const double qwr = rk(3).a;
        const double qxr = rk(4).a;
        const double qyr = rk(5).a;
        const double qzr = rk(6).a;
        const Jet&   qwj = rk(3);
        const Jet&   qxj = rk(4);
        const Jet&   qyj = rk(5);
        const Jet&   qzj = rk(6);

        const Jet x_dq = qwr * qxj - qxr * qwj - qyr * qzj + qzr * qyj;
        const Jet y_dq = qwr * qyj + qxr * qzj - qyr * qwj - qzr * qxj;
        const Jet z_dq = qwr * qzj - qxr * qyj + qyr * qxj - qzr * qwj;

        for (int j = 0; j < kRigidTangent; ++j) {
            F_(tan_off + 3, tan_off + j) = 2.0 * x_dq.v[j];
            F_(tan_off + 4, tan_off + j) = 2.0 * y_dq.v[j];
            F_(tan_off + 5, tan_off + j) = 2.0 * z_dq.v[j];
        }

        // Velocity rows.
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < kRigidTangent; ++j) {
                F_(tan_off + 6 + i, tan_off + j) = rk(7 + i).v[j];
            }
        }
        // Angular velocity rows.
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < kRigidTangent; ++j) {
                F_(tan_off + 9 + i, tan_off + j) = rk(10 + i).v[j];
            }
        }
    }

    void inject_delta(const TangentVec& delta) {
        for (int k = 0; k < NumCrafts; ++k) {
            const int ref_off = 13 * k;
            const int tan_off = 12 * k;

            for (int i = 0; i < 3; ++i) x_ref_(ref_off + i) += delta(tan_off + i);

            Eigen::Matrix<double, 3, 1> dtheta(
                delta(tan_off + 3), delta(tan_off + 4), delta(tan_off + 5));
            Eigen::Quaterniond exp_q = geom::angle_axis_to_quat<double>(dtheta);
            Eigen::Quaterniond q_old(
                x_ref_(ref_off + 3), x_ref_(ref_off + 4),
                x_ref_(ref_off + 5), x_ref_(ref_off + 6));
            Eigen::Quaterniond q_new = q_old * exp_q;
            q_new.normalize();
            x_ref_(ref_off + 3) = q_new.w();
            x_ref_(ref_off + 4) = q_new.x();
            x_ref_(ref_off + 5) = q_new.y();
            x_ref_(ref_off + 6) = q_new.z();

            for (int i = 0; i < 3; ++i) {
                x_ref_(ref_off + 7  + i) += delta(tan_off + 6 + i);
                x_ref_(ref_off + 10 + i) += delta(tan_off + 9 + i);
            }
        }
    }

    void mirror_to_real() {
        for (int k = 0; k < NumCrafts; ++k) {
            if (!crafts_real_[k]) continue;
            typename CraftR::RigidState xk =
                x_ref_.template segment<13>(13 * k);
            crafts_real_[k]->set_rigid_state(xk);
        }
    }

    static void renormalize_quats(StateVec& x) noexcept {
        for (int k = 0; k < NumCrafts; ++k) {
            const int o = 13 * k + 3;
            const double n2 = x(o)*x(o) + x(o+1)*x(o+1)
                            + x(o+2)*x(o+2) + x(o+3)*x(o+3);
            if (n2 > 1e-24) {
                const double inv_n = 1.0 / std::sqrt(n2);
                x(o)   *= inv_n;
                x(o+1) *= inv_n;
                x(o+2) *= inv_n;
                x(o+3) *= inv_n;
            }
        }
    }

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

    StateVec  x_ref_;
    StateCov  P_;

    // Per-tick fused state.
    double                                                       dt_pending_ = 0.0;
    StateCov                                                     Q_pending_  = StateCov::Zero();
    std::vector<std::function<void(TangentVec&, StateCov&)>>     queue_;
    StateCov                                                     F_          = StateCov::Zero();
    StateVec                                                     x_ref_post_ = StateVec::Zero();
    int                                                          active_craft_ = -1;
};

} // namespace manta::estimation
