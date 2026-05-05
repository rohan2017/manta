#pragma once

// WorldEKF — an EKF wired directly against a user's templated WorldT/CraftT.
//
// The filter estimates the joint state of every Craft in a single World
// simultaneously. State layout is the concatenation of each craft's 13-DOF
// rigid-body state in the order the user binds them:
//
//   [c0.p (3) | c0.q (4) | c0.v (3) | c0.ω (3) |
//    c1.p (3) | c1.q (4) | c1.v (3) | c1.ω (3) | ... ]
//
//   StateDim = 13 * NumCrafts
//
// The Jacobian step uses Ceres Jets — `Jet<double, StateDim>`. The user's
// `WorldT<Jet>` instance, built identically to the Real-side world (same
// crafts, same fields, same registered planet pose), runs full multi-craft
// physics (kinematic pass, force aggregation, integration) for every Jet
// `predict` call. Cross-craft physics (tethers, contacts, fluid coupling)
// participates in the Jacobian for free — anything that's correct in the
// Real World is automatically correct in the Jet World.
//
// Selective Jet propagation: planets are Real-typed and shared between the
// two worlds. Their KinematicLink<World, Planet> is cast Real→Jet
// component-wise inside SceneT<Jet>::update with zero gradient. The EKF
// thus treats planet pose as a non-estimated input.
//
// Performance note: this monolithic predict tracks `13·NumCrafts` partials
// per Jet op, so a Jet world pass costs O(NumCrafts²). It's the right
// choice when crafts physically couple (tethers, contact, fluid coupling)
// — the autodiff captures every cross-craft Jacobian for free. For
// decoupled-craft swarms where F is block-diagonal, see
// `WorldEKFBlockDecomposed` (world_ekf_block.hpp): per-craft Jet passes
// with width 13, NumCrafts passes per tick, linear scaling.
//
// Usage sketch:
//
//     // Build the Real world (same shape as a sim World).
//     manta::World w_real;
//     w_real.clock().set_dt(0.01f);
//     auto& s_real = w_real.create_scene();
//     w_real.register_field(gravity);
//     MyCraft<double> craft_real;
//     s_real.add_craft(craft_real);
//
//     // Build the Jet shadow World identically.
//     using EkfT = manta::estimation::WorldEKF</*NumCrafts=*/1, /*MeasDim=*/3>;
//     manta::WorldT<EkfT::Jet> w_jet;
//     w_jet.clock().set_dt(0.01f);
//     auto& s_jet = w_jet.create_scene();
//     w_jet.register_field(gravity);   // SAME field instance — read-only
//     MyCraft<EkfT::Jet> craft_jet;
//     s_jet.add_craft(craft_jet);
//
//     // Construct the EKF and bind to both sides.
//     EkfT ekf;
//     ekf.bind(w_jet, {&craft_real}, {&craft_jet});
//
//     // Drive it.
//     ekf.set_state(...); ekf.set_covariance(...);
//     for (int i = 0; i < N; ++i) {
//         ekf.predict(dt, Q);
//         ekf.update_n<3>(some_h, z, R);
//     }
//
// `craft.set_measurement(...)` writes happen on the REAL crafts the user
// owns directly (via the namespace-scope handle the codegen exposes). The
// Jet crafts are internal to the predict step.

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
class WorldEKF {
    static_assert(NumCrafts >= 1, "WorldEKF needs at least one craft");

public:
    static constexpr int kCrafts        = NumCrafts;
    static constexpr int kRigidStateDim = 13;
    static constexpr int kStateDim      = NumCrafts * kRigidStateDim;

    using Jet      = ceres::Jet<double, kStateDim>;
    using WorldR   = WorldT<double>;
    using WorldJ   = WorldT<Jet>;
    using CraftR   = CraftT<double>;
    using CraftJ   = CraftT<Jet>;

    using StateVec = Eigen::Matrix<double, kStateDim, 1>;
    using StateCov = Eigen::Matrix<double, kStateDim, kStateDim>;
    using MeasVec  = Eigen::Matrix<double, MeasDim,   1>;
    using MeasCov  = Eigen::Matrix<double, MeasDim,   MeasDim>;

    WorldEKF() = default;

    // Bind the filter to its Jet shadow World + the matching craft pointers.
    // Crafts must appear in the same slot order on both sides — that order
    // is the one the state vector encodes (craft 0 = state[0..12], craft 1
    // = state[13..25], ...).
    //
    // The Real World itself isn't needed by the EKF — `predict()` only ever
    // advances the Jet world. The Real-craft pointers are kept so each
    // post-predict state can be mirrored back into the user's Real-side
    // crafts, keeping their `set_measurement`-fed sensor parts in sync
    // with the filter's belief.
    void bind(WorldJ& w_jet,
              std::array<CraftR*, NumCrafts> real_crafts,
              std::array<CraftJ*, NumCrafts> jet_crafts) noexcept {
        w_jet_       = &w_jet;
        crafts_real_ = real_crafts;
        crafts_jet_  = jet_crafts;
    }

    void set_state(const StateVec& x)        noexcept { ekf_.set_state(x); }
    void set_covariance(const StateCov& P)   noexcept { ekf_.set_covariance(P); }

    const StateVec& state()      const noexcept { return ekf_.state(); }
    const StateCov& covariance() const noexcept { return ekf_.covariance(); }

    // Predict step — drives a full Jet-typed `WorldT::update()` with seeded
    // Jet state. Extracts the value + the kStateDim×kStateDim Jacobian. After
    // the EKF state has advanced, mirrors back into the Real-side crafts.
    //
    // `dt` is the timestep in seconds. `Q` is the process-noise covariance.
    void predict(double dt, const StateCov& Q) {
        if (!w_jet_) return;
        w_jet_->clock().set_dt(static_cast<float>(dt));

        auto f = [this, dt](const auto& x_arg, double /*dt_*/) {
            using S = typename std::decay_t<decltype(x_arg)>::Scalar;
            // EKF::predict always invokes f with Jet input; the inline cast
            // covers Real inputs in case a unit test or future call site
            // wants the value path standalone.
            if constexpr (std::is_same_v<S, double>) {
                Eigen::Matrix<double, kStateDim, 1> y;
                // Manual evaluation against per-craft real instances when
                // available — but predict always goes through Jet. This
                // branch isn't normally exercised.
                for (int k = 0; k < NumCrafts; ++k) {
                    typename CraftR::RigidState xk;
                    for (int i = 0; i < 13; ++i) xk(i) = x_arg(13*k + i);
                    auto y_k = crafts_real_[k]->evaluate(xk, S(dt));
                    for (int i = 0; i < 13; ++i) y(13*k + i) = y_k(i);
                }
                return y;
            } else {
                // Jet path — write each craft's state slice, advance the
                // entire Jet world, read each craft's new state slice.
                for (int k = 0; k < NumCrafts; ++k) {
                    typename CraftJ::RigidState xk;
                    for (int i = 0; i < 13; ++i) xk(i) = x_arg(13*k + i);
                    crafts_jet_[k]->set_rigid_state(xk);
                }
                w_jet_->update();
                Eigen::Matrix<S, kStateDim, 1> y;
                for (int k = 0; k < NumCrafts; ++k) {
                    auto xk_out = crafts_jet_[k]->get_rigid_state();
                    for (int i = 0; i < 13; ++i) y(13*k + i) = xk_out(i);
                }
                return y;
            }
        };
        ekf_.predict(f, dt, Q);

        // Mirror the new state back into the Real-side crafts so the user's
        // sensor part `last_*` values, telemetry reads, and any subsequent
        // `set_measurement` calls operate on the post-predict state.
        for (int k = 0; k < NumCrafts; ++k) {
            if (!crafts_real_[k]) continue;
            typename CraftR::RigidState xk =
                ekf_.state().template segment<kRigidStateDim>(kRigidStateDim * k);
            crafts_real_[k]->set_rigid_state(xk);
        }
    }

    // Measurement update with a per-call dimension N. Lets a single EKF
    // absorb varying-width per-sensor reads (DVL=3, IMU=6, Mag=3) without
    // instantiating multiple filters.
    template <int N, class MeasureH>
    void update_n(const MeasureH& h,
                  const Eigen::Matrix<double, N, 1>& z,
                  const Eigen::Matrix<double, N, N>& R) {
        ekf_.template update_n<N>(h, z, R);

        // Mirror updated state into Real crafts (consistent with predict's
        // post-step mirror). Strictly only necessary when downstream code
        // reads from the Real crafts before the next predict; cheap enough
        // to do unconditionally.
        for (int k = 0; k < NumCrafts; ++k) {
            if (!crafts_real_[k]) continue;
            typename CraftR::RigidState xk =
                ekf_.state().template segment<kRigidStateDim>(kRigidStateDim * k);
            crafts_real_[k]->set_rigid_state(xk);
        }
    }

    // Single-measurement convenience overload for callers that fixed
    // MeasDim at template-construction time.
    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        update_n<MeasDim>(h, z, R);
    }

    // ---- Per-craft slice accessors ----
    // craft_idx defaults to 0 for backward-compat with single-craft callers.
    Eigen::Matrix<double, 3, 1> position(int craft_idx = 0) const noexcept {
        return ekf_.state().template segment<3>(craft_idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 4, 1> orientation(int craft_idx = 0) const noexcept {
        return ekf_.state().template segment<4>(craft_idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear(int craft_idx = 0) const noexcept {
        return ekf_.state().template segment<3>(craft_idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular(int craft_idx = 0) const noexcept {
        return ekf_.state().template segment<3>(craft_idx * kRigidStateDim + 10);
    }
    const StateVec& full_state() const noexcept { return ekf_.state(); }

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

    // Per-craft handle. Useful in measurement functors that read sensor
    // last_* values from a specific Real craft.
    CraftR&       craft(int idx = 0)       noexcept { return *crafts_real_[idx]; }
    const CraftR& craft(int idx = 0) const noexcept { return *crafts_real_[idx]; }

    // Per-craft handle on the Jet-shadow craft. Used by measurement
    // functors written for the fused begin_step / add_update / end_step
    // path: between begin_step and end_step the Jet craft holds x_pre
    // values with identity Jet derivatives, and `evaluate()` has already
    // populated the kinematic + acc_linear caches at x_pre. Functors
    // read sensor accessors (e.g. `imu().specific_force_body()`) and
    // return the result as Jet vectors; the Jet derivatives are the
    // measurement Jacobian H.
    CraftJ&       craft_jet(int idx = 0)       noexcept { return *crafts_jet_[idx]; }
    const CraftJ& craft_jet(int idx = 0) const noexcept { return *crafts_jet_[idx]; }

    // ===========================================================
    // Fused single-pass predict + update (PyPose-style linearize-at-
    // x_{k-1} formulation).
    //
    // Lifecycle for one tick:
    //
    //   ekf.begin_step(dt, Q);                 // seeds Jets at x_pre
    //                                          // with identity, runs
    //                                          // w_jet.evaluate()
    //                                          // (kinematic + agg).
    //                                          // Cache is now at x_pre
    //                                          // with derivs w.r.t.
    //                                          // x_pre.
    //
    //   if (sensor.consume_fresh())
    //       ekf.add_update<N>(h_at_pre, z, R);
    //                                          // Reads h(x_pre) +
    //                                          // H = ∂h/∂x_pre from
    //                                          // the Jet sensors.
    //                                          // Queues the (h, H, z, R)
    //                                          // tuple for application
    //                                          // in end_step.
    //   ...
    //
    //   ekf.end_step();                        // advances Jet world
    //                                          // (no re-aggregate);
    //                                          // reads x_post + F;
    //                                          // P_pre = F P F^T + Q;
    //                                          // applies queued updates
    //                                          // sequentially; mirrors
    //                                          // posterior to Real
    //                                          // crafts.
    //
    // Cost: one Jet world pass (kin + agg + integrate) per tick — h
    // evaluations are cheap reads on the same Jet caches that predict
    // already had to populate. Compare to the older predict()+update_n
    // path which ran (predict's update + IMU's evaluate) = roughly 1.7×
    // a single pass.
    //
    // Innovation linearization happens at x_{k-1|k-1} (PyPose / fixed-
    // reference variant) rather than at the predicted x_{k|k-1}. The
    // approximation is O(dt · ||F − I||); for high-rate filters
    // (kHz) this is below other modeling errors.
    void begin_step(double dt, const StateCov& Q) {
        if (!w_jet_) return;
        w_jet_->clock().set_dt(static_cast<float>(dt));
        dt_pending_ = dt;
        Q_pending_  = Q;
        queue_.clear();

        // Seed Jet crafts at x_pre with identity derivatives.
        const auto& x_pre = ekf_.state();
        for (int k = 0; k < NumCrafts; ++k) {
            typename CraftJ::RigidState xk;
            for (int i = 0; i < kRigidStateDim; ++i) {
                xk(i) = Jet(x_pre(kRigidStateDim*k + i),
                            kRigidStateDim*k + i);
            }
            crafts_jet_[k]->set_rigid_state(xk);
        }

        // One Jet pass: kin + agg at x_pre. Caches the Jet sensors'
        // last_* values and the scene_to_part acc_linear field; both
        // carry derivs w.r.t. x_pre suitable for measurement functors.
        w_jet_->evaluate();
    }

    // Read a measurement functor's h(x_pre) + Jacobian and queue the
    // (h, H, z, R) tuple for application in end_step.
    //
    // The HReader is a callable invoked as `h(*this)` — it returns
    // `Eigen::Matrix<Jet, N, 1>`. Functors reach into the Jet sensors
    // via `craft_jet(i).<sensor>().<accessor>()`; the cached values
    // are at x_pre with identity-seeded derivatives so each Jet's `.v`
    // slot holds one row of H.
    template <int N, class HReader>
    void add_update(const HReader& h,
                    const Eigen::Matrix<double, N, 1>& z,
                    const Eigen::Matrix<double, N, N>& R) {
        Eigen::Matrix<Jet, N, 1> h_jet = h(*this);

        Eigen::Matrix<double, N, 1>        h_pre;
        Eigen::Matrix<double, N, kStateDim> H;
        for (int i = 0; i < N; ++i) {
            h_pre(i) = h_jet(i).a;
            for (int j = 0; j < kStateDim; ++j) {
                H(i, j) = h_jet(i).v[j];
            }
        }
        Eigen::Matrix<double, N, 1> y_innov = z - h_pre;

        // Capture H, y, R by value into a type-erased apply step.
        queue_.push_back([H, y_innov, R](StateVec& x, StateCov& P) {
            Eigen::Matrix<double, N, N> S = H * P * H.transpose() + R;
            Eigen::Matrix<double, kStateDim, N> K =
                P * H.transpose() * S.inverse();
            x = x + K * y_innov;
            P = (StateCov::Identity() - K * H) * P;
            P = 0.5 * (P + P.transpose().eval());
        });
    }

    void end_step() {
        if (!w_jet_) return;

        // Advance Jet world from x_pre to x_post via integrate-only
        // (no re-aggregate; the next begin_step re-seeds + re-evaluates
        // anyway, so refreshing the cache here would be wasted work).
        w_jet_->advance_only(static_cast<Jet>(dt_pending_));

        // Extract x_post + F from the Jet craft state.
        StateVec x_post;
        StateCov F;
        for (int k = 0; k < NumCrafts; ++k) {
            auto xk_jet = crafts_jet_[k]->get_rigid_state();
            for (int i = 0; i < kRigidStateDim; ++i) {
                x_post(kRigidStateDim*k + i) = xk_jet(i).a;
                for (int j = 0; j < kStateDim; ++j) {
                    F(kRigidStateDim*k + i, j) = xk_jet(i).v[j];
                }
            }
        }

        // P_pre = F P F^T + Q.
        StateCov P = F * ekf_.covariance() * F.transpose() + Q_pending_;
        StateVec x = x_post;

        // Apply queued sensor updates sequentially (each shrinks P,
        // shifts x). All H matrices were captured at x_pre so the
        // ordering doesn't matter for correctness — they share the
        // same linearization point.
        for (auto& apply : queue_) apply(x, P);
        queue_.clear();

        // Renormalize each craft's quaternion to unit norm. The Jet
        // path's `q.normalize()` happens inside set_rigid_state, but
        // post-Kalman-update x can have q drifted off unit, so we
        // project back here before the next predict seeds Jets from
        // it. We do NOT project the q-block of P onto the tangent
        // space — for the redundant-direction problem the analytical
        // story would say "zero out the radial component" but in
        // practice on this codebase the projection makes things
        // worse by clamping legitimate qx/qy/qz uncertainty too
        // aggressively. Skipping the P projection leaves the standard
        // EKF behavior (well-conditioned for the typical case).
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

        ekf_.set_state(x);
        ekf_.set_covariance(P);

        // Mirror posterior to Real-side crafts so user-facing handles
        // (sensor `set_measurement` writes, telemetry reads) reflect
        // the filter's current belief.
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
        const auto& P = ekf_.covariance();
        for (int i = 0; i < N; ++i) {
            const double v = P(start + i, start + i);
            out(i) = std::sqrt(v > 0.0 ? v : 0.0);
        }
        return out;
    }

    WorldJ*                          w_jet_       = nullptr;
    std::array<CraftR*, NumCrafts>   crafts_real_{};
    std::array<CraftJ*, NumCrafts>   crafts_jet_ {};
    EKF<kStateDim, MeasDim>          ekf_;

    // Fused step state: held between begin_step() and end_step().
    double                                                       dt_pending_ = 0.0;
    StateCov                                                     Q_pending_  = StateCov::Zero();
    std::vector<std::function<void(StateVec&, StateCov&)>>       queue_;
};

} // namespace manta::estimation
