#pragma once

// `manta::estimation::EKF` — error-state EKF wired against a templated
// WorldT/CraftT. Tracks a 13·N ambient reference state on the value-side
// crafts and a 12·N × 12·N covariance over the tangent space. The
// 4-component quaternion lives in the reference; the rotation manifold's
// 3 real DOFs live in the tangent. See the long doc-block below for the
// `why two worlds` discussion.
//
// State conventions:
//
//   Reference (13·N, ambient):
//     [c0.p (3) | c0.q (4) | c0.v (3) | c0.ω (3) |
//      c1.p (3) | ... ]
//
//   Tangent / error (12·N, used by F, P, K, δ):
//     [c0.δp (3) | c0.δθ (3) | c0.δv (3) | c0.δω (3) |
//      c1.δp (3) | ... ]
//
//   Boxplus retraction `x_ref ⊞ δ`:
//     p_full = p_ref + δp
//     q_full = q_ref ⊗ exp(δθ/2)        (so(3) exponential)
//     v_full = v_ref + δv
//     ω_full = ω_ref + δω
//
// The orientation block is 3-DOF instead of 4 because the rotation manifold
// is 3-dimensional; the unit-quaternion ambient representation has a
// redundant radial direction that the tangent omits.
//
// ---- Why two Worlds: a value-typed one AND a Jet-typed one ----
//
// An EKF needs two things from its process model on every predict step:
//
//   1. The propagated value         x_post = f(x_pre, dt)
//   2. The process Jacobian         F      = ∂f/∂δ_pre   (12·N × 12·N)
//
// We get F via Ceres forward-mode autodiff. The `f` we run accepts Ceres
// `Jet<double, kTangentDim>` scalars, propagates them through the full
// physics, and the resulting Jets' `.v` rows ARE F. That requires every
// arithmetic op `f` performs — kinematic chain rule, force aggregation,
// integrator step — to be templated on the scalar. manta's WorldT,
// SceneT, CraftT, PartT family is templated on Scalar for exactly this
// reason: the same craft authoring artifact runs at `Scalar = double`
// (sim, telemetry, user code) AND `Scalar = Jet<double, 12·N>` (the
// predict Jacobian).
//
// Two World *instances* rather than one because each WorldT<Scalar> owns
// a typed Scene/Craft graph: pointers, kinematic caches, rigid-body
// state are all templated on Scalar. You cannot ask a `WorldT<double>`
// to "run a tick in Jets" — its parts and craft state ARE doubles. The
// Jet world is built once at setup as a Jet-templated twin: same parts,
// same coefficients, same registered fields and planet. They live
// alongside each other.
//
//   * The value world is the canonical reference state owner. Its crafts
//     hold the filter's posterior belief between predicts (mirrored back
//     after every step), feed user-facing telemetry, absorb
//     `set_measurement` writes from sensor parts.
//
//   * The Jet world is internal scratch for the Jacobian. `predict()`
//     seeds each Jet craft's 13-DOF state with the boxplus(ref, δ_jet)
//     pattern so the Jet `.v` rows live in the 12·N tangent. After one
//     `step()`, boxminus(post_state, x_ref_post) extracts F's rows.
//
// Selective Jet propagation: planets and field instances are value-typed
// and shared. Inside SceneT<Jet>::refresh_world_to_scene the planet's
// KinematicLink is cast value→Jet with zero `.v`; field queries on the
// Jet side go through `state_at_templated<Jet>` which wraps the value
// field's state_at and finite-diffs only the position dependency. So the
// EKF treats planet pose and field structure as non-estimated inputs.
//
// Performance note: this monolithic predict tracks 12·N partials per Jet
// op, so a Jet world pass costs O(NumCrafts²). Right choice when crafts
// physically couple (tethers, contact, fluid coupling) — the autodiff
// captures every cross-craft Jacobian for free. For decoupled-craft
// swarms see `BlockDecomposedEKF` (block_decomposed_ekf.hpp).
//
// Usage sketch:
//
//     manta::World w_real;
//     w_real.clock().set_dt(0.01f);
//     auto& s_real = w_real.create_scene();
//     w_real.register_field(gravity);
//     MyCraft<double> craft_real;
//     s_real.add_craft(craft_real);
//
//     using EkfT = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3>;
//     manta::WorldT<EkfT::Jet> w_jet;
//     w_jet.clock().set_dt(0.01f);
//     auto& s_jet = w_jet.create_scene();
//     w_jet.register_field(gravity);   // SAME field instance — read-only
//     MyCraft<EkfT::Jet> craft_jet;
//     s_jet.add_craft(craft_jet);
//
//     EkfT ekf;
//     ekf.bind(w_jet, {&craft_real}, {&craft_jet});
//     ekf.set_state(...);              // 13·N ambient
//     ekf.set_covariance(...);         // 12·N × 12·N tangent
//     for (int i = 0; i < N; ++i) {
//         ekf.predict(dt, Q);          // Q is 12·N × 12·N
//         ekf.update_n<3>(some_h, z, R);
//     }

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <ceres/jet.h>

#include "../core/craft.hpp"
#include "../core/noise_registry.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "../geom/kinematic_link.hpp"   // angle_axis_to_quat

namespace manta::estimation {

template <int NumCrafts, int MeasDim, int NumNoiseSlots = 0, int BiasDim = 0>
class EKF {
    static_assert(NumCrafts >= 1, "manta::estimation::EKF needs at least one craft");
    static_assert(NumNoiseSlots >= 0, "NumNoiseSlots must be non-negative");
    static_assert(BiasDim       >= 0, "BiasDim must be non-negative");

public:
    static constexpr int kCrafts         = NumCrafts;
    static constexpr int kRigidStateDim  = 13;                  // ambient per craft
    static constexpr int kRigidTangent   = 12;                  // tangent per craft
    static constexpr int kStateDim       = NumCrafts * kRigidStateDim;
    static constexpr int kRigidTangentDim = NumCrafts * kRigidTangent;
    static constexpr int kBiasDim        = BiasDim;             // augmented bias states
    static constexpr int kTangentDim     = kRigidTangentDim + BiasDim;
    static constexpr int kBiasColStart   = kRigidTangentDim;    // bias rows/cols start here
    static constexpr int kNoiseSlots     = NumNoiseSlots;
    static constexpr int kNoiseColStart  = kTangentDim;          // noise inputs in Jet width
    static constexpr int kJetWidth       = kTangentDim + NumNoiseSlots;

    // Jet width covers state-tangent partials AND noise-input partials.
    // The first kTangentDim columns are F; the last NumNoiseSlots columns
    // are L (the noise-input gain), used for auto-Q assembly.
    using Jet      = ceres::Jet<double, kJetWidth>;
    using WorldR   = WorldT<double>;
    using WorldJ   = WorldT<Jet>;
    using CraftR   = CraftT<double>;
    using CraftJ   = CraftT<Jet>;

    // User-facing types: ambient reference state + tangent covariance.
    using StateVec   = Eigen::Matrix<double, kStateDim,   1>;            // 13·N
    using StateCov   = Eigen::Matrix<double, kTangentDim, kTangentDim>;  // 12·N × 12·N
    using TangentVec = Eigen::Matrix<double, kTangentDim, 1>;            // 12·N
    using NoiseGain  = Eigen::Matrix<double, kTangentDim, (NumNoiseSlots > 0 ? NumNoiseSlots : 1)>;
    using MeasVec    = Eigen::Matrix<double, MeasDim,     1>;
    using MeasCov    = Eigen::Matrix<double, MeasDim,     MeasDim>;

    EKF() : x_ref_(StateVec::Zero()), P_(StateCov::Identity()) {
        // Default reference: identity quaternion in each craft slot.
        for (int k = 0; k < NumCrafts; ++k) x_ref_(13 * k + 3) = 1.0;
    }

    // Bind the filter to its Jet shadow World + matching craft pointers.
    // Crafts must appear in the same slot order on both sides — the slot
    // order encodes the state vector layout.
    //
    // **Templated craft requirement**: every Craft passed here must be a
    // `<Scalar>`-templated class (e.g. `MyCraftT<Scalar> : CraftT<Scalar>`)
    // so the Jet shadow world can be built from `MyCraftT<Jet>`. The
    // codegen sets `scalar_templated=True` automatically when a craft is
    // wrapped by an EKF/UKF; for hand-written crafts, write your craft as
    // a single-parameter class template. A non-templated craft will fail
    // to compile in the user's bind() callsite (the std::array<CraftJ*,
    // …> conversion from MyCraft* to CraftT<Jet>* is what fails first).
    //
    // Walks the Jet-side part tree to register white-noise sources for
    // auto-Q assembly. The number of registered slots must not exceed
    // NumNoiseSlots; this is checked at bind time and throws on overflow.
    void bind(WorldJ& w_jet,
              std::array<CraftR*, NumCrafts> real_crafts,
              std::array<CraftJ*, NumCrafts> jet_crafts) {
        // The std::array<CraftR*, …> / <CraftJ*, …> parameter types
        // already enforce that the user's craft is convertible to
        // CraftT<double>*/CraftT<Jet>* — non-templated crafts fail to
        // convert and the compiler reports the brace-init mismatch at
        // the user's call site. See the doc-block above for the
        // templated-craft requirement.
        w_jet_       = &w_jet;
        crafts_real_ = real_crafts;
        crafts_jet_  = jet_crafts;

        noise_registry_.clear();
        for (int k = 0; k < NumCrafts; ++k) {
            walk_register_noise(crafts_jet_[k]->root(), noise_registry_);
        }
        // Promote registry-local slot indices to global Jet-column /
        // tangent-row indices:
        //   noise inputs go into [kNoiseColStart .. kJetWidth)
        //   bias states  go into [kBiasColStart  .. kTangentDim)
        noise_registry_.apply_slot_offsets(
            /*noise_input_offset=*/kNoiseColStart,
            /*bias_state_offset =*/kBiasColStart);

        if (noise_registry_.num_slots() > NumNoiseSlots) {
            throw std::runtime_error(
                "EKF::bind: registered noise slots (" +
                std::to_string(noise_registry_.num_slots()) +
                ") exceed NumNoiseSlots template arg (" +
                std::to_string(NumNoiseSlots) +
                "). Sum each part's noise_channels() and re-run codegen.");
        }
        if (noise_registry_.num_bias_slots() > BiasDim) {
            throw std::runtime_error(
                "EKF::bind: registered RW bias slots (" +
                std::to_string(noise_registry_.num_bias_slots()) +
                ") exceed BiasDim template arg (" +
                std::to_string(BiasDim) +
                "). Sum each part's noise_channels() and re-run codegen.");
        }
    }

    void set_state(const StateVec& x) noexcept {
        x_ref_ = x;
        renormalize_quats(x_ref_);
        mirror_to_real();
    }
    void set_covariance(const StateCov& P) noexcept { P_ = P; }

    const StateVec& state()      const noexcept { return x_ref_; }
    const StateCov& covariance() const noexcept { return P_; }

    // Per-craft slice accessors over the reference state. Each method
    // accepts either a positional index (slot order in bind()) or the
    // craft's name. Name-based lookup is O(NumCrafts); for a hot loop
    // resolve once via `idx_of(name)` and cache.
    Eigen::Matrix<double, 3, 1> position(int idx = 0) const noexcept {
        return x_ref_.template segment<3>(idx * kRigidStateDim + 0);
    }
    Eigen::Matrix<double, 3, 1> position(std::string_view name) const {
        return position(idx_of(name));
    }
    Eigen::Matrix<double, 4, 1> orientation(int idx = 0) const noexcept {
        return x_ref_.template segment<4>(idx * kRigidStateDim + 3);
    }
    Eigen::Matrix<double, 4, 1> orientation(std::string_view name) const {
        return orientation(idx_of(name));
    }
    Eigen::Matrix<double, 3, 1> vel_linear(int idx = 0) const noexcept {
        return x_ref_.template segment<3>(idx * kRigidStateDim + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_linear(std::string_view name) const {
        return vel_linear(idx_of(name));
    }
    Eigen::Matrix<double, 3, 1> vel_angular(int idx = 0) const noexcept {
        return x_ref_.template segment<3>(idx * kRigidStateDim + 10);
    }
    Eigen::Matrix<double, 3, 1> vel_angular(std::string_view name) const {
        return vel_angular(idx_of(name));
    }
    const StateVec& full_state() const noexcept { return x_ref_; }

    // Stddev accessors over the tangent covariance.
    Eigen::Matrix<double, 3, 1> position_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 0);
    }
    Eigen::Matrix<double, 3, 1> position_stddev(std::string_view name) const {
        return position_stddev(idx_of(name));
    }
    // Orientation stddev is now 3-DOF (axis-angle), not 4-DOF — the
    // tangent layout omits the redundant radial direction. Returned as a
    // (δθx, δθy, δθz) standard-deviation triple.
    Eigen::Matrix<double, 3, 1> orientation_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 3);
    }
    Eigen::Matrix<double, 3, 1> orientation_stddev(std::string_view name) const {
        return orientation_stddev(idx_of(name));
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 6);
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev(std::string_view name) const {
        return vel_linear_stddev(idx_of(name));
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 9);
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev(std::string_view name) const {
        return vel_angular_stddev(idx_of(name));
    }

    // Per-craft handle. Useful in measurement functors that read sensor
    // last_* values from a specific value-side craft.
    CraftR&       craft(int idx = 0)       noexcept { return *crafts_real_[idx]; }
    const CraftR& craft(int idx = 0) const noexcept { return *crafts_real_[idx]; }
    CraftR&       craft(std::string_view name)       { return craft(idx_of(name)); }
    const CraftR& craft(std::string_view name) const { return craft(idx_of(name)); }
    CraftJ&       craft_jet(int idx = 0)       noexcept { return *crafts_jet_[idx]; }
    const CraftJ& craft_jet(int idx = 0) const noexcept { return *crafts_jet_[idx]; }
    CraftJ&       craft_jet(std::string_view name)       { return craft_jet(idx_of(name)); }
    const CraftJ& craft_jet(std::string_view name) const { return craft_jet(idx_of(name)); }

    // Resolve a craft name to its slot index (the order it was passed
    // to bind()). Throws if the name doesn't match any bound craft.
    int idx_of(std::string_view name) const {
        for (int i = 0; i < NumCrafts; ++i) {
            if (crafts_real_[i] && crafts_real_[i]->name() == name) return i;
        }
        throw std::runtime_error(
            std::string("EKF: no craft named '") + std::string(name) + "'");
    }

    // ---- Predict ----
    //
    // Drives a full Jet-typed `WorldT::step()` with seeded Jet state.
    // Extracts (x_post_ref, F) via boxminus at the post state. When
    // NumNoiseSlots > 0, also extracts the noise-input gain L from the
    // noise Jet columns and adds L·diag(σᵢ²·dt)·Lᵀ to user-supplied Q.
    void predict(double dt, const StateCov& Q) {
        if (!w_jet_) return;
        w_jet_->clock().set_dt(static_cast<float>(dt));

        seed_jets();
        w_jet_->step();

        StateVec  x_ref_post;
        StateCov  F;
        NoiseGain L;
        extract_F_L_and_ref(x_ref_post, F, L, dt);

        // x_ref_post comes from the integrated Jet world, whose
        // kinematic_link::integrate already normalized each craft's
        // quaternion (see kinematic_link.hpp). No explicit renormalize
        // needed here — the integrator owns drift control.
        x_ref_ = x_ref_post;

        StateCov Q_total = Q;
        if constexpr (NumNoiseSlots > 0) {
            add_auto_q(dt, L, Q_total);
        }

        P_ = F * P_ * F.transpose() + Q_total;
        P_ = 0.5 * (P_ + P_.transpose().eval());

        mirror_to_real();
    }

    // Measurement update with a per-call dimension N. h(x_jet) reads a
    // 13·N ambient state vector built from the seeded Jet crafts; returns
    // an N-Jet measurement. The Jet `.v` rows are 12·N-wide and form H.
    template <int N, class MeasureH>
    void update_n(const MeasureH& h,
                  const Eigen::Matrix<double, N, 1>& z,
                  const Eigen::Matrix<double, N, N>& R) {
        if (!w_jet_) return;

        seed_jets();
        w_jet_->kinematic_and_aggregate();

        Eigen::Matrix<Jet, kStateDim, 1> x_jet;
        for (int k = 0; k < NumCrafts; ++k) {
            auto rk = crafts_jet_[k]->get_rigid_state();
            for (int i = 0; i < 13; ++i) x_jet(13 * k + i) = rk(i);
        }
        Eigen::Matrix<Jet, N, 1> z_jet = h(x_jet);

        Eigen::Matrix<double, N, 1>             z_pred;
        Eigen::Matrix<double, N, kTangentDim>   H;
        for (int i = 0; i < N; ++i) {
            z_pred(i) = z_jet(i).a;
            for (int j = 0; j < kTangentDim; ++j) H(i, j) = z_jet(i).v[j];
        }

        Eigen::Matrix<double, N, 1>           y = z - z_pred;
        Eigen::Matrix<double, N, N>           S = H * P_ * H.transpose() + R;
        Eigen::Matrix<double, kTangentDim, N> K = P_ * H.transpose() * S.inverse();

        TangentVec delta = K * y;
        inject_delta(delta);

        P_ = (StateCov::Identity() - K * H) * P_;
        P_ = 0.5 * (P_ + P_.transpose().eval());

        mirror_to_real();
    }

    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        update_n<MeasDim>(h, z, R);
    }

    // ===========================================================
    // Fused single-pass predict + update (PyPose-style linearize-at-x_ref).
    //
    //   ekf.begin_step(dt, Q);            // seeds Jets at x_ref via ⊞ +
    //                                       // identity tangent perturbations,
    //                                       // runs w_jet.kinematic_and_aggregate()
    //
    //   if (sensor.consume_fresh())
    //       ekf.add_update<N>(h_at_pre, z, R);   // reads h(x_ref) + H from Jet sensors
    //
    //   ekf.end_step();                   // advances Jet world (integrate-only),
    //                                       // extracts (x_ref_post, F) via boxminus,
    //                                       // P ← F P F^T + Q,
    //                                       // applies queued updates → δ,
    //                                       // x_ref ← x_ref_post ⊞ δ,
    //                                       // mirrors to value crafts.
    void begin_step(double dt, const StateCov& Q) {
        if (!w_jet_) return;
        w_jet_->clock().set_dt(static_cast<float>(dt));
        dt_pending_ = dt;
        Q_pending_  = Q;
        queue_.clear();
        seed_jets();
        w_jet_->kinematic_and_aggregate();
    }

    template <int N, class HReader>
    void add_update(const HReader& h,
                    const Eigen::Matrix<double, N, 1>& z,
                    const Eigen::Matrix<double, N, N>& R) {
        Eigen::Matrix<Jet, N, 1> h_jet = h(*this);

        Eigen::Matrix<double, N, 1>            h_pre;
        Eigen::Matrix<double, N, kTangentDim>  H;
        for (int i = 0; i < N; ++i) {
            h_pre(i) = h_jet(i).a;
            for (int j = 0; j < kTangentDim; ++j) H(i, j) = h_jet(i).v[j];
        }
        Eigen::Matrix<double, N, 1> y_innov = z - h_pre;

        // Each apply step adds K·y to the running δ accumulator and
        // shrinks P. The accumulated δ is injected into x_ref at end_step.
        queue_.push_back([H, y_innov, R](TangentVec& delta_acc, StateCov& P) {
            Eigen::Matrix<double, N, N>           S = H * P * H.transpose() + R;
            Eigen::Matrix<double, kTangentDim, N> K = P * H.transpose() * S.inverse();
            delta_acc += K * y_innov;
            P = (StateCov::Identity() - K * H) * P;
            P = 0.5 * (P + P.transpose().eval());
        });
    }

    void end_step() {
        if (!w_jet_) return;

        // Advance Jet world from x_pre to x_post (no re-aggregate; the
        // next begin_step re-seeds + re-evaluates).
        w_jet_->integrate(static_cast<Jet>(dt_pending_));

        StateVec  x_ref_post;
        StateCov  F;
        NoiseGain L;
        extract_F_L_and_ref(x_ref_post, F, L, dt_pending_);

        // Integrator (Jet kinematic_link) already normalized — see
        // predict() comment. No double-renorm.
        x_ref_ = x_ref_post;

        StateCov Q_total = Q_pending_;
        if constexpr (NumNoiseSlots > 0) {
            add_auto_q(dt_pending_, L, Q_total);
        }

        StateCov P = F * P_ * F.transpose() + Q_total;
        P = 0.5 * (P + P.transpose().eval());

        TangentVec delta_acc = TangentVec::Zero();
        for (auto& apply : queue_) apply(delta_acc, P);
        queue_.clear();

        inject_delta(delta_acc);

        P_ = P;
        mirror_to_real();
    }

private:
    // ---- Manifold helpers ----

    // Seed each Jet craft from x_ref using boxplus(ref, δ_jet) where δ_jet
    // is identity in the per-craft tangent slots.
    void seed_jets() {
        for (int k = 0; k < NumCrafts; ++k) {
            const int ref_off = 13 * k;
            const int tan_off = 12 * k;
            typename CraftJ::RigidState xk;

            // Position: identity Jets in slots [tan_off+0 .. tan_off+2].
            for (int i = 0; i < 3; ++i) {
                xk(i) = Jet(x_ref_(ref_off + i), tan_off + i);
            }

            // Orientation: q_seeded = q_ref ⊗ exp(δθ/2).
            // exp(0) = (1, 0, 0, 0); ∂exp/∂δθ_i at 0 = (0, 0.5·e_i).
            const double qwr = x_ref_(ref_off + 3);
            const double qxr = x_ref_(ref_off + 4);
            const double qyr = x_ref_(ref_off + 5);
            const double qzr = x_ref_(ref_off + 6);

            Jet exp_w(1.0);
            Jet exp_x(0.0); exp_x.v[tan_off + 3] = 0.5;
            Jet exp_y(0.0); exp_y.v[tan_off + 4] = 0.5;
            Jet exp_z(0.0); exp_z.v[tan_off + 5] = 0.5;

            // Hamilton product: q_ref ⊗ exp_q.
            xk(3) = qwr * exp_w - qxr * exp_x - qyr * exp_y - qzr * exp_z;
            xk(4) = qwr * exp_x + qxr * exp_w + qyr * exp_z - qzr * exp_y;
            xk(5) = qwr * exp_y - qxr * exp_z + qyr * exp_w + qzr * exp_x;
            xk(6) = qwr * exp_z + qxr * exp_y - qyr * exp_x + qzr * exp_w;

            // Velocity: identity Jets in slots [tan_off+6 .. tan_off+8].
            for (int i = 0; i < 3; ++i) {
                xk(7 + i) = Jet(x_ref_(ref_off + 7 + i), tan_off + 6 + i);
            }
            // Angular velocity: slots [tan_off+9 .. tan_off+11].
            for (int i = 0; i < 3; ++i) {
                xk(10 + i) = Jet(x_ref_(ref_off + 10 + i), tan_off + 9 + i);
            }

            crafts_jet_[k]->set_rigid_state(xk);
        }
    }

    // Extract (x_ref_post, F, L) from Jet craft state. F's rigid rows
    // (rows [0..kRigidTangentDim)) come from Jet `.v` columns
    // [0..kTangentDim) — including any bias-state cols, in case rigid
    // dynamics depend on bias (e.g. RW noise on a thruster's force).
    // L's rigid rows come from Jet `.v` columns [kNoiseColStart..kJetWidth).
    //
    // F's bias rows (rows [kBiasColStart..kTangentDim)) are set
    // analytically: bias_post = bias_pre · I + driver · σ_rw·√dt. So
    // F[bias_rows, bias_cols] = I and L[bias_rows, driver_cols] =
    // σ_rw·√dt at the relevant entries.
    void extract_F_L_and_ref(StateVec& x_ref_post, StateCov& F, NoiseGain& L,
                             double dt) {
        F.setZero();
        if constexpr (NumNoiseSlots > 0) L.setZero();

        // ---- Rigid rows: from Jet world output ----
        for (int k = 0; k < NumCrafts; ++k) {
            auto rk = crafts_jet_[k]->get_rigid_state();
            const int ref_off = 13 * k;
            const int tan_off = 12 * k;

            // Reference (post) — `.a` channel only.
            for (int i = 0; i < 13; ++i) x_ref_post(ref_off + i) = rk(i).a;

            // Helper: write the tangent row at `tan_row` from a single
            // Jet's `.v` channel, into both F (state-tangent columns) and
            // L (noise columns) as appropriate.
            auto fill_row = [&](int tan_row, const Jet& jr, double scale = 1.0) {
                for (int j = 0; j < kTangentDim; ++j) {
                    F(tan_row, j) = scale * jr.v[j];
                }
                if constexpr (NumNoiseSlots > 0) {
                    for (int j = 0; j < NumNoiseSlots; ++j) {
                        L(tan_row, j) = scale * jr.v[kNoiseColStart + j];
                    }
                }
            };

            // Position rows.
            fill_row(tan_off + 0, rk(0));
            fill_row(tan_off + 1, rk(1));
            fill_row(tan_off + 2, rk(2));

            // Orientation rows: small-angle quat error δθ ≈ 2·imag(q).
            //
            // Long form is `δθ = 2·imag(q_ref_post.conj ⊗ q_jet_post)`,
            // but `q_ref_post` is read from the same Jet's value channel
            // (`rk(*).a`), so the conjugate-product evaluates to
            // (1, 0, 0, 0) at evaluation. Its derivative wrt q_jet on
            // the imag part is the identity, so 2·imag(q_jet) gives the
            // exact same Jacobian rows at the same cost minus four
            // multiplications per axis.
            fill_row(tan_off + 3, rk(4), 2.0);
            fill_row(tan_off + 4, rk(5), 2.0);
            fill_row(tan_off + 5, rk(6), 2.0);

            // Velocity rows.
            fill_row(tan_off + 6, rk(7));
            fill_row(tan_off + 7, rk(8));
            fill_row(tan_off + 8, rk(9));
            // Angular velocity rows.
            fill_row(tan_off +  9, rk(10));
            fill_row(tan_off + 10, rk(11));
            fill_row(tan_off + 11, rk(12));
        }

        // ---- Bias rows: analytical (constant + driver) ----
        if constexpr (BiasDim > 0) {
            // F[bias, bias] = identity. Bias state evolves as
            //   bias_post = bias_pre + driver · σ_rw · √dt
            // so ∂bias_post/∂bias_pre = 1 (no coupling to other states).
            for (int b = 0; b < BiasDim; ++b) {
                F(kBiasColStart + b, kBiasColStart + b) = 1.0;
            }

            // L[bias_row, driver_col] = σ_rw · √dt at the matching axis.
            // Use the registry's RW source list to find each (bias, driver)
            // slot pair.
            if constexpr (NumNoiseSlots > 0) {
                const double sqrt_dt = std::sqrt(dt);
                for (const auto& rw : noise_registry_.rw_sources()) {
                    const double s_sqrt_dt =
                        static_cast<double>(rw.source->sigma()) * sqrt_dt;
                    // bias_slot is global tangent index;
                    // driver_slot is global Jet column index. L's column is
                    // (driver_slot - kNoiseColStart) in the noise-input range.
                    const int row_base = rw.source->state_slot();
                    const int col_base = rw.source->driver_slot() - kNoiseColStart;
                    for (int i = 0; i < rw.dim; ++i) {
                        L(row_base + i, col_base + i) = s_sqrt_dt;
                    }
                }
            }
        }
    }

    // Add the noise-driven contribution L·Σ·Lᵀ to Q_out. The σ scaling
    // is baked into L (the noise-input gain inherits σ from the
    // `Vec3 + Noise` operator's Jet injection), so Σ is identity for
    // Noise<WhiteGaussian> slots; per-policy variance comes from the
    // registry. Honors state-dependent σ — the latest σ at the moment
    // of predict is whatever the part wrote into the Noise via
    // set_sigma(), since L was built from that σ during the Jet pass.
    //
    // L has NumNoiseSlots columns; only the first registry.num_slots()
    // are populated, so this works even when the template arg
    // generously oversizes the noise width.
    void add_auto_q(double dt, const NoiseGain& L, StateCov& Q_out) {
        const int n = noise_registry_.num_slots();
        if (n == 0) return;
        const auto& input_var = noise_registry_.input_variance_diag(dt);
        Eigen::Map<const Eigen::VectorXd> sigma_diag(input_var.data(), n);
        // Only the first n columns of L are populated (the registry's
        // active slot range); the trailing NumNoiseSlots-n cols are zero
        // by setZero() at the top of extract_F_L_and_ref but skipping
        // them keeps the matrix product tight.
        Q_out.noalias() += L.leftCols(n)
                         * sigma_diag.asDiagonal()
                         * L.leftCols(n).transpose();
    }

    // x_ref ← x_ref ⊞ δ (per-craft) plus bias state updates on
    // Noise<RandomWalk> sources via δ's bias-state slots.
    void inject_delta(const TangentVec& delta) {
        for (int k = 0; k < NumCrafts; ++k) {
            const int ref_off = 13 * k;
            const int tan_off = 12 * k;

            // Position
            for (int i = 0; i < 3; ++i) x_ref_(ref_off + i) += delta(tan_off + i);

            // Orientation: q_new = q_old ⊗ exp(δθ/2).
            Eigen::Matrix<double, 3, 1> dtheta(
                delta(tan_off + 3), delta(tan_off + 4), delta(tan_off + 5));
            Eigen::Quaterniond exp_q = geom::angle_axis_to_quat<double>(dtheta);
            Eigen::Quaterniond q_old(
                x_ref_(ref_off + 3), x_ref_(ref_off + 4),
                x_ref_(ref_off + 5), x_ref_(ref_off + 6));
            Eigen::Quaterniond q_new = q_old * exp_q;
            q_new.normalize();   // numerical drift safety
            x_ref_(ref_off + 3) = q_new.w();
            x_ref_(ref_off + 4) = q_new.x();
            x_ref_(ref_off + 5) = q_new.y();
            x_ref_(ref_off + 6) = q_new.z();

            // Velocity / angular velocity
            for (int i = 0; i < 3; ++i) {
                x_ref_(ref_off + 7  + i) += delta(tan_off + 6 + i);
                x_ref_(ref_off + 10 + i) += delta(tan_off + 9 + i);
            }
        }

        // Bias state injections: for each registered RW source, add the
        // δ slice to the Jet-side Noise's stored bias estimate. The
        // Jet-side Noise's state is the EKF's running bias estimate;
        // the operator+ Jet path reads it on the next predict so the
        // measurement Jacobian carries the latest bias dependency.
        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int row_base = rw.source->state_slot();
                float* bias = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    bias[i] += static_cast<float>(delta(row_base + i));
                }
            }
        }
    }

    // Mirror x_ref into the value-side crafts so user-facing handles
    // (sensor `set_measurement` writes, telemetry reads) reflect the
    // current belief.
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

    StateVec        x_ref_;        // 13·N ambient
    StateCov        P_;            // 12·N × 12·N tangent
    NoiseRegistry   noise_registry_;

    // Fused step state: held between begin_step() and end_step().
    double                                                       dt_pending_ = 0.0;
    StateCov                                                     Q_pending_  = StateCov::Zero();
    std::vector<std::function<void(TangentVec&, StateCov&)>>     queue_;
};

} // namespace manta::estimation
