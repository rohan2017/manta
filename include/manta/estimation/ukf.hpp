#pragma once

// `manta::estimation::UKF` — error-state UKF wired against a user's
// `manta::WorldT<double>`. Estimates the joint state of every Craft via
// sigma-point propagation, with sigma points perturbing in the 12·N
// (+ BiasDim) tangent rather than the 13·N ambient. The unit
// quaternion's redundant radial direction never enters the covariance.
//
// State conventions match the manta-aware EKF:
//
//   Reference (13·N, ambient):
//     [c0.p (3) | c0.q (4) | c0.v (3) | c0.ω (3) |
//      c1.p (3) | ... ]
//
//   Tangent / error (kTangentDim = 12·N + BiasDim):
//     [c0.δp (3) | c0.δθ (3) | c0.δv (3) | c0.δω (3) |
//      c1.... | ... | bias-state slots ]
//
//   Boxplus retraction `x_ref ⊞ δ`:
//     p_full = p_ref + δp
//     q_full = q_ref ⊗ exp(δθ/2)
//     v_full = v_ref + δv
//     ω_full = ω_ref + δω
//     bias_i_full = bias_i_ref + δ_i
//
// ---- UKF vs EKF ----
//
// Compared to the manta-aware EKF, UKF has a notable structural
// advantage: it does NOT need a Jet-shadow World. Every sigma point is
// a vector of `double`s, propagated through the same `WorldT<double>`.
//
// Trade-offs:
//   * No autodiff, no Jet path — fewer compile-time templates, faster
//     builds, parts that aren't Scalar-templated still work.
//   * 2*kTangentDim+1 forward evaluations of the entire World per
//     predict; EKF does one Jet-instantiated update.
//   * Captures nonlinearity to second order; EKF only to first order.
//
// ---- Auto-Q for RW biases ----
//
// White-Gaussian noise auto-Q via the EKF's Jet-input trick is NOT
// available in UKF (no autodiff). The user supplies Q for those
// channels.
//
// Random-walk biases ARE supported: each `Noise<RandomWalk>` registered
// via `PartT::register_noise()` contributes a bias DOF to the
// augmented tangent. UKF auto-augments Q at the bias diagonal with
// σ_rw²·dt per tick. The bias estimate lives in
// `Noise<RandomWalk>::state3()` on the bound craft; the `+ rw_noise`
// operator's value branch reads it during sensor evaluation, and
// after each predict / update the UKF re-mirrors its corrected estimate.

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "../core/craft.hpp"
#include "../core/noise_registry.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "../geom/kinematic_link.hpp"   // angle_axis_to_quat
#include "ukf_kernel.hpp"

namespace manta::estimation {

template <int NumCrafts, int MeasDim, int BiasDim = 0>
class UKF {
    static_assert(NumCrafts >= 1, "manta::estimation::UKF needs at least one craft");
    static_assert(BiasDim    >= 0, "BiasDim must be non-negative");

public:
    static constexpr int kCrafts          = NumCrafts;
    static constexpr int kRigidStateDim   = 13;
    static constexpr int kRigidTangent    = 12;
    static constexpr int kStateDim        = NumCrafts * kRigidStateDim;
    static constexpr int kRigidTangentDim = NumCrafts * kRigidTangent;
    static constexpr int kBiasDim         = BiasDim;
    static constexpr int kTangentDim      = kRigidTangentDim + BiasDim;
    static constexpr int kBiasColStart    = kRigidTangentDim;

    using WorldR     = WorldT<double>;
    using CraftR     = CraftT<double>;

    using StateVec   = Eigen::Matrix<double, kStateDim,   1>;            // 13·N
    using StateCov   = Eigen::Matrix<double, kTangentDim, kTangentDim>;
    using TangentVec = Eigen::Matrix<double, kTangentDim, 1>;
    using MeasVec    = Eigen::Matrix<double, MeasDim,     1>;
    using MeasCov    = Eigen::Matrix<double, MeasDim,     MeasDim>;

    explicit UKF(double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : kernel_(alpha, beta, kappa),
          x_ref_(StateVec::Zero()),
          P_(StateCov::Identity()) {
        for (int k = 0; k < NumCrafts; ++k) x_ref_(13 * k + 3) = 1.0;
    }

    // Bind to the value World + craft pointers in slot order matching
    // the state vector. Walks the part tree to register Noise<RandomWalk>
    // sources for bias-state augmentation.
    //
    // **Templated craft requirement**: every Craft must be a
    // `<Scalar>`-templated class (the codegen sets this automatically;
    // hand-written crafts must define `MyCraftT<Scalar> : CraftT<Scalar>`).
    // The static_asserts below produce a friendly error if not.
    void bind(WorldR& w_real,
              std::array<CraftR*, NumCrafts> real_crafts) {
        // std::array<CraftR*, …> already enforces convertibility to
        // CraftT<double>*; non-templated user crafts fail to convert at
        // the call site. See the doc-block above.
        w_real_      = &w_real;
        crafts_real_ = real_crafts;

        noise_registry_.clear();
        for (int k = 0; k < NumCrafts; ++k) {
            walk_register_noise(crafts_real_[k]->root(), noise_registry_);
        }
        // Promote bias state slots to the augmented-tangent range. UKF
        // doesn't have a noise-input range (no Jet inputs) so the noise
        // input offset is irrelevant; pass 0 to keep the registry's
        // Noise<WhiteGaussian> slots harmless (UKF ignores them).
        noise_registry_.apply_slot_offsets(
            /*noise_input_offset=*/0,
            /*bias_state_offset =*/kBiasColStart);

        if (noise_registry_.num_bias_slots() > BiasDim) {
            throw std::runtime_error(
                "UKF::bind: registered RW bias slots (" +
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

    // ---- Predict ----
    //
    // Propagate every sigma point (a tangent perturbation around x_ref)
    // through the value World's step(). Lifts via boxplus; reduces via
    // boxminus to recover post-tangent. Mean-shift is folded back into
    // x_ref so the kernel always sees a zero-tangent mean.
    void predict(double dt, const StateCov& Q) {
        if (!w_real_) return;
        w_real_->clock().set_dt(static_cast<float>(dt));

        // Save reference state of crafts + biases for restoration between
        // sigma points.
        save_reference();

        // Reset kernel mean (we live in the tangent — perturbations
        // around x_ref are zero by definition).
        kernel_.set_state(TangentVec::Zero());
        kernel_.set_covariance(P_);

        // Slot 0 (mean sigma) propagation captures x_ref_post; subsequent
        // sigmas reduce relative to that.
        bool have_chi0_post = false;
        StateVec x_ref_post;
        BiasSnapshot bias_chi0_post{};

        auto f = [&](const TangentVec& xi, double /*dt_*/) -> TangentVec {
            // Lift: x_full = x_ref ⊞ xi.
            apply_tangent_to_crafts(xi);
            w_real_->step();

            // Read back ambient state.
            StateVec x_full_post;
            for (int k = 0; k < NumCrafts; ++k) {
                x_full_post.template segment<13>(13 * k) =
                    crafts_real_[k]->get_rigid_state();
            }
            BiasSnapshot bias_post{};
            read_bias_estimates(bias_post);

            // Restore craft + bias state for the next sigma.
            restore_reference();

            if (!have_chi0_post) {
                // chi_0: this propagation defines x_ref_post.
                x_ref_post = x_full_post;
                bias_chi0_post = bias_post;
                have_chi0_post = true;
            }
            return reduce_to_tangent(x_full_post, x_ref_post,
                                     bias_post, bias_chi0_post);
        };

        // Augment Q with bias-state diffusion (σ_rw²·dt).
        StateCov Q_total = Q;
        if constexpr (BiasDim > 0) {
            add_bias_process_noise(dt, Q_total);
        }

        kernel_.predict(f, dt, Q_total);

        // chi_0's propagation IS x_ref_post; the kernel's mean (now in
        // kernel.state()) is the residual offset of the weighted mean.
        // Apply both: x_ref ← x_ref_post ⊞ residual_mean.
        // chi_0 came from the integrator (sigma-point 0 path), which
        // already normalized each craft's q. No double-renorm.
        x_ref_ = x_ref_post;
        write_bias_estimates(bias_chi0_post);

        const TangentVec residual_mean = kernel_.state();
        inject_delta(residual_mean);

        // Reset kernel mean to 0; covariance stays as kernel computed.
        kernel_.set_state(TangentVec::Zero());
        P_ = kernel_.covariance();
        P_ = 0.5 * (P_ + P_.transpose().eval());

        // Push final reference state to crafts.
        mirror_to_real();
    }

    // ---- Measurement update ----
    //
    // For each sigma point: lift, run kinematic_and_aggregate, call user
    // measurement functor. Returns measurement Jet — not really, just
    // doubles for UKF. Standard tangent-space update math.
    template <int N, class MeasureH>
    void update_n(const MeasureH& h,
                  const Eigen::Matrix<double, N, 1>& z,
                  const Eigen::Matrix<double, N, N>& R) {
        if (!w_real_) return;

        save_reference();

        kernel_.set_state(TangentVec::Zero());
        kernel_.set_covariance(P_);

        auto h_wrapper = [&](const TangentVec& xi) {
            apply_tangent_to_crafts(xi);
            w_real_->kinematic_and_aggregate();

            StateVec x_full;
            for (int k = 0; k < NumCrafts; ++k) {
                x_full.template segment<13>(13 * k) =
                    crafts_real_[k]->get_rigid_state();
            }
            auto z_pred_i = h(x_full);

            restore_reference();
            return z_pred_i;
        };

        kernel_.template update_n<N>(h_wrapper, z, R);

        const TangentVec correction = kernel_.state();
        inject_delta(correction);

        kernel_.set_state(TangentVec::Zero());
        P_ = kernel_.covariance();
        P_ = 0.5 * (P_ + P_.transpose().eval());

        mirror_to_real();
    }

    template <class MeasureH>
    void update(const MeasureH& h, const MeasVec& z, const MeasCov& R) {
        update_n<MeasDim>(h, z, R);
    }

    // ---- Per-craft slice accessors ----
    //
    // Each method accepts either a positional index (slot order in
    // bind()) or the craft's name. Name lookup is O(NumCrafts).
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

    Eigen::Matrix<double, 3, 1> position_stddev(int idx = 0) const noexcept {
        return diag_stddev_segment<3>(idx * kRigidTangent + 0);
    }
    Eigen::Matrix<double, 3, 1> position_stddev(std::string_view name) const {
        return position_stddev(idx_of(name));
    }
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

    CraftR&       craft(int idx = 0)       noexcept { return *crafts_real_[idx]; }
    const CraftR& craft(int idx = 0) const noexcept { return *crafts_real_[idx]; }
    CraftR&       craft(std::string_view name)       { return craft(idx_of(name)); }
    const CraftR& craft(std::string_view name) const { return craft(idx_of(name)); }

    // Resolve a craft name to its slot index (the order it was passed
    // to bind()). Throws if the name doesn't match any bound craft.
    int idx_of(std::string_view name) const {
        for (int i = 0; i < NumCrafts; ++i) {
            if (crafts_real_[i] && crafts_real_[i]->name() == name) return i;
        }
        throw std::runtime_error(
            std::string("UKF: no craft named '") + std::string(name) + "'");
    }

private:
    // Bias snapshot is a flat float buffer of length BiasDim, indexed by
    // bias-local tangent slot (= global state_slot − kBiasColStart). One
    // entry per scalar bias dimension across all registered RW sources,
    // independent of each source's individual Dim.
    using BiasSnapshot = std::conditional_t<
        (BiasDim > 0),
        std::array<float, (BiasDim > 0) ? BiasDim : 1>,
        std::array<float, 1>>;

    // Save x_ref + biases as the per-sigma restoration target. Crafts
    // already hold x_ref (from set_state / mirror_to_real). We snapshot
    // the rigid state into ref_state_snapshot_ and biases into
    // bias_ref_snapshot_ for fast restore between sigma points.
    void save_reference() {
        for (int k = 0; k < NumCrafts; ++k) {
            ref_state_snapshot_[k] = crafts_real_[k]->get_rigid_state();
        }
        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int slot_local = rw.source->state_slot() - kBiasColStart;
                const float* src = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    bias_ref_snapshot_[slot_local + i] = src[i];
                }
            }
        }
    }

    void restore_reference() {
        for (int k = 0; k < NumCrafts; ++k) {
            crafts_real_[k]->set_rigid_state(ref_state_snapshot_[k]);
        }
        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int slot_local = rw.source->state_slot() - kBiasColStart;
                float* dst = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    dst[i] = bias_ref_snapshot_[slot_local + i];
                }
            }
        }
    }

    // Apply a tangent perturbation to crafts (rigid via boxplus) and to
    // bias estimates on Noise<RandomWalk<Dim>>. Assumes crafts currently
    // hold the reference state (call after restore_reference()).
    void apply_tangent_to_crafts(const TangentVec& xi) {
        for (int k = 0; k < NumCrafts; ++k) {
            const int tan_off = 12 * k;
            // Read current rigid (= ref) and apply boxplus.
            CraftStateT<double> s = read_craft_state(k);
            CraftTangentT<double> dt;
            dt.position    = geom::Vec3<SceneFrame, double>::from_raw(xi.template segment<3>(tan_off + 0));
            dt.orientation = geom::Vec3<SceneFrame, double>::from_raw(xi.template segment<3>(tan_off + 3));
            dt.vel_linear  = geom::Vec3<SceneFrame, double>::from_raw(xi.template segment<3>(tan_off + 6));
            dt.vel_angular = geom::Vec3<CraftFrame,  double>::from_raw(xi.template segment<3>(tan_off + 9));
            CraftStateT<double> s_full = boxplus(s, dt);
            crafts_real_[k]->set_state(s_full);
        }
        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int slot_local = rw.source->state_slot() - kBiasColStart;
                const int row        = rw.source->state_slot();
                float* dst = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    dst[i] = bias_ref_snapshot_[slot_local + i]
                           + static_cast<float>(xi(row + i));
                }
            }
        }
    }

    // Compute post tangent: residuals of x_full (and bias) against
    // chi_0's propagated state (the reference at the post-step time).
    // Both bias snapshots are flat float buffers indexed bias-locally.
    TangentVec reduce_to_tangent(
            const StateVec& x_full,
            const StateVec& x_ref_post,
            const BiasSnapshot& bias_full,
            const BiasSnapshot& bias_ref_post) {
        TangentVec out = TangentVec::Zero();
        for (int k = 0; k < NumCrafts; ++k) {
            CraftStateT<double> a = read_craft_state_from_vec(x_full,    k);
            CraftStateT<double> b = read_craft_state_from_vec(x_ref_post, k);
            CraftTangentT<double> d = boxminus(a, b);
            const int tan_off = 12 * k;
            out.template segment<3>(tan_off + 0) = d.position.raw();
            out.template segment<3>(tan_off + 3) = d.orientation.raw();
            out.template segment<3>(tan_off + 6) = d.vel_linear.raw();
            out.template segment<3>(tan_off + 9) = d.vel_angular.raw();
        }
        if constexpr (BiasDim > 0) {
            // Bias residuals are pure subtraction (no manifold curvature).
            for (int i = 0; i < BiasDim; ++i) {
                out(kBiasColStart + i) =
                    static_cast<double>(bias_full[i] - bias_ref_post[i]);
            }
        }
        return out;
    }

    // x_ref ← x_ref ⊞ δ + push δ_bias to each Noise<RandomWalk<Dim>>::state.
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

        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int row = rw.source->state_slot();
                float* bias   = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    bias[i] += static_cast<float>(delta(row + i));
                }
            }
        }
    }

    // Augment Q's bias diagonal with σ_rw²·dt — the discretized
    // continuous-time RW driver covariance.
    void add_bias_process_noise(double dt, StateCov& Q) {
        for (const auto& rw : noise_registry_.rw_sources()) {
            const double s = static_cast<double>(rw.source->sigma());
            const double s2dt = s * s * dt;
            const int row = rw.source->state_slot();
            for (int i = 0; i < rw.dim; ++i) Q(row + i, row + i) += s2dt;
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

    // Snapshot all RW bias estimates into a flat float buffer indexed
    // bias-locally (= state_slot − kBiasColStart). BiasDim total floats.
    void read_bias_estimates(BiasSnapshot& out) const {
        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int slot_local = rw.source->state_slot() - kBiasColStart;
                const float* src = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    out[slot_local + i] = src[i];
                }
            }
        } else {
            (void)out;
        }
    }

    void write_bias_estimates(const BiasSnapshot& in) {
        if constexpr (BiasDim > 0) {
            for (const auto& rw : noise_registry_.rw_sources()) {
                const int slot_local = rw.source->state_slot() - kBiasColStart;
                float* dst = rw.source->state_data();
                for (int i = 0; i < rw.dim; ++i) {
                    dst[i] = in[slot_local + i];
                }
            }
        } else {
            (void)in;
        }
    }

    static CraftStateT<double> read_craft_state_from_vec(
            const StateVec& x, int k) {
        CraftStateT<double> s;
        const int o = 13 * k;
        s.position    = geom::Vec3<SceneFrame, double>::from_raw(x.template segment<3>(o + 0));
        s.orientation = geom::Ori<SceneFrame, double>{Eigen::Quaterniond(
            x(o + 3), x(o + 4), x(o + 5), x(o + 6)).normalized()};
        s.vel_linear  = geom::Vec3<SceneFrame, double>::from_raw(x.template segment<3>(o + 7));
        s.vel_angular = geom::Vec3<CraftFrame,  double>::from_raw(x.template segment<3>(o + 10));
        return s;
    }
    CraftStateT<double> read_craft_state(int k) const {
        return crafts_real_[k]->get_state();
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

    UKFKernel<kTangentDim, MeasDim>  kernel_;
    WorldR*                          w_real_      = nullptr;
    std::array<CraftR*, NumCrafts>   crafts_real_{};

    StateVec       x_ref_;
    StateCov       P_;
    NoiseRegistry  noise_registry_;

    // Per-sigma snapshot storage. Stored as members rather than
    // function-local so the predict/update lambdas can read them
    // without heap traffic per sigma.
    std::array<typename CraftR::RigidState, NumCrafts> ref_state_snapshot_;
    BiasSnapshot                                       bias_ref_snapshot_{};
};

} // namespace manta::estimation
