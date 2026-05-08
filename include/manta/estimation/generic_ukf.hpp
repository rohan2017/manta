#pragma once

// Generic UKF over a StateSpec.
// =============================
//
// `UKFGeneric<StateSpec, MeasDim>` is the manifold-aware UKF. Same shape
// as `EKFGeneric` but driven by the unscented transform: 2·tangent_dim+1
// sigma points propagated through the value world, no Jet machinery.
//
// The internal `UKFKernel` operates on tangent vectors of size
// `StateSpec::tangent_dim`. The wrapper lifts each sigma to the ambient
// manifold via `StateSpec::boxplus`, runs the process model (one
// `WorldT::step()` per sigma), reduces the post-ambient back to tangent
// via `StateSpec::boxminus`, and lets the kernel do the weighted-mean +
// covariance recovery.
//
// User-facing flow:
//
//   auto state = make_state().track(craft0).build();
//   UKFGeneric<decltype(state), MeasDim> ukf{state};
//
//   manta::WorldT<double> world_real;
//   /* ...build value-side scene + craft0... */
//   ukf.bind(world_real);
//
//   ukf.predict(dt, Q);
//   ukf.update<MeasDim>(measurement_functor, z, R);

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../core/craft.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "manifold.hpp"
#include "state_spec.hpp"
#include "ukf_kernel.hpp"

namespace manta::estimation {

template <class StateSpecT, int MeasDim>
class UKFGeneric {
    static_assert(MeasDim >= 0, "MeasDim must be non-negative");

public:
    using StateSpec     = StateSpecT;
    using StateSpecType = StateSpecT;

    static constexpr int kStateDim   = StateSpecT::ambient_dim;
    static constexpr int kTangentDim = StateSpecT::tangent_dim;

    using WorldR     = WorldT<double>;
    using StateVec   = typename StateSpecT::AmbientVec;
    using TangentVec = typename StateSpecT::TangentVec;
    using StateCov   = Eigen::Matrix<double, kTangentDim, kTangentDim>;
    using MeasVec    = Eigen::Matrix<double, MeasDim, 1>;
    using MeasCov    = Eigen::Matrix<double, MeasDim, MeasDim>;

    explicit UKFGeneric(StateSpecT spec,
                        double alpha = 1e-3,
                        double beta  = 2.0,
                        double kappa = 0.0) noexcept
        : spec_(std::move(spec)),
          kernel_(alpha, beta, kappa),
          x_ref_(StateVec::Zero()),
          P_(StateCov::Identity()) {
        spec_.pull_ambient(x_ref_);
    }

    void bind(WorldR& w_real) { w_real_ = &w_real; }

    void set_state(const StateVec& x) noexcept {
        x_ref_ = x;
        spec_.push_ambient(x_ref_);
    }
    void set_covariance(const StateCov& P) noexcept { P_ = P; }

    const StateVec&   state()      const noexcept { return x_ref_; }
    const StateCov&   covariance() const noexcept { return P_; }
    const StateSpecT& spec()       const noexcept { return spec_; }

    // ---- Predict ----
    //
    // Propagate sigma points (perturbations around x_ref) through the
    // value world. The first sigma (chi_0 = zero tangent) propagation
    // captures x_ref_post; subsequent sigmas reduce relative to that.
    void predict(double dt, const StateCov& Q) {
        if (!w_real_) return;
        w_real_->clock().set_dt(static_cast<float>(dt));

        // Snapshot current ambient state — every sigma evaluation
        // restores it before the next sigma.
        save_reference();

        // Kernel lives in tangent space — perturbations around x_ref are
        // zero by definition; previous-tick residuals are folded back
        // into x_ref via inject below.
        kernel_.set_state(TangentVec::Zero());
        kernel_.set_covariance(P_);

        bool have_chi0_post = false;
        StateVec x_ref_post;

        auto f = [&](const TangentVec& xi, double /*dt_*/) -> TangentVec {
            // Lift: x_full = x_ref ⊞ xi.
            apply_tangent_to_ambient(xi);
            w_real_->step();

            // Read back ambient state.
            StateVec x_full_post;
            spec_.pull_ambient(x_full_post);

            // Restore the reference for the next sigma.
            restore_reference();

            if (!have_chi0_post) {
                // chi_0 (xi = 0) propagation IS x_ref_post.
                x_ref_post     = x_full_post;
                have_chi0_post = true;
            }

            // Reduce: residual = x_full_post ⊟ x_ref_post.
            TangentVec residual;
            StateSpecT::template boxminus<double>(x_full_post.data(),
                                                  x_ref_post.data(),
                                                  residual.data());
            return residual;
        };

        kernel_.predict(f, dt, Q);

        // chi_0's propagation IS x_ref_post; the kernel's mean (now in
        // kernel.state()) is the residual offset of the weighted mean.
        // Apply both: x_ref ← x_ref_post ⊞ residual_mean.
        x_ref_ = x_ref_post;
        const TangentVec residual_mean = kernel_.state();
        inject_delta(residual_mean);

        kernel_.set_state(TangentVec::Zero());
        P_ = kernel_.covariance();

        spec_.push_ambient(x_ref_);
    }

    // ---- Update ----
    //
    // Standard sigma-point measurement update. `h` takes an ambient
    // state vector and returns a predicted measurement.
    template <int N, class HFunctor>
    void update(const HFunctor& h,
                const Eigen::Matrix<double, N, 1>& z,
                const Eigen::Matrix<double, N, N>& R) {
        kernel_.set_state(TangentVec::Zero());
        kernel_.set_covariance(P_);

        // h_lifted(xi) = h(x_ref ⊞ xi). The kernel's sigma points run
        // through this; the kernel computes y = z - z_pred and the
        // tangent correction.
        auto h_lifted = [&](const TangentVec& xi) {
            StateVec x_full;
            StateSpecT::template boxplus<double>(x_ref_.data(), xi.data(),
                                                 x_full.data());
            return h(x_full);
        };

        kernel_.template update_n<N>(h_lifted, z, R);

        const TangentVec correction = kernel_.state();
        inject_delta(correction);
        kernel_.set_state(TangentVec::Zero());
        P_ = kernel_.covariance();

        spec_.push_ambient(x_ref_);
    }

    // Per-slice ambient view over x_ref_.
    template <int I>
    auto slice_ambient() const {
        return x_ref_.template segment<slice_ambient_dim<I>()>(
            StateSpecT::template ambient_offset<I>);
    }

    template <int I>
    auto slice_stddev() const {
        constexpr int off = StateSpecT::template tangent_offset<I>;
        constexpr int n   = slice_tangent_dim<I>();
        Eigen::Matrix<double, n, 1> out;
        for (int i = 0; i < n; ++i) {
            const double v = P_(off + i, off + i);
            out(i) = std::sqrt(v > 0.0 ? v : 0.0);
        }
        return out;
    }

private:
    template <int I>
    static constexpr int slice_ambient_dim() {
        if constexpr (I + 1 < StateSpecT::num_slices) {
            return StateSpecT::template ambient_offset<I + 1>
                 - StateSpecT::template ambient_offset<I>;
        } else {
            return StateSpecT::ambient_dim
                 - StateSpecT::template ambient_offset<I>;
        }
    }
    template <int I>
    static constexpr int slice_tangent_dim() {
        if constexpr (I + 1 < StateSpecT::num_slices) {
            return StateSpecT::template tangent_offset<I + 1>
                 - StateSpecT::template tangent_offset<I>;
        } else {
            return StateSpecT::tangent_dim
                 - StateSpecT::template tangent_offset<I>;
        }
    }

    // Save the value-world ambient state into ref_snapshot_.
    void save_reference() {
        spec_.pull_ambient(ref_snapshot_);
    }

    // Restore: push ref_snapshot_ back into the value-world handles.
    void restore_reference() {
        spec_.push_ambient(ref_snapshot_);
    }

    // Apply a tangent perturbation to the value-world ambient state via
    // boxplus(ref_snapshot_, xi).
    void apply_tangent_to_ambient(const TangentVec& xi) {
        StateVec x_full;
        StateSpecT::template boxplus<double>(ref_snapshot_.data(), xi.data(),
                                             x_full.data());
        spec_.push_ambient(x_full);
    }

    void inject_delta(const TangentVec& delta) {
        StateVec x_post;
        StateSpecT::template boxplus<double>(x_ref_.data(), delta.data(),
                                             x_post.data());
        x_ref_ = x_post;
    }

    StateSpecT                       spec_;
    UKFKernel<kTangentDim, MeasDim>  kernel_;
    WorldR*                          w_real_ = nullptr;
    StateVec                         x_ref_;
    StateCov                         P_;
    StateVec                         ref_snapshot_ = StateVec::Zero();
};

} // namespace manta::estimation
