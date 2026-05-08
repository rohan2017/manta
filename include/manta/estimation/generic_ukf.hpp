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
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../core/craft.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "manifold.hpp"
#include "measurement.hpp"
#include "reading.hpp"
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

    // ---- Measurement registration (parallel to EKFGeneric) ----
    template <int Dim>
    void measure(Measurement* model_value_side, Reading<Dim> reading) {
        if (!model_value_side) {
            throw std::runtime_error("UKFGeneric::measure: model is null");
        }
        if (Dim != model_value_side->dim) {
            throw std::runtime_error(
                "UKFGeneric::measure: Reading dim doesn't match Measurement dim");
        }
        bindings_.push_back(Binding{
            .dim = Dim,
            .meas_name = model_value_side->name,
            .model_value = model_value_side,
            .pull_z = [r = std::move(reading)]() mutable {
                auto s = r.pull();
                Eigen::VectorXd z(Dim);
                for (int i = 0; i < Dim; ++i) z(i) = s.z(i);
                return std::pair{z, s.fresh};
            },
        });
    }

    void run_pending_updates() {
        if (!w_real_) return;
        if (bindings_.empty()) return;

        for (auto& b : bindings_) {
            auto [z, fresh] = b.pull_z();
            if (!fresh) continue;

            // h(x) closure that runs the value-side part's update on
            // boxplus(x_ref, xi) and reads the typed measurement field.
            // The kernel's sigma-point sweep calls this 2*tangent+1 times.
            //
            // Resolve the value-side part once per binding (by walking
            // the tracked crafts) — same lookup as the EKF, but here
            // we read the value-typed cache directly via the
            // Measurement's read_value.
            auto* model = b.model_value;
            const int n = model->dim;

            auto h = [model, n, this](const StateVec& x_full) {
                Eigen::VectorXd hv(n);
                // Push x_full into the bound crafts so the part's
                // update() reads the right state, then run the world
                // step to populate caches, then read the measurement.
                spec_.push_ambient(x_full);
                w_real_->kinematic_and_aggregate();
                model->read_value(hv.data());
                return hv;
            };

            // Fixed R = σ²·I per the model's noise.
            const double sigma = model->r_sigma();
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n) * sigma * sigma;

            // Save state, run the kernel update over xi, restore state.
            kernel_.set_state(TangentVec::Zero());
            kernel_.set_covariance(P_);

            auto h_lifted = [&](const TangentVec& xi) {
                StateVec x_full;
                StateSpecT::template boxplus<double>(x_ref_.data(),
                                                     xi.data(),
                                                     x_full.data());
                return h(x_full);
            };
            // Dispatch to a runtime-N update via a switch on dim.
            // Common dims: 1, 3, 6.
            if (n == 1) {
                Eigen::Matrix<double, 1, 1> z1; z1(0) = z(0);
                Eigen::Matrix<double, 1, 1> R1; R1(0, 0) = R(0, 0);
                auto h1 = [&](const TangentVec& xi) {
                    auto v = h_lifted(xi); Eigen::Matrix<double, 1, 1> r;
                    r(0) = v(0); return r;
                };
                kernel_.template update_n<1>(h1, z1, R1);
            } else if (n == 3) {
                Eigen::Matrix<double, 3, 1> z3 = z;
                Eigen::Matrix<double, 3, 3> R3 = R;
                auto h3 = [&](const TangentVec& xi) {
                    auto v = h_lifted(xi); Eigen::Matrix<double, 3, 1> r;
                    for (int i = 0; i < 3; ++i) r(i) = v(i); return r;
                };
                kernel_.template update_n<3>(h3, z3, R3);
            } else if (n == 6) {
                Eigen::Matrix<double, 6, 1> z6 = z;
                Eigen::Matrix<double, 6, 6> R6 = R;
                auto h6 = [&](const TangentVec& xi) {
                    auto v = h_lifted(xi); Eigen::Matrix<double, 6, 1> r;
                    for (int i = 0; i < 6; ++i) r(i) = v(i); return r;
                };
                kernel_.template update_n<6>(h6, z6, R6);
            } else {
                throw std::runtime_error(
                    "UKFGeneric::run_pending_updates: dim " +
                    std::to_string(n) +
                    " not yet supported (add a switch arm for it).");
            }

            const TangentVec correction = kernel_.state();
            inject_delta(correction);
            kernel_.set_state(TangentVec::Zero());
            P_ = kernel_.covariance();
        }

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

    struct Binding {
        int dim;
        std::string  meas_name;
        Measurement* model_value = nullptr;
        std::function<std::pair<Eigen::VectorXd, bool>()> pull_z;
    };

    StateSpecT                       spec_;
    UKFKernel<kTangentDim, MeasDim>  kernel_;
    WorldR*                          w_real_ = nullptr;
    StateVec                         x_ref_;
    StateCov                         P_;
    StateVec                         ref_snapshot_ = StateVec::Zero();
    std::vector<Binding>             bindings_;
};

} // namespace manta::estimation
