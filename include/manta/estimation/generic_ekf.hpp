#pragma once

// Generic EKF over a StateSpec.
// =============================
//
// `EKFGeneric<StateSpec, MeasDim, NumNoiseSlots>` is the manifold-aware
// EKF. The state's layout, manifold structure, and tracked-variable
// identities all live in the StateSpec — the filter just does the
// linear algebra and dispatches manifold ops via StateSpec::boxplus /
// boxminus.
//
// Phase 2 scope: full craft tracking only. `track(craft)` slices contribute
// 13 ambient / 12 tangent each. White-noise auto-Q from
// `Noise<WhiteGaussian>` parts is wired through the existing
// NoiseRegistry. RW-bias-as-tracked-slice (`track(imu.accel_bias())`)
// is supported by the StateSpec but the EKF's bias-state machinery is
// deferred to Phase 5; for now register RW noises through parts and let
// the legacy EKF's bias path handle them. The two filters coexist
// during the migration.
//
// User-facing flow:
//
//   auto state = make_state()
//       .track(craft0)               // → RigidBody slice
//       .track(craft1)               // → RigidBody slice
//       .build();
//
//   constexpr int kNoiseSlots = /* sum of registered white noises */;
//   EKFGeneric<decltype(state), MeasDim, kNoiseSlots> ekf{state};
//
//   // CraftView constructs the Jet shadow internally — no manual jet
//   // world or void* casts. Pass a generic-Scalar craft factory.
//   CraftView<decltype(ekf), 0> view0(ekf, [&](auto& w) {
//       auto c = std::make_unique<MyCraft<typename decltype(w)::Scalar>>(...);
//       w.create_scene().add_craft(*c);
//       return c;
//   });
//
//   ekf.predict(dt, Q);
//   ekf.update<MeasDim>(measurement_functor, z, R);

#include <Eigen/Core>
#include <ceres/jet.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../core/craft.hpp"
#include "../core/noise.hpp"
#include "../core/noise_registry.hpp"
#include "../core/scene.hpp"
#include "../core/world.hpp"
#include "manifold.hpp"
#include "measurement.hpp"
#include "reading.hpp"
#include "state_spec.hpp"

namespace manta::estimation {

template <class StateSpecT, int MeasDim, int NumNoiseSlots = 0>
class EKFGeneric {
    static_assert(MeasDim >= 0, "MeasDim must be non-negative");
    static_assert(NumNoiseSlots >= 0, "NumNoiseSlots must be non-negative");

public:
    // ---- Spec accessor (used by CraftView) ----
    using StateSpec = StateSpecT;
    // Convenience alias for clients (older naming).
    using StateSpecType = StateSpecT;

    // ---- Compile-time dimensions ----
    static constexpr int kStateDim     = StateSpecT::ambient_dim;
    static constexpr int kTangentDim   = StateSpecT::tangent_dim;
    static constexpr int kNoiseSlots   = NumNoiseSlots;
    static constexpr int kJetWidth     = kTangentDim + NumNoiseSlots;
    static constexpr int kNoiseColStart = kTangentDim;

    using Jet      = ceres::Jet<double, kJetWidth>;
    using WorldJ   = WorldT<Jet>;

    using StateVec   = typename StateSpecT::AmbientVec;
    using TangentVec = typename StateSpecT::TangentVec;
    using StateCov   = Eigen::Matrix<double, kTangentDim, kTangentDim>;
    using NoiseGain  = Eigen::Matrix<double, kTangentDim,
                                     (NumNoiseSlots > 0 ? NumNoiseSlots : 1)>;
    using MeasVec    = Eigen::Matrix<double, MeasDim, 1>;
    using MeasCov    = Eigen::Matrix<double, MeasDim, MeasDim>;

    explicit EKFGeneric(StateSpecT spec) noexcept
        : spec_(std::move(spec)),
          x_ref_(StateVec::Zero()),
          P_(StateCov::Identity()) {
        // Pull initial ambient state from the tracked sources so user
        // setup of crafts (CraftT::set_rigid_state) is reflected.
        spec_.pull_ambient(x_ref_);
    }

    // The EKF owns its Jet shadow world. CraftView's factory ctor uses
    // this reference to populate the world with Jet-typed crafts.
    WorldJ&       jet_world()       noexcept { return jet_world_; }
    const WorldJ& jet_world() const noexcept { return jet_world_; }

    // Register a Jet-side craft for slice `I` (called by CraftView; not
    // typically invoked directly by users). The slice index must match
    // the StateSpec's `track(...)` order. Bindings finalize lazily on
    // the next `predict()` / `run_pending_updates()` call.
    template <int I>
    void register_jet_craft(CraftT<Jet>* jet_craft) {
        static_assert(I >= 0 && I < StateSpecT::num_slices,
                      "register_jet_craft: slice index out of range");
        if (!jet_craft) {
            throw std::runtime_error(
                "EKFGeneric::register_jet_craft: null craft pointer");
        }
        jet_handles_[I] = static_cast<void*>(jet_craft);
        finalized_ = false;
    }

private:
    // Finalize Jet-side bindings: walk the registered Jet crafts to populate
    // the noise registry (white noise) and the RW-bias slot table. Called
    // lazily on the first predict()/run_pending_updates() after a new
    // register_jet_craft. Throws if any tracked craft slice lacks a
    // registered Jet handle, or if the total noise-slot count exceeds the
    // user's NumNoiseSlots template arg.
    void finalize_bindings() {
        // Walk every Jet-side craft's parts to register white-noise sources
        // for auto-Q. RW biases are skipped — they're tracked as
        // BiasRandomWalk slices and wired up below.
        noise_registry_.clear();
        noise_registry_.set_skip_random_walk(true);
        const auto& src_handles = spec_.handles();
        for (int i = 0; i < StateSpecT::num_slices; ++i) {
            if (src_handles[i].kind != TrackedKind::Craft) continue;
            if (!jet_handles_[i]) {
                throw std::runtime_error(
                    "EKFGeneric: tracked craft slice " + std::to_string(i) +
                    " has no registered Jet counterpart. Construct a "
                    "CraftView<Filter, " + std::to_string(i) +
                    "> with a craft factory before predict().");
            }
            auto* cj = static_cast<CraftT<Jet>*>(jet_handles_[i]);
            walk_register_noise(cj->root(), noise_registry_);
        }
        noise_registry_.apply_slot_offsets(
            /*noise_input_offset=*/kNoiseColStart,
            /*bias_state_offset =*/0);

        rw_bindings_.clear();
        int next_driver_slot_local = noise_registry_.num_slots();
        std::array<int, StateSpecT::num_slices> tan_offsets =
            compute_tangent_offsets();
        for (int i = 0; i < StateSpecT::num_slices; ++i) {
            const auto& h = src_handles[i];
            if (h.kind != TrackedKind::RandomWalkBias) continue;
            auto* nv = static_cast<NoiseRandomWalkBase*>(h.ptr);
            std::string_view rw_name;
            std::string      part_name;
            int              owner_craft_slice = -1;
            bool found = false;
            for (int k = 0; k < StateSpecT::num_slices && !found; ++k) {
                if (src_handles[k].kind != TrackedKind::Craft) continue;
                auto* cv = static_cast<CraftT<double>*>(src_handles[k].ptr);
                find_owning_rw_in(cv->root(), nv, rw_name, part_name, found);
                if (found) {
                    owner_craft_slice = k;
                }
            }
            if (!found) {
                throw std::runtime_error(
                    "EKFGeneric: tracked RW bias not found in any "
                    "tracked craft's parts. Did you call track(part.bias()) "
                    "for a part that wasn't also track()'d at the craft level?");
            }
            auto* cj = static_cast<CraftT<Jet>*>(jet_handles_[owner_craft_slice]);
            NoiseRandomWalkBase* nj = nullptr;
            find_jet_rw_in(cj->root(), part_name, rw_name, nj);
            if (!nj) {
                throw std::runtime_error(
                    "EKFGeneric: Jet-side counterpart of RW bias '" +
                    part_name + "." + std::string(rw_name) +
                    "' not found. Sim and Jet part trees must mirror.");
            }
            nj->set_state_slot(tan_offsets[i]);
            nj->set_driver_slot(kNoiseColStart + next_driver_slot_local);
            rw_bindings_.push_back(RWBinding{
                /*jet_noise=*/nj,
                /*state_slot=*/tan_offsets[i],
                /*driver_slot_local=*/next_driver_slot_local,
                /*dim=*/h.dim,
            });
            next_driver_slot_local += h.dim;
        }

        const int total_slots = next_driver_slot_local;
        if (total_slots > NumNoiseSlots) {
            throw std::runtime_error(
                "EKFGeneric: total noise slots (" +
                std::to_string(total_slots) +
                ") exceed NumNoiseSlots template arg (" +
                std::to_string(NumNoiseSlots) + ").");
        }

        finalized_ = true;
    }

    void finalize_if_needed() {
        if (!finalized_) finalize_bindings();
    }

    std::array<int, StateSpecT::num_slices> compute_tangent_offsets() {
        std::array<int, StateSpecT::num_slices> out{};
        compute_tangent_offsets_impl<0>(out);
        return out;
    }
    template <int I>
    void compute_tangent_offsets_impl(
            std::array<int, StateSpecT::num_slices>& out) {
        if constexpr (I < StateSpecT::num_slices) {
            out[I] = StateSpecT::template tangent_offset<I>;
            compute_tangent_offsets_impl<I + 1>(out);
        }
    }

public:

    void set_state(const StateVec& x) noexcept {
        x_ref_ = x;
        spec_.push_ambient(x_ref_);
    }
    void set_covariance(const StateCov& P) noexcept { P_ = P; }

    const StateVec&   state()      const noexcept { return x_ref_; }
    const StateCov&   covariance() const noexcept { return P_; }
    const StateSpecT& spec()       const noexcept { return spec_; }

    // Per-slice ambient slice over x_ref_. Call as `ekf.slice<I>()`
    // where I is the slice index in the StateSpec's slice list.
    template <int I>
    auto slice_ambient() const {
        return x_ref_.template segment<slice_ambient_dim<I>()>(
            StateSpecT::template ambient_offset<I>);
    }

    // Per-slice tangent stddev — sqrt of covariance diagonal in this
    // slice's tangent slot range.
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

    // ---- Predict ----
    void predict(double dt, const StateCov& Q) {
        if (!jet_world_.crafts_added()) return;
        finalize_if_needed();
        jet_world_.clock().set_dt(static_cast<float>(dt));
        dt_for_L_ = dt;

        seed_jets();
        jet_world_.step();

        StateVec  x_ref_post;
        StateCov  F;
        NoiseGain L;
        extract_F_L_and_ref(x_ref_post, F, L);

        x_ref_ = x_ref_post;

        StateCov Q_total = Q;
        if constexpr (NumNoiseSlots > 0) {
            add_auto_q(dt, L, Q_total);
        }

        P_ = F * P_ * F.transpose() + Q_total;
        P_ = 0.5 * (P_ + P_.transpose().eval());

        spec_.push_ambient(x_ref_);
    }

    // ---- Measurement registration ----
    //
    // Bind a Measurement (the model — h(x) cache + R σ source) to a
    // Reading (where z comes from). At bind() time the EKF resolves the
    // Jet-side Measurement counterpart by walking the Jet-side craft's
    // part tree and matching by part-name / measurement-name. The Jet
    // measurement is what `run_pending_updates()` reads each tick.
    //
    // Usage:
    //   ekf.measure(&est_craft.imu().accel, reading_from(sim_craft.imu().accel));
    template <int Dim>
    void measure(Measurement* model_value_side, Reading<Dim> reading) {
        if (!model_value_side) {
            throw std::runtime_error("EKFGeneric::measure: model is null");
        }
        if (Dim != model_value_side->dim) {
            throw std::runtime_error(
                "EKFGeneric::measure: Reading dim doesn't match Measurement dim");
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

    // Run all registered measurement updates this tick. Internally:
    //   1. seed_jets() — populate Jet-side ambient state with identity Jets.
    //   2. w_jet.kinematic_and_aggregate() — runs every Jet-side part's
    //      update(), filling the Jet-side Measurement caches.
    //   3. Iterate bindings: pull z + freshness from Reading; if fresh,
    //      read h(x_pre) + H + R from the Jet-side Measurement; apply
    //      Kalman update; accumulate δ and shrink P.
    void run_pending_updates() {
        if (!jet_world_.crafts_added()) return;
        if (bindings_.empty()) return;
        finalize_if_needed();

        seed_jets();
        jet_world_.kinematic_and_aggregate();

        Eigen::VectorXd buf_a(64);    // scratch — resized per binding
        Eigen::VectorXd buf_v(kJetWidth);

        for (auto& b : bindings_) {
            auto [z, fresh] = b.pull_z();
            if (!fresh) continue;

            // Resolve at first use — the value-side Measurement we
            // captured at registration carries the part name; we walk
            // the Jet-side craft tree once to find the matching
            // Measurement on the Jet-typed instance.
            if (!b.model_jet) b.model_jet = resolve_jet_measurement(b);
            if (!b.model_jet) continue;

            const int n = b.dim;
            buf_a.conservativeResize(n);

            // h(x_pre) — value channel of the Jet's ambient.
            b.model_jet->read_value(buf_a.data());

            // H — the .v channel rows (kTangentDim columns of interest;
            // the trailing kNoiseSlots columns are noise-input gains we
            // don't need here since R is taken from the σ field).
            Eigen::Matrix<double, Eigen::Dynamic, kTangentDim> H(n, kTangentDim);
            for (int row = 0; row < n; ++row) {
                b.model_jet->read_jacobian_row(row, buf_v.data(), kJetWidth);
                for (int j = 0; j < kTangentDim; ++j) H(row, j) = buf_v(j);
            }

            // R = σ²·I_n.
            const double sigma = b.model_jet->r_sigma();
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n) * sigma * sigma;

            // Innovation.
            Eigen::VectorXd y = z - buf_a;

            // S = H P Hᵀ + R.
            Eigen::MatrixXd S = H * P_ * H.transpose() + R;
            // K = P Hᵀ S⁻¹.
            Eigen::Matrix<double, kTangentDim, Eigen::Dynamic> K =
                P_ * H.transpose() * S.ldlt().solve(
                    Eigen::MatrixXd::Identity(n, n));

            TangentVec delta = K * y;
            inject_delta(delta);

            P_ = (StateCov::Identity() - K * H) * P_;
            P_ = 0.5 * (P_ + P_.transpose().eval());
        }

        spec_.push_ambient(x_ref_);
    }

    // ---- Single-shot Update (legacy / explicit-functor path) ----
    template <int N, class HFunctor>
    void update(const HFunctor& h,
                const Eigen::Matrix<double, N, 1>& z,
                const Eigen::Matrix<double, N, N>& R) {
        // Seed the ambient state with identity tangent Jets.
        Eigen::Matrix<Jet, kTangentDim, 1> delta_jet;
        for (int i = 0; i < kTangentDim; ++i) delta_jet(i) = Jet(0.0, i);

        Eigen::Matrix<Jet, kStateDim, 1> x_jet;
        StateSpecT::template boxplus<Jet>(x_ref_.data(), delta_jet.data(),
                                          x_jet.data());

        Eigen::Matrix<Jet, N, 1> z_jet = h(x_jet);

        Eigen::Matrix<double, N, 1>           z_pred;
        Eigen::Matrix<double, N, kTangentDim> H;
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

        spec_.push_ambient(x_ref_);
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

    // Seed Jet-side handles from x_ref ⊞ identity-Jet δ.
    void seed_jets() {
        Eigen::Matrix<Jet, kTangentDim, 1> delta_jet;
        for (int i = 0; i < kTangentDim; ++i) delta_jet(i) = Jet(0.0, i);

        Eigen::Matrix<Jet, kStateDim, 1> x_jet_full;
        StateSpecT::template boxplus<Jet>(x_ref_.data(), delta_jet.data(),
                                          x_jet_full.data());
        push_jet_ambient(x_jet_full);
    }

    // Read post-step Jet ambient + boxminus to extract (x_ref_post, F, L).
    void extract_F_L_and_ref(StateVec& x_ref_post, StateCov& F, NoiseGain& L) {
        F.setZero();
        if constexpr (NumNoiseSlots > 0) L.setZero();

        Eigen::Matrix<Jet, kStateDim, 1> x_jet_post;
        pull_jet_ambient(x_jet_post);

        // .a channel of the post-state IS x_ref_post.
        for (int i = 0; i < kStateDim; ++i) x_ref_post(i) = x_jet_post(i).a;

        // boxminus(x_post, x_ref_post) on the Jet path yields a Jet whose
        // .v carries (F | L).
        Eigen::Matrix<Jet, kTangentDim, 1> delta_jet;
        StateSpecT::template boxminus<Jet>(x_jet_post.data(),
                                           x_ref_post.data(),
                                           delta_jet.data());
        for (int i = 0; i < kTangentDim; ++i) {
            for (int j = 0; j < kTangentDim; ++j) F(i, j) = delta_jet(i).v[j];
            if constexpr (NumNoiseSlots > 0) {
                for (int j = 0; j < NumNoiseSlots; ++j) {
                    L(i, j) = delta_jet(i).v[kNoiseColStart + j];
                }
            }
        }

        // Bias-driver gain: each tracked RW slice contributes σ_rw·√dt
        // at L[state_slot..+dim, driver_slot..+dim] (identity scaling).
        // The full Q = L·diag(σ²)·Lᵀ then picks up σ_rw²·dt on the bias
        // diagonal automatically via add_auto_q.
        if constexpr (NumNoiseSlots > 0) {
            const double sqrt_dt = std::sqrt(dt_for_L_);
            for (const auto& rw : rw_bindings_) {
                const double s = static_cast<double>(rw.jet_noise->sigma());
                const double s_sqrt_dt = s * sqrt_dt;
                for (int i = 0; i < rw.dim; ++i) {
                    L(rw.state_slot + i, rw.driver_slot_local + i) = s_sqrt_dt;
                }
            }
        }
    }

    void add_auto_q(double /*dt*/, const NoiseGain& L, StateCov& Q_out) {
        // Total active noise driver slots = white slots from registry +
        // RW bias driver slots from rw_bindings_. Both contribute via L.
        int n = noise_registry_.num_slots();
        for (const auto& rw : rw_bindings_) {
            n = std::max(n, rw.driver_slot_local + rw.dim);
        }
        if (n == 0) return;
        // All slots have unit variance — σ is already baked into L's
        // entries (σ for white, σ·√dt for RW driver). Q = L·Iₙ·Lᵀ.
        Q_out.noalias() += L.leftCols(n) * L.leftCols(n).transpose();
    }

    void inject_delta(const TangentVec& delta) {
        StateVec x_post;
        StateSpecT::template boxplus<double>(x_ref_.data(), delta.data(),
                                             x_post.data());
        x_ref_ = x_post;
    }

    void push_jet_ambient(const Eigen::Matrix<Jet, kStateDim, 1>& x) {
        const auto& handles = spec_.handles();
        int off = 0;
        for (int i = 0; i < StateSpecT::num_slices; ++i) {
            const auto& h = handles[i];
            if (h.kind == TrackedKind::Craft) {
                auto* cj = static_cast<CraftT<Jet>*>(jet_handles_[i]);
                typename CraftT<Jet>::RigidState rs;
                for (int j = 0; j < 13; ++j) rs(j) = x(off + j);
                cj->set_rigid_state(rs);
                off += 13;
            } else {
                // BiasRandomWalk: not yet wired into the Jet path
                // (Phase 5 will route this through the registry).
                off += h.dim;
            }
        }
    }

    void pull_jet_ambient(Eigen::Matrix<Jet, kStateDim, 1>& x) const {
        const auto& handles = spec_.handles();
        int off = 0;
        for (int i = 0; i < StateSpecT::num_slices; ++i) {
            const auto& h = handles[i];
            if (h.kind == TrackedKind::Craft) {
                auto* cj = static_cast<CraftT<Jet>*>(jet_handles_[i]);
                auto rs = cj->get_rigid_state();
                for (int j = 0; j < 13; ++j) x(off + j) = rs(j);
                off += 13;
            } else {
                // BiasRandomWalk placeholder — see header note.
                auto* nb = static_cast<NoiseRandomWalkBase*>(handles[i].ptr);
                const float* sv = nb->state_data();
                for (int j = 0; j < h.dim; ++j) {
                    x(off + j) = Jet(static_cast<double>(sv[j]));
                }
                off += h.dim;
            }
        }
    }

    // Per-measurement registration. Stored in registration order;
    // resolved Jet-side Measurement* is filled lazily on first use.
    struct Binding {
        int dim;
        std::string  meas_name;
        Measurement* model_value = nullptr;
        Measurement* model_jet   = nullptr;
        std::function<std::pair<Eigen::VectorXd, bool>()> pull_z;
    };

    // Walk the Jet-side craft for a given binding's value-side
    // Measurement, find the part that owns the same Measurement object
    // by walking the *value-side* part trees in parallel slot order,
    // then look up the matching Measurement on the Jet-side part by
    // name.
    Measurement* resolve_jet_measurement(const Binding& b) {
        // Locate which value-side craft owns model_value.
        const auto& handles = spec_.handles();
        for (int i = 0; i < StateSpecT::num_slices; ++i) {
            if (handles[i].kind != TrackedKind::Craft) continue;
            auto* craft_real = static_cast<CraftT<double>*>(handles[i].ptr);
            auto* part_real  = find_owning_part(craft_real, b.model_value);
            if (!part_real) continue;
            // Found. Walk the Jet-side craft and look for the part with
            // the same name.
            auto* craft_jet = static_cast<CraftT<Jet>*>(jet_handles_[i]);
            auto* part_jet  = find_part_by_name(craft_jet, part_real->name());
            if (!part_jet) continue;
            // Look up the Jet-side Measurement by name.
            for (Measurement* const m : part_jet->measurements()) {
                if (m && m->name == b.meas_name) return m;
            }
        }
        return nullptr;
    }

    static PartT<double>* find_owning_part(CraftT<double>* craft,
                                           const Measurement* m) {
        return find_owning_part_in(craft->root(), m);
    }
    static PartT<double>* find_owning_part_in(PartT<double>& part,
                                               const Measurement* m) {
        for (Measurement* const x : part.measurements()) {
            if (x == m) return &part;
        }
        if (auto* kids = part.children()) {
            for (auto& child : *kids) {
                if (auto* found = find_owning_part_in(*child, m)) return found;
            }
        }
        return nullptr;
    }
    static PartT<Jet>* find_part_by_name(CraftT<Jet>* craft,
                                          std::string_view name) {
        return find_part_by_name_in(craft->root(), name);
    }
    static PartT<Jet>* find_part_by_name_in(PartT<Jet>& part,
                                             std::string_view name) {
        if (part.name() == name) return &part;
        if (auto* kids = part.children()) {
            for (auto& child : *kids) {
                if (auto* found = find_part_by_name_in(*child, name)) return found;
            }
        }
        return nullptr;
    }

    // Walk a value-side part subtree to find the part that owns `nv`
    // and what name the part registered it under. Sets `out_*` and
    // `found` if hit.
    static void find_owning_rw_in(PartT<double>& part,
                                   const NoiseRandomWalkBase* nv,
                                   std::string_view& out_rw_name,
                                   std::string& out_part_name,
                                   bool& found) {
        for (const auto& named : part.random_walks()) {
            if (named.noise == nv) {
                out_rw_name   = named.name;
                out_part_name = part.name();
                found = true;
                return;
            }
        }
        if (auto* kids = part.children()) {
            for (auto& child : *kids) {
                find_owning_rw_in(*child, nv, out_rw_name, out_part_name, found);
                if (found) return;
            }
        }
    }

    // Walk a Jet-side part subtree, find a part by name, find its RW
    // noise registered under `rw_name`. Writes pointer into `out`.
    static void find_jet_rw_in(PartT<Jet>& part,
                                const std::string& part_name,
                                std::string_view rw_name,
                                NoiseRandomWalkBase*& out) {
        if (part.name() == part_name) {
            for (const auto& named : part.random_walks()) {
                if (named.name == rw_name) {
                    out = named.noise;
                    return;
                }
            }
        }
        if (auto* kids = part.children()) {
            for (auto& child : *kids) {
                find_jet_rw_in(*child, part_name, rw_name, out);
                if (out) return;
            }
        }
    }

    // Per RW-bias slice resolved at bind time.
    struct RWBinding {
        NoiseRandomWalkBase* jet_noise;
        int state_slot;
        int driver_slot_local;
        int dim;
    };

    StateSpecT             spec_;
    StateVec               x_ref_;
    StateCov               P_;
    WorldJ                 jet_world_{};
    std::array<void*, StateSpecT::num_slices> jet_handles_{};
    NoiseRegistry          noise_registry_;
    std::vector<Binding>   bindings_;
    std::vector<RWBinding> rw_bindings_;
    bool                   finalized_ = false;
    double                 dt_for_L_ = 0.0;
};

} // namespace manta::estimation
