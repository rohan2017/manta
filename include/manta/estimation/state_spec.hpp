#pragma once

// StateSpec: the compile-time description of a Kalman filter's state
// vector layout. Built by `make_state().track(...)...build()`.
// =====================================================================
//
// A StateSpec wraps a tuple of `manifold::*` slices PLUS pointers to the
// runtime variables each slice represents (so the filter can copy values
// in and out of its internal x_ref vector).
//
// User-facing API:
//
//   auto state = manta::estimation::make_state()
//       .track(craft0)              // → RigidBody slice (pos/ori/v/w)
//       .track(craft1)
//       .track(imu0.accel_bias())   // → BiasRandomWalk<3> slice
//       .track(imu0.gyro_bias())
//       .build();
//
// The user never types `RigidBody`, `Euclidean<3>`, `SO3`, etc. — those
// are deduced from each tracked variable's type via `slice_for<T>`.
//
// The state's compile-time properties are exposed as
//   `decltype(state)::ambient_dim`, `::tangent_dim`, `::num_slices`,
//   plus per-slice offsets via `decltype(state)::ambient_offset<I>` and
//   `tangent_offset<I>`. The EKF/UKF templates use these to size matrices.

#include <Eigen/Core>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../core/craft.hpp"
#include "../core/noise.hpp"
#include "manifold.hpp"

namespace manta::estimation {

// ---------------------------------------------------------------------
// slice_for<T> specializations — map runtime types to manifold slices.
// ---------------------------------------------------------------------

namespace manifold {
    using namespace manta::manifold;

    // CraftT<S>& → RigidBody slice.
    template <class Scalar>
    struct slice_for_impl_craft {
        using type = ::manta::manifold::RigidBody;
    };

    // Noise<RandomWalk<N>>& → BiasRandomWalk<N> slice.
    template <int N>
    struct slice_for_impl_rw {
        using type = ::manta::manifold::BiasRandomWalk<N>;
    };
}

} // namespace manta::estimation

// Inject the slice_for traits into manta::manifold so the deduction
// works on bare type names without users importing two namespaces.
namespace manta::manifold {

template <>
struct slice_for<::manta::CraftT<double>> { using type = RigidBody; };

template <int N>
struct slice_for<::manta::Noise<::manta::RandomWalk<N>>> { using type = BiasRandomWalk<N>; };

} // namespace manta::manifold


namespace manta::estimation {

// ---------------------------------------------------------------------
// TrackedVar — type-erased handle for a single registered variable.
//
// At build time we hold a (slice type, source pointer, source kind)
// triple. At runtime, the EKF reads ambient values from `source` to
// initialize x_ref, and writes back ambient values after inject_delta.
// The mechanics of the read/write are slice-specific and live in the
// `pull_ambient` / `push_ambient` static functions below.
// ---------------------------------------------------------------------

enum class TrackedKind : unsigned char {
    Craft,           // CraftT<MFloat>* — full rigid body
    RandomWalkBias,  // NoiseRandomWalkBase* — RW bias
};

struct TrackedHandle {
    TrackedKind kind;
    void*       ptr;     // type-erased pointer to the source object
    int         dim;     // RW bias dim (ignored for Craft)
};

// ---------------------------------------------------------------------
// StateSpec<Slices...> — the compile-time spec.
//
// Holds a tuple of TrackedHandle (one per slice). Used by the filter to
// pull/push ambient state and to dispatch boxplus/boxminus.
// ---------------------------------------------------------------------

template <class... Slices>
class StateSpec {
public:
    static constexpr int num_slices  = sizeof...(Slices);
    static constexpr int ambient_dim = (Slices::ambient + ... + 0);
    static constexpr int tangent_dim = (Slices::tangent + ... + 0);

    using AmbientVec = Eigen::Matrix<double, ambient_dim, 1>;
    using TangentVec = Eigen::Matrix<double, tangent_dim, 1>;

    // Compile-time slice offsets.
    template <int I>
    static constexpr int ambient_offset = []{
        constexpr int dims[] = { Slices::ambient... };
        int sum = 0;
        for (int k = 0; k < I; ++k) sum += dims[k];
        return sum;
    }();

    template <int I>
    static constexpr int tangent_offset = []{
        constexpr int dims[] = { Slices::tangent... };
        int sum = 0;
        for (int k = 0; k < I; ++k) sum += dims[k];
        return sum;
    }();

    explicit StateSpec(std::array<TrackedHandle, num_slices> handles) noexcept
        : handles_(handles) {}

    const std::array<TrackedHandle, num_slices>& handles() const noexcept {
        return handles_;
    }

    // x_post[ambient_dim] ← x_ref[ambient_dim] ⊞ delta[tangent_dim].
    // Dispatches to each slice's boxplus on its sub-range.
    template <class S>
    static void boxplus(const double* x_ref, const S* delta, S* x_post) noexcept {
        boxplus_impl<S, 0, 0, Slices...>(x_ref, delta, x_post);
    }

    // delta[tangent_dim] ← a[ambient_dim] ⊟ b[ambient_dim]. `a` carries
    // the same scalar as `delta` (Jet for F-extraction, double for the
    // value path); `b` is always the value-side linearization point.
    template <class S>
    static void boxminus(const S* a, const double* b, S* delta) noexcept {
        boxminus_impl<S, 0, 0, Slices...>(a, b, delta);
    }

    // Pull current ambient state from the tracked sources into `x`.
    void pull_ambient(AmbientVec& x) const;
    // Push ambient state from `x` back into the tracked sources.
    void push_ambient(const AmbientVec& x) const;

private:
    std::array<TrackedHandle, num_slices> handles_;

    template <class S, int AmbOff, int TanOff, class Head, class... Tail>
    static void boxplus_impl(const double* x_ref, const S* delta, S* x_post) noexcept {
        Head::template boxplus<S>(x_ref + AmbOff, delta + TanOff, x_post + AmbOff);
        if constexpr (sizeof...(Tail) > 0) {
            boxplus_impl<S, AmbOff + Head::ambient, TanOff + Head::tangent, Tail...>(
                x_ref, delta, x_post);
        }
    }

    template <class S, int AmbOff, int TanOff, class Head, class... Tail>
    static void boxminus_impl(const S* a, const double* b, S* delta) noexcept {
        Head::template boxminus<S>(a + AmbOff, b + AmbOff, delta + TanOff);
        if constexpr (sizeof...(Tail) > 0) {
            boxminus_impl<S, AmbOff + Head::ambient, TanOff + Head::tangent, Tail...>(
                a, b, delta);
        }
    }
};

// ---------------------------------------------------------------------
// pull_ambient / push_ambient
//
// Read/write a single TrackedHandle's ambient values from/to a slot in
// the AmbientVec. These dispatch on TrackedKind because the source
// object's API differs (Craft has rigid_state, RW noise has state_data).
// ---------------------------------------------------------------------

namespace detail {

inline void pull_one(const TrackedHandle& h, double* dst) noexcept {
    switch (h.kind) {
    case TrackedKind::Craft: {
        auto* c = static_cast<CraftT<double>*>(h.ptr);
        auto rs = c->get_rigid_state();   // 13-vec [p|q|v|w]
        for (int i = 0; i < 13; ++i) dst[i] = rs(i);
        break;
    }
    case TrackedKind::RandomWalkBias: {
        auto* n = static_cast<NoiseRandomWalkBase*>(h.ptr);
        const float* s = n->state_data();
        for (int i = 0; i < h.dim; ++i) dst[i] = static_cast<double>(s[i]);
        break;
    }
    }
}

inline void push_one(const TrackedHandle& h, const double* src) noexcept {
    switch (h.kind) {
    case TrackedKind::Craft: {
        auto* c = static_cast<CraftT<double>*>(h.ptr);
        typename CraftT<double>::RigidState rs;
        for (int i = 0; i < 13; ++i) rs(i) = src[i];
        c->set_rigid_state(rs);
        break;
    }
    case TrackedKind::RandomWalkBias: {
        auto* n = static_cast<NoiseRandomWalkBase*>(h.ptr);
        float* s = n->state_data();
        for (int i = 0; i < h.dim; ++i) s[i] = static_cast<float>(src[i]);
        break;
    }
    }
}

} // namespace detail

template <class... Slices>
void StateSpec<Slices...>::pull_ambient(AmbientVec& x) const {
    int off = 0;
    constexpr int amb[] = { Slices::ambient... };
    for (int i = 0; i < num_slices; ++i) {
        detail::pull_one(handles_[i], x.data() + off);
        off += amb[i];
    }
}

template <class... Slices>
void StateSpec<Slices...>::push_ambient(const AmbientVec& x) const {
    int off = 0;
    constexpr int amb[] = { Slices::ambient... };
    for (int i = 0; i < num_slices; ++i) {
        detail::push_one(handles_[i], x.data() + off);
        off += amb[i];
    }
}

// ---------------------------------------------------------------------
// StateBuilder + track() overloads.
// ---------------------------------------------------------------------

template <class... Slices>
class StateBuilder {
public:
    explicit StateBuilder(std::array<TrackedHandle, sizeof...(Slices)> handles) noexcept
        : handles_(handles) {}

    // Track a full craft (RigidBody slice). The value-side craft is
    // always `CraftT<double>` — the EKF does its arithmetic at double
    // precision regardless of how the sim was built (MFloat may be
    // float). The Jet-side craft is constructed and managed by the
    // CraftView's factory ctor; the user does not pass it here.
    //
    // LIFETIME: the tracked `craft` must outlive the filter and the
    // CraftView. The StateSpec stores a raw pointer; destroying or
    // moving the craft while a filter is bound dangles it.
    auto track(CraftT<double>& craft) {
        std::array<TrackedHandle, sizeof...(Slices) + 1> h;
        for (std::size_t i = 0; i < handles_.size(); ++i) h[i] = handles_[i];
        h.back() = TrackedHandle{
            TrackedKind::Craft,
            static_cast<void*>(&craft),
            /*dim=*/13,
        };
        return StateBuilder<Slices..., manifold::RigidBody>{h};
    }

    // Track a random-walk bias (BiasRandomWalk<Dim> slice).
    template <int Dim>
    auto track(Noise<RandomWalk<Dim>>& bias) {
        std::array<TrackedHandle, sizeof...(Slices) + 1> h;
        for (std::size_t i = 0; i < handles_.size(); ++i) h[i] = handles_[i];
        h.back() = TrackedHandle{
            TrackedKind::RandomWalkBias,
            static_cast<void*>(static_cast<NoiseRandomWalkBase*>(&bias)),
            /*dim=*/Dim,
        };
        return StateBuilder<Slices..., manifold::BiasRandomWalk<Dim>>{h};
    }

    // Finalize and return a StateSpec.
    StateSpec<Slices...> build() const {
        return StateSpec<Slices...>{handles_};
    }

private:
    std::array<TrackedHandle, sizeof...(Slices)> handles_;
};

inline StateBuilder<> make_state() noexcept {
    return StateBuilder<>{ {} };
}

} // namespace manta::estimation
