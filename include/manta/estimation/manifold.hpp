#pragma once

// Manifold primitives for the craft-agnostic estimator interface.
// ===============================================================
//
// A "manifold slice" describes one variable in a Kalman filter's state
// vector. It carries:
//
//   * `ambient` — how many doubles the variable occupies in the ambient
//     state vector (4 for a quaternion, 3 for a position, etc.).
//   * `tangent` — how many doubles its tangent (error) lives in (3 for a
//     quaternion, 3 for a position).
//   * `boxplus(x_ref, delta, x_post)` — the retraction
//         x_post = x_ref ⊞ delta
//     used by the EKF to apply a Kalman δ correction back onto the
//     ambient state, and by the UKF to lift sigma points off x_ref.
//   * `boxminus(a, b, delta)` — the inverse
//         delta = a ⊟ b
//     used by the UKF to reduce sigma-point states back to a tangent
//     vector around the mean.
//
// Built-in slices: Euclidean<N>, SO3, RigidBody, BiasRandomWalk<N>.
// Composite slices (RigidBody is one) just chain primitive slices in
// order.
//
// User code never types these directly. The estimator API exposes
// `make_state().track(craft0).track(imu.accel_bias()).build()` and
// auto-deduces the right slice from each tracked variable's type via
// the slice_for<T> trait at the bottom of this file.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/jet.h>
#include <type_traits>

#include "../core/noise.hpp"
#include "../geom/kinematic_link.hpp"   // angle_axis_to_quat (Taylor-safe)

namespace manta::manifold {

// ---------------------------------------------------------------------
// Euclidean<N> — R^N. Trivial additive boxplus.
// ---------------------------------------------------------------------
template <int N>
struct Euclidean {
    static_assert(N >= 1, "Euclidean dim must be >= 1");

    static constexpr int ambient = N;
    static constexpr int tangent = N;

    // x_post[i] = x_ref[i] + delta[i]
    template <class S>
    static void boxplus(const double* x_ref, const S* delta, S* x_post) noexcept {
        for (int i = 0; i < N; ++i) x_post[i] = S(x_ref[i]) + delta[i];
    }

    template <class S>
    static void boxminus(const S* a, const double* b, S* delta) noexcept {
        for (int i = 0; i < N; ++i) delta[i] = a[i] - S{b[i]};
    }
};

// ---------------------------------------------------------------------
// SO3 — quaternion ambient (w, x, y, z) order, axis-angle tangent.
// ---------------------------------------------------------------------
//
// Boxplus convention:  q_post = q_ref ⊗ exp(δθ / 2)
//                      where exp(δθ/2) = (cos(|δθ|/2), sin(|δθ|/2)·δθ̂/|δθ|).
//                      Implemented via geom::angle_axis_to_quat (Taylor-safe at 0).
//
// Boxminus convention: δθ = 2·imag(q_ref^conj ⊗ q_full).
struct SO3 {
    static constexpr int ambient = 4;
    static constexpr int tangent = 3;

    template <class S>
    static void boxplus(const double* x_ref, const S* dtheta, S* x_post) noexcept {
        // dq = exp(dtheta) (axis-angle → unit quat, Taylor-safe at 0).
        Eigen::Matrix<S, 3, 1> aa{dtheta[0], dtheta[1], dtheta[2]};
        Eigen::Quaternion<S> dq = geom::angle_axis_to_quat<S>(aa);

        // q_full = q_ref ⊗ dq. Cast q_ref into the Jet (or value) scalar.
        const S qrw{x_ref[0]}, qrx{x_ref[1]}, qry{x_ref[2]}, qrz{x_ref[3]};
        Eigen::Quaternion<S> qr{qrw, qrx, qry, qrz};
        Eigen::Quaternion<S> qf = qr * dq;
        x_post[0] = qf.w();
        x_post[1] = qf.x();
        x_post[2] = qf.y();
        x_post[3] = qf.z();
    }

    template <class S>
    static void boxminus(const S* a, const double* b, S* delta) noexcept {
        // delta = 2·imag(b^conj ⊗ a). For tangent recovery used by UKF
        // sigma-point reduction; also the F-extraction path on the
        // EKF (where `a` carries Jet derivatives and `b` is the value
        // linearization point).
        Eigen::Quaternion<S> qa{a[0], a[1], a[2], a[3]};
        const S bw{b[0]}, bx{b[1]}, by{b[2]}, bz{b[3]};
        Eigen::Quaternion<S> qb{bw, bx, by, bz};
        Eigen::Quaternion<S> qd = qb.conjugate() * qa;
        delta[0] = S{2.0} * qd.x();
        delta[1] = S{2.0} * qd.y();
        delta[2] = S{2.0} * qd.z();
    }
};

// ---------------------------------------------------------------------
// Compose<Slices...> — concatenate slices into one bigger slice.
//
// Slot offsets are computed at compile time. The composed slice's
// boxplus/boxminus dispatches to each child slice with the right
// sub-pointer.
// ---------------------------------------------------------------------
template <class... Slices>
struct Compose {
    static constexpr int ambient = (Slices::ambient + ... + 0);
    static constexpr int tangent = (Slices::tangent + ... + 0);

    template <class S>
    static void boxplus(const double* x_ref, const S* delta, S* x_post) noexcept {
        boxplus_impl<S, 0, 0, Slices...>(x_ref, delta, x_post);
    }

    template <class S>
    static void boxminus(const S* a, const double* b, S* delta) noexcept {
        boxminus_impl<S, 0, 0, Slices...>(a, b, delta);
    }

private:
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
// RigidBody — pos (R^3) + ori (SO3) + vel_linear (R^3) + vel_angular (R^3).
// ---------------------------------------------------------------------
// A craft's full 13-d ambient / 12-d tangent state. Matches the layout
// used by the legacy EKF.
using RigidBody = Compose<Euclidean<3>, SO3, Euclidean<3>, Euclidean<3>>;

// ---------------------------------------------------------------------
// BiasRandomWalk<N> — Euclidean state with a known per-second RW driver
// strength. Tagged so the EKF auto-augments Q with σ²·dt on its diagonal.
// ---------------------------------------------------------------------
template <int N>
struct BiasRandomWalk {
    static_assert(N >= 1);
    static constexpr int ambient = N;
    static constexpr int tangent = N;

    // Tag: enables the EKF's auto-Q diagonal contribution. The σ value
    // is read from the Noise<RandomWalk<N>> object the user tracked, so
    // it can change at runtime without re-emitting the spec.
    using IsBiasRandomWalk = void;

    template <class S>
    static void boxplus(const double* x_ref, const S* delta, S* x_post) noexcept {
        Euclidean<N>::template boxplus<S>(x_ref, delta, x_post);
    }
    template <class S>
    static void boxminus(const S* a, const double* b, S* delta) noexcept {
        Euclidean<N>::template boxminus<S>(a, b, delta);
    }
};

// ---------------------------------------------------------------------
// slice_for<T> — type trait that maps a tracked-variable type to its
// manifold slice. Specialized inside `<manta/estimation/state_spec.hpp>`
// to keep manifold.hpp free of the part / craft / Noise dependencies.
// ---------------------------------------------------------------------
template <class T> struct slice_for;
template <class T> using slice_for_t = typename slice_for<T>::type;

} // namespace manta::manifold
