#pragma once

// Scalar-bridge helpers between the value-typed sim path and Scalar-typed
// (value or Jet) crafts. The estimator path queries value-typed fields and
// planet poses from inside Jet-templated craft code; these helpers strip
// the autodiff layer once at the boundary instead of repeating the
// `if constexpr` branch at every call site.

#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "../core/types.hpp"
#include "kinematic_link.hpp"
#include "ori.hpp"
#include "static_link.hpp"
#include "vec3.hpp"

namespace manta::geom {

// ----- Vec3 / Ori bridging -----

// MFloat ← any Scalar. For floating-point Scalars this is a value cast; for
// Jet Scalars it strips `.a` to recover the value channel.
template <class Scalar>
inline MFloat strip_to_real(Scalar s) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) return MFloat(s);
    else                                            return MFloat(s.a);
}

template <class Frame, class Scalar>
inline Vec3<Frame, MFloat> cast_to_real(const Vec3<Frame, Scalar>& v) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return Vec3<Frame, MFloat>::from_raw(v.raw().template cast<MFloat>());
    } else {
        Eigen::Matrix<MFloat, 3, 1> r;
        for (int i = 0; i < 3; ++i) r(i) = MFloat(v.raw()(i).a);
        return Vec3<Frame, MFloat>::from_raw(r);
    }
}

template <class Frame, class Scalar>
inline Ori<Frame, MFloat> cast_to_real(const Ori<Frame, Scalar>& q) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return Ori<Frame, MFloat>{q.raw().template cast<MFloat>()};
    } else {
        const auto& r = q.raw();
        return Ori<Frame, MFloat>{Eigen::Quaternion<MFloat>(
            MFloat(r.w().a), MFloat(r.x().a), MFloat(r.y().a), MFloat(r.z().a))};
    }
}

// ----- MFloat → Scalar (no autodiff sensitivity) -----
//
// The inverse direction: lift a value value into a Scalar context with zero
// derivatives. Used when a Jet-templated tick needs to consume a value that
// is not part of the EKF state (e.g. a planet's pose, a field's stationary
// reading) — we want it to participate in arithmetic but not in the
// Jacobian.

template <class Scalar, class Frame>
inline Vec3<Frame, Scalar> lift_from_real(const Vec3<Frame, MFloat>& v) noexcept {
    return Vec3<Frame, Scalar>::from_raw(v.raw().template cast<Scalar>());
}

template <class Scalar, class Frame>
inline Ori<Frame, Scalar> lift_from_real(const Ori<Frame, MFloat>& q) noexcept {
    return Ori<Frame, Scalar>{Eigen::Quaternion<Scalar>(
        Scalar(q.raw().w()), Scalar(q.raw().x()),
        Scalar(q.raw().y()), Scalar(q.raw().z()))};
}

template <class NewScalar, class From, class To>
inline KinematicLink<From, To, NewScalar>
cast_kinematic_link(const KinematicLink<From, To, MFloat>& link) noexcept {
    return KinematicLink<From, To, NewScalar>{
        lift_from_real<NewScalar>(link.position()),
        lift_from_real<NewScalar>(link.orientation()),
        lift_from_real<NewScalar>(link.vel_linear()),
        lift_from_real<NewScalar>(link.vel_angular()),
        lift_from_real<NewScalar>(link.acc_linear()),
        lift_from_real<NewScalar>(link.acc_angular()),
    };
}

template <class NewScalar, class From, class To>
inline StaticLink<From, To, NewScalar>
cast_static_link(const StaticLink<From, To, MFloat>& link) noexcept {
    return StaticLink<From, To, NewScalar>{
        lift_from_real<NewScalar>(link.position()),
        lift_from_real<NewScalar>(link.orientation()),
    };
}

} // namespace manta::geom
