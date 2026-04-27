#pragma once

#include <Eigen/Core>

#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "vec3.hpp"

namespace manta::geom {

// Frame-tagged 3x3 matrix. Two-frame because matrices typically act as
// linear maps between frames (rotation, inertia mapping). When From == To,
// represents a within-frame tensor (inertia, covariance).
template <typename From, typename To = From, typename Scalar = mFloat>
class Mat3 {
public:
    using EigenT = Eigen::Matrix<Scalar, 3, 3>;

    constexpr Mat3() noexcept = default;
    explicit Mat3(const EigenT& m) noexcept : m_(m) {}

    static Mat3 zero() noexcept     { return Mat3{EigenT::Zero()}; }
    static Mat3 identity() noexcept { return Mat3{EigenT::Identity()}; }

    static Mat3 from_diagonal(Scalar a, Scalar b, Scalar c) noexcept {
        EigenT m = EigenT::Zero();
        m(0,0) = a; m(1,1) = b; m(2,2) = c;
        return Mat3{m};
    }

    const EigenT& raw() const noexcept { return m_; }
    EigenT&       raw() noexcept       { return m_; }

    // Mat3<A,B> * Vec3<B> -> Vec3<A>
    Vec3<From, Scalar> operator*(const Vec3<To, Scalar>& v) const noexcept {
        return Vec3<From, Scalar>::from_raw(m_ * v.raw());
    }

    // Mat3<A,B> * Mat3<B,C> -> Mat3<A,C>
    template <typename Other, typename Scalar2>
    Mat3<From, Other, Scalar> operator*(const Mat3<To, Other, Scalar2>& o) const noexcept {
        return Mat3<From, Other, Scalar>{m_ * o.raw()};
    }

    Mat3<To, From, Scalar> transpose() const noexcept {
        return Mat3<To, From, Scalar>{m_.transpose()};
    }

    Mat3 operator+(const Mat3& o) const noexcept { return Mat3{m_ + o.m_}; }
    Mat3 operator-(const Mat3& o) const noexcept { return Mat3{m_ - o.m_}; }
    Mat3 operator*(Scalar s) const noexcept      { return Mat3{m_ * s}; }

private:
    EigenT m_ = EigenT::Zero();
};

} // namespace manta::geom
