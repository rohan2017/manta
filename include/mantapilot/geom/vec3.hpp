#pragma once

#include <Eigen/Core>
#include <cassert>

#include "../core/frame.hpp"
#include "../core/types.hpp"

namespace manta::geom {

// Frame-tagged 3-vector. Same-frame arithmetic is checked at compile time;
// identity is checked at runtime in debug builds.
//
// Inside a part's update() users may freely operate on raw Eigen vectors via
// .raw() and rewrap with from_raw() at framework boundaries.
template <typename Frame, typename Scalar = mFloat>
class Vec3 {
public:
    using EigenT = Eigen::Matrix<Scalar, 3, 1>;

    constexpr Vec3() noexcept = default;

    constexpr Vec3(Scalar x, Scalar y, Scalar z, FrameId id = {}) noexcept
        : v_(x, y, z)
#if defined(MANTAPILOT_DEBUG_FRAMES)
        , id_(id)
#endif
    {
#if !defined(MANTAPILOT_DEBUG_FRAMES)
        (void)id;
#endif
    }

    static Vec3 from_raw(const EigenT& v, FrameId id = {}) noexcept {
        Vec3 out;
        out.v_ = v;
#if defined(MANTAPILOT_DEBUG_FRAMES)
        out.id_ = id;
#else
        (void)id;
#endif
        return out;
    }

    static Vec3 zero(FrameId id = {}) noexcept {
        return from_raw(EigenT::Zero(), id);
    }

    const EigenT& raw() const noexcept { return v_; }
    EigenT&       raw() noexcept       { return v_; }

    Scalar x() const noexcept { return v_.x(); }
    Scalar y() const noexcept { return v_.y(); }
    Scalar z() const noexcept { return v_.z(); }

    FrameId id() const noexcept {
#if defined(MANTAPILOT_DEBUG_FRAMES)
        return id_;
#else
        return {};
#endif
    }

    Vec3 operator+(const Vec3& o) const noexcept {
        check_id(o);
        return from_raw(v_ + o.v_, id());
    }
    Vec3 operator-(const Vec3& o) const noexcept {
        check_id(o);
        return from_raw(v_ - o.v_, id());
    }
    Vec3 operator-() const noexcept {
        return from_raw(-v_, id());
    }
    Vec3 operator*(Scalar s) const noexcept {
        return from_raw(v_ * s, id());
    }
    Vec3 operator/(Scalar s) const noexcept {
        return from_raw(v_ / s, id());
    }

    Vec3& operator+=(const Vec3& o) noexcept { check_id(o); v_ += o.v_; return *this; }
    Vec3& operator-=(const Vec3& o) noexcept { check_id(o); v_ -= o.v_; return *this; }

    Scalar dot(const Vec3& o) const noexcept {
        check_id(o);
        return v_.dot(o.v_);
    }
    Vec3 cross(const Vec3& o) const noexcept {
        check_id(o);
        return from_raw(v_.cross(o.v_), id());
    }
    Scalar norm() const noexcept         { return v_.norm(); }
    Scalar squared_norm() const noexcept { return v_.squaredNorm(); }
    Vec3   normalized() const noexcept   { return from_raw(v_.normalized(), id()); }

private:
    EigenT v_ = EigenT::Zero();

#if defined(MANTAPILOT_DEBUG_FRAMES)
    FrameId id_ = {};
#endif

    void check_id(const Vec3& o) const noexcept {
#if defined(MANTAPILOT_DEBUG_FRAMES)
        // Identity 0 is a wildcard (e.g. constants like zero()), passes.
        assert((id_.value() == 0 || o.id_.value() == 0 || id_ == o.id_)
               && "Vec3 frame identity mismatch");
#else
        (void)o;
#endif
    }
};

template <typename Frame, typename Scalar>
inline Vec3<Frame, Scalar> operator*(Scalar s, const Vec3<Frame, Scalar>& v) noexcept {
    return v * s;
}

} // namespace manta::geom
