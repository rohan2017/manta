#pragma once

#include <Eigen/Geometry>

#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "vec3.hpp"

namespace manta::geom {

// Frame-tagged orientation. Conventionally represents the attitude of an
// implicit child frame relative to Frame F, with quaternion components stored
// in F's basis. Ori is data — it does not rotate vectors by itself. To
// transform between frames, use a StaticLink or KinematicLink.
template <typename Frame, typename Scalar = mFloat>
class Ori {
public:
    using QuatT = Eigen::Quaternion<Scalar>;

    constexpr Ori() noexcept = default;
    explicit Ori(const QuatT& q, FrameId id = {}) noexcept
        : q_(q)
#if defined(MANTAPILOT_DEBUG_FRAMES)
        , id_(id)
#endif
    {
#if !defined(MANTAPILOT_DEBUG_FRAMES)
        (void)id;
#endif
    }

    static Ori identity(FrameId id = {}) noexcept {
        return Ori{QuatT::Identity(), id};
    }

    static Ori from_axis_angle(const Vec3<Frame, Scalar>& axis, Scalar angle,
                               FrameId id = {}) noexcept {
        return Ori{QuatT{Eigen::AngleAxis<Scalar>{angle, axis.raw().normalized()}}, id};
    }

    const QuatT& raw() const noexcept { return q_; }
    QuatT&       raw() noexcept       { return q_; }

    FrameId id() const noexcept {
#if defined(MANTAPILOT_DEBUG_FRAMES)
        return id_;
#else
        return {};
#endif
    }

    Ori inverse() const noexcept { return Ori{q_.conjugate(), id()}; }

    void normalize() noexcept { q_.normalize(); }
    Ori  normalized() const noexcept { return Ori{q_.normalized(), id()}; }

    // Composition of two orientations in the same frame.
    Ori operator*(const Ori& o) const noexcept {
        return Ori{q_ * o.q_, id()};
    }

private:
    QuatT q_ = QuatT::Identity();

#if defined(MANTAPILOT_DEBUG_FRAMES)
    FrameId id_ = {};
#endif
};

} // namespace manta::geom
