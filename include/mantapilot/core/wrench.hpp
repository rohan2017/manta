#pragma once

#include "../geom/vec3.hpp"
#include "frame.hpp"
#include "types.hpp"

namespace manta {

// A wrench is a (force, torque) pair expressed in a single frame. The torque
// is taken about the frame's origin. Wrenches in the same frame add.
template <typename Frame, typename Scalar = mFloat>
class Wrench {
public:
    using Vec = geom::Vec3<Frame, Scalar>;

    constexpr Wrench() noexcept = default;

    Wrench(Vec force, Vec torque) noexcept
        : force_(force), torque_(torque) {}

    static Wrench zero(FrameId id = {}) noexcept {
        return Wrench{Vec::zero(id), Vec::zero(id)};
    }

    // Construct a wrench from a force applied at a point (both in the same
    // frame). The equivalent torque about the frame origin is point x force.
    static Wrench from_force_at(const Vec& force, const Vec& point) noexcept {
        return Wrench{force, point.cross(force)};
    }

    static Wrench from_torque(const Vec& torque) noexcept {
        return Wrench{Vec::zero(torque.id()), torque};
    }

    const Vec& force()  const noexcept { return force_; }
    const Vec& torque() const noexcept { return torque_; }

    Wrench operator+(const Wrench& o) const noexcept {
        return Wrench{force_ + o.force_, torque_ + o.torque_};
    }
    Wrench& operator+=(const Wrench& o) noexcept {
        force_  += o.force_;
        torque_ += o.torque_;
        return *this;
    }
    Wrench operator-() const noexcept {
        return Wrench{-force_, -torque_};
    }

private:
    Vec force_;
    Vec torque_;
};

} // namespace manta
