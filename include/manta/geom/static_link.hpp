#pragma once

#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "ori.hpp"
#include "vec3.hpp"

namespace manta::geom {

// A fixed transform between two frames. Position is the To-origin expressed
// in From; orientation is the To-frame attitude with components in From.
//
// Use this for parent-to-child part transforms that never move (most static
// part attachments).
template <typename From, typename To, typename Scalar = MFloat>
class StaticLink {
public:
    using Pos = Vec3<From, Scalar>;
    using Att = Ori<From,  Scalar>;

    constexpr StaticLink() noexcept = default;

    StaticLink(Pos position, Att orientation) noexcept
        : position_(position), orientation_(orientation) {}

    static StaticLink identity() noexcept {
        return StaticLink{Pos::zero(), Att::identity()};
    }

    const Pos& position()    const noexcept { return position_; }
    const Att& orientation() const noexcept { return orientation_; }

    // Transform a position from To-frame to From-frame.
    Pos apply_position(const Vec3<To, Scalar>& p_to) const noexcept {
        return Pos::from_raw(orientation_.raw() * p_to.raw() + position_.raw());
    }

    // Pure rotation of a vector from To-frame to From-frame (no offset).
    Pos rotate(const Vec3<To, Scalar>& v_to) const noexcept {
        return Pos::from_raw(orientation_.raw() * v_to.raw());
    }
    Vec3<To, Scalar> rotate_inverse(const Pos& v_from) const noexcept {
        return Vec3<To, Scalar>::from_raw(orientation_.raw().conjugate() * v_from.raw());
    }

    // Inverse: To -> From becomes From -> To.
    StaticLink<To, From, Scalar> inverse() const noexcept {
        auto inv_q = orientation_.raw().conjugate();
        auto inv_p = -(inv_q * position_.raw());
        return StaticLink<To, From, Scalar>{
            Vec3<To, Scalar>::from_raw(inv_p),
            Ori<To,  Scalar>{inv_q}
        };
    }

    // Composition: StaticLink<A,B> * StaticLink<B,C> -> StaticLink<A,C>.
    template <typename C>
    StaticLink<From, C, Scalar> operator*(const StaticLink<To, C, Scalar>& other) const noexcept {
        auto new_q = orientation_.raw() * other.orientation().raw();
        auto new_p = orientation_.raw() * other.position().raw() + position_.raw();
        return StaticLink<From, C, Scalar>{
            Vec3<From, Scalar>::from_raw(new_p),
            Ori<From,  Scalar>{new_q}
        };
    }

private:
    Pos position_;
    Att orientation_ = Att::identity();
};

} // namespace manta::geom
