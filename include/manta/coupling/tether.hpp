#pragma once

#include <cmath>
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta {
namespace parts { class TetherEndpoint; }
namespace coupling {

// A pairwise spring-damper tether between two TetherEndpoint parts.
// Endpoints may belong to the same craft or to different crafts within the
// same Scene (positions are compared in SceneFrame).
//
// Force model:
//   d = |p2 - p1|
//   if d <= rest_length:  F = 0  (slack — ropes don't push)
//   else:
//     extension = d - rest_length
//     v_along   = (v_other - v_self) · rhat       // separation rate
//     |F|       = k * extension + c * v_along
//     F_on_self points from self → other (pulls toward other when extended)
//
// Lifetime: the Tether holds raw pointers to its endpoints (set when the
// endpoint Part is constructed). The user is responsible for keeping the
// Tether alive at least as long as either endpoint Part.
class Tether {
public:
    Tether(Real rest_length, Real stiffness, Real damping = Real(0)) noexcept
        : rest_length_(rest_length), k_(stiffness), c_(damping) {}

    // Compute the SceneFrame force ON the body at p_self, pulling toward
    // p_other when the tether is stretched. v_self / v_other are linear
    // velocities of the endpoints in SceneFrame.
    geom::Vec3<SceneFrame> force_on_self(const geom::Vec3<SceneFrame>& p_self,
                                         const geom::Vec3<SceneFrame>& p_other,
                                         const geom::Vec3<SceneFrame>& v_self,
                                         const geom::Vec3<SceneFrame>& v_other) const noexcept {
        auto r = p_other.raw() - p_self.raw();
        Real d2 = r.squaredNorm();
        if (d2 < Real(1e-12f)) return geom::Vec3<SceneFrame>::zero();
        Real d = std::sqrt(d2);
        if (d <= rest_length_) return geom::Vec3<SceneFrame>::zero();
        auto rhat = r / d;
        Real extension = d - rest_length_;
        // separation rate along rhat: positive when other moving away from self
        Real v_along = (v_other.raw() - v_self.raw()).dot(rhat);
        Real F_mag = k_ * extension + c_ * v_along;
        return geom::Vec3<SceneFrame>::from_raw(rhat * F_mag);
    }

    // Set after both endpoints are constructed. Endpoints register themselves.
    void set_endpoint_a(parts::TetherEndpoint* ep) noexcept { a_ = ep; }
    void set_endpoint_b(parts::TetherEndpoint* ep) noexcept { b_ = ep; }
    parts::TetherEndpoint* endpoint_a() const noexcept { return a_; }
    parts::TetherEndpoint* endpoint_b() const noexcept { return b_; }

    Real rest_length() const noexcept { return rest_length_; }
    Real stiffness()   const noexcept { return k_; }
    Real damping()     const noexcept { return c_; }

private:
    Real rest_length_;
    Real k_;
    Real c_;
    parts::TetherEndpoint* a_ = nullptr;
    parts::TetherEndpoint* b_ = nullptr;
};

} // namespace coupling
} // namespace manta
