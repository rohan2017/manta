#pragma once

#include <cmath>
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta {
namespace parts {
template <class Scalar> class TetherEndpointT;
}
namespace coupling {

// A pairwise spring-damper tether between two TetherEndpoint parts.
// Endpoints may belong to the same craft or to different crafts within the
// same Scene (positions are compared in SceneFrame).
//
// Templated on Scalar so estimator crafts (Scalar = ceres::Jet) can compute
// tether forces with autodiff active. Both endpoints sharing a Tether must
// use the same Scalar — typically all crafts in a multi-craft estimator are
// templated on the same Scalar.
//
// Lifetime: TetherT and its two TetherEndpointT parts hold non-owning
// pointers at each other. Either side may be destroyed first; the dtors
// null the surviving side's pointer so a stale endpoint or a dead tether
// is observable as `endpoint_a()/b() == nullptr` or `tether_ == nullptr`
// rather than dangling.
//
// Force model:
//   d = |p2 - p1|
//   if d <= rest_length:  F = 0  (slack — ropes don't push)
//   else:
//     extension = d - rest_length
//     v_along   = (v_other - v_self) · rhat       // separation rate
//     |F|       = k * extension + c * v_along
//     F_on_self points from self → other (pulls toward other when extended)
template <class Scalar = Real>
class TetherT {
public:
    TetherT(Scalar rest_length, Scalar stiffness, Scalar damping = Scalar(0)) noexcept
        : rest_length_(rest_length), k_(stiffness), c_(damping) {}

    ~TetherT();
    TetherT(const TetherT&) = delete;
    TetherT& operator=(const TetherT&) = delete;

    // Compute the SceneFrame force ON the body at p_self, pulling toward
    // p_other when the tether is stretched. v_self / v_other are linear
    // velocities of the endpoints in SceneFrame.
    geom::Vec3<SceneFrame, Scalar> force_on_self(
        const geom::Vec3<SceneFrame, Scalar>& p_self,
        const geom::Vec3<SceneFrame, Scalar>& p_other,
        const geom::Vec3<SceneFrame, Scalar>& v_self,
        const geom::Vec3<SceneFrame, Scalar>& v_other) const noexcept {
        Eigen::Matrix<Scalar, 3, 1> r = p_other.raw() - p_self.raw();
        Scalar d2 = r.squaredNorm();
        if (d2 < Scalar(1e-12f)) return geom::Vec3<SceneFrame, Scalar>::zero();
        // ceres::sqrt is ADL-found when Scalar is a Jet.
        using std::sqrt;
        Scalar d = sqrt(d2);
        if (d <= rest_length_) return geom::Vec3<SceneFrame, Scalar>::zero();
        Eigen::Matrix<Scalar, 3, 1> rhat = r / d;
        Scalar extension = d - rest_length_;
        // Separation rate along rhat: positive when other moving away from self.
        Scalar v_along = (v_other.raw() - v_self.raw()).dot(rhat);
        Scalar F_mag = k_ * extension + c_ * v_along;
        return geom::Vec3<SceneFrame, Scalar>::from_raw(rhat * F_mag);
    }

    // Set after both endpoints are constructed. Endpoints register themselves.
    void set_endpoint_a(parts::TetherEndpointT<Scalar>* ep) noexcept { a_ = ep; }
    void set_endpoint_b(parts::TetherEndpointT<Scalar>* ep) noexcept { b_ = ep; }
    parts::TetherEndpointT<Scalar>* endpoint_a() const noexcept { return a_; }
    parts::TetherEndpointT<Scalar>* endpoint_b() const noexcept { return b_; }

    Scalar rest_length() const noexcept { return rest_length_; }
    Scalar stiffness()   const noexcept { return k_; }
    Scalar damping()     const noexcept { return c_; }

private:
    Scalar rest_length_;
    Scalar k_;
    Scalar c_;
    parts::TetherEndpointT<Scalar>* a_ = nullptr;
    parts::TetherEndpointT<Scalar>* b_ = nullptr;
};

using Tether = TetherT<Real>;

} // namespace coupling
} // namespace manta

// Cross-class dtor: needs the full TetherEndpointT definition to clear
// the survivor's tether pointer. Defined inline at the bottom of
// tether_endpoint.hpp.

