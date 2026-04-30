#pragma once

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// Inverse-square central gravity. g(p) = -mu * (p - center) / |p - center|^3.
//
// `mu` is GM (gravitational parameter), m^3/s^2.
//   Earth   : 3.986004418e14
//   Moon    :  4.9048695e12
//   Mars    : 4.282837e13
//   Sun     : 1.32712440018e20
//
// The body itself does not rotate in this model — gravity is radial, so body
// rotation does not affect the gravity vector. Track Earth attitude in the
// visualizer if you need a ground-fixed view.
class PointGravityField : public Field {
public:
    PointGravityField(Real mu,
                      geom::Vec3<SceneFrame> center = geom::Vec3<SceneFrame>::zero()) noexcept
        : mu_(mu), center_(center) {}

    geom::Vec3<SceneFrame> g_at(const geom::Vec3<SceneFrame>& pos) const noexcept {
        auto r = pos.raw() - center_.raw();
        Real r2 = r.squaredNorm();
        if (r2 < Real(1e-12f)) return geom::Vec3<SceneFrame>::zero();
        Real r_norm = std::sqrt(r2);
        // -mu / |r|^3 * r
        auto g_raw = r * (-mu_ / (r2 * r_norm));
        return geom::Vec3<SceneFrame>::from_raw(g_raw);
    }

    Real                          mu()     const noexcept { return mu_; }
    const geom::Vec3<SceneFrame>&   center() const noexcept { return center_; }

    void set_mu(Real m) noexcept                       { mu_ = m; }
    void set_center(const geom::Vec3<SceneFrame>& c) noexcept { center_ = c; }

private:
    Real                 mu_;
    geom::Vec3<SceneFrame> center_;
};

} // namespace manta::fields
