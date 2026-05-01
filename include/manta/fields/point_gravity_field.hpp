#pragma once

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// Inverse-square central gravity, optionally with the J2 oblateness
// perturbation (the second zonal harmonic of a body's gravity field —
// the dominant non-spherical term for Earth-like rotating bodies).
//
// `mu` is GM (gravitational parameter), m^3/s^2.
//   Earth   : 3.986004418e14
//   Moon    :  4.9048695e12
//   Mars    : 4.282837e13
//   Sun     : 1.32712440018e20
//
// J2 perturbation acts only when `j2_coeff > 0`. The polar axis is the
// direction of the body's spin (in scene frame); for an Earth-anchored
// scene this is the scene z axis, which is the default. For a non-rotating
// body or a non-anchored scene, leave J2 at 0 (point gravity only).
//
//   Earth values: j2_coeff = 1.0826267e-3, equatorial_radius = 6.378137e6 m
//
// The body itself does not rotate in this model — gravity is radial in the
// non-J2 case and axisymmetric about polar_axis in the J2 case, so body
// rotation does not affect the gravity vector. Track Earth attitude in the
// visualizer if you need a ground-fixed view.
class PointGravityField : public Field {
public:
    PointGravityField(Real mu,
                      geom::Vec3<SceneFrame> center = geom::Vec3<SceneFrame>::zero()) noexcept
        : mu_(mu), center_(center)
        , polar_axis_(geom::Vec3<SceneFrame>{Real(0), Real(0), Real(1)}) {}

    // J2 oblateness perturbation. Set both j2_coeff and equatorial_radius
    // to non-zero values to activate; otherwise the perturbation is zero
    // and `g_at` collapses to plain point gravity.
    void set_j2(Real j2_coeff, Real equatorial_radius) noexcept {
        j2_coeff_ = j2_coeff;
        eq_radius_ = equatorial_radius;
    }

    void set_polar_axis(const geom::Vec3<SceneFrame>& axis) noexcept {
        polar_axis_ = axis;
    }

    geom::Vec3<SceneFrame> g_at(const geom::Vec3<SceneFrame>& pos) const noexcept {
        auto r = pos.raw() - center_.raw();
        Real r2 = r.squaredNorm();
        if (r2 < Real(1e-12f)) return geom::Vec3<SceneFrame>::zero();
        Real r_norm = std::sqrt(r2);
        Real inv_r3 = Real(1) / (r2 * r_norm);

        // Point-mass term.
        Eigen::Matrix<Real, 3, 1> g_raw = r * (-mu_ * inv_r3);

        // J2 perturbation, if configured. Standard formula in body-frame
        // axes aligned with `polar_axis_`. For polar_axis = ẑ:
        //   a_x = -GM·x/r^3 · (3/2)·J2·(R_eq/r)^2 · (5·z²/r² − 1)
        //   a_y = (same)·y
        //   a_z = -GM·z/r^3 · (3/2)·J2·(R_eq/r)^2 · (5·z²/r² − 3)
        // For arbitrary polar axis ẑ_p we decompose r into axial and
        // equatorial parts: z_p = r · ẑ_p, then apply the same formula.
        if (j2_coeff_ != Real(0) && eq_radius_ != Real(0)) {
            Real inv_r2  = Real(1) / r2;
            Real z_p     = r.dot(polar_axis_.raw());
            Real z_p_sq  = z_p * z_p;
            Real factor  = Real(1.5) * j2_coeff_
                         * (eq_radius_ * eq_radius_) * inv_r2;
            Real five_z2_over_r2 = Real(5) * z_p_sq * inv_r2;
            // Equatorial contribution: along (r − z_p ẑ_p), with coefficient
            //   −mu·inv_r3 · factor · (5·z²/r² − 1)
            Eigen::Matrix<Real, 3, 1> r_eq = r - polar_axis_.raw() * z_p;
            Eigen::Matrix<Real, 3, 1> a_J2 =
                  r_eq * (-mu_ * inv_r3 * factor * (five_z2_over_r2 - Real(1)))
                + polar_axis_.raw() * (-mu_ * inv_r3 * factor * z_p
                                       * (five_z2_over_r2 - Real(3)));
            g_raw += a_J2;
        }

        return geom::Vec3<SceneFrame>::from_raw(g_raw);
    }

    Real                          mu()     const noexcept { return mu_; }
    Real                          j2()     const noexcept { return j2_coeff_; }
    Real                          eq_radius() const noexcept { return eq_radius_; }
    const geom::Vec3<SceneFrame>&   center() const noexcept { return center_; }
    const geom::Vec3<SceneFrame>&   polar_axis() const noexcept { return polar_axis_; }

    void set_mu(Real m) noexcept                       { mu_ = m; }
    void set_center(const geom::Vec3<SceneFrame>& c) noexcept { center_ = c; }

private:
    Real                 mu_;
    geom::Vec3<SceneFrame> center_;
    geom::Vec3<SceneFrame> polar_axis_;
    Real                 j2_coeff_  = Real(0);
    Real                 eq_radius_ = Real(0);
};

} // namespace manta::fields
