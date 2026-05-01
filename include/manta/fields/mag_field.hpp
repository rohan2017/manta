#pragma once

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// Magnetic field — abstract base class so different planet/body magnetic
// models share a typeid for sensor parts (e.g. a future Magnetometer) to
// look up. Implementations override `b_at(pos)` returning the field vector
// in scene-frame Tesla.
class MagField : public Field {
public:
    virtual ~MagField() = default;

    // Magnetic flux density at a scene-frame point, in Tesla.
    virtual geom::Vec3<SceneFrame> b_at(const geom::Vec3<SceneFrame>& pos) const noexcept = 0;
};

// DipoleMagField — magnetic dipole approximation. Useful for Earth-like
// bodies where the leading IGRF term is a tilted dipole; for simplicity
// we align the dipole with the body's spin axis (no tilt) by default.
//
//   B(r) = (μ₀ / 4π) · (3 (m̂ · r̂) r̂ − m̂) · |m| / r³
//
// `moment` carries both magnitude and direction in scene frame:
//   Earth IGRF dipole magnitude ≈ 7.94e22 A·m²;
//   tilted ~11° from the spin axis. Default: aligned with -z (so a
//   compass on the surface of an Earth-anchored scene with z up points
//   toward magnetic north along the equator), magnitude 7.94e22.
class DipoleMagField : public MagField {
public:
    static constexpr Real kMu0Over4Pi = Real(1.0e-7f);    // T·m/A

    DipoleMagField(geom::Vec3<SceneFrame> moment =
                   geom::Vec3<SceneFrame>{Real(0), Real(0), Real(-7.94e22f)},
                   geom::Vec3<SceneFrame> center =
                   geom::Vec3<SceneFrame>::zero()) noexcept
        : moment_(moment), center_(center) {}

    geom::Vec3<SceneFrame> b_at(const geom::Vec3<SceneFrame>& pos) const noexcept override {
        Eigen::Matrix<Real, 3, 1> r = pos.raw() - center_.raw();
        Real r2 = r.squaredNorm();
        if (r2 < Real(1e-12f)) return geom::Vec3<SceneFrame>::zero();
        Real r_norm = std::sqrt(r2);
        Real inv_r3 = Real(1) / (r2 * r_norm);
        Real inv_r2 = Real(1) / r2;
        Real m_dot_r_over_r2 = moment_.raw().dot(r) * inv_r2;
        Eigen::Matrix<Real, 3, 1> b =
              kMu0Over4Pi * inv_r3
              * (Real(3) * m_dot_r_over_r2 * r - moment_.raw());
        return geom::Vec3<SceneFrame>::from_raw(b);
    }

    const geom::Vec3<SceneFrame>& moment() const noexcept { return moment_; }
    const geom::Vec3<SceneFrame>& center() const noexcept { return center_; }
    void set_moment(const geom::Vec3<SceneFrame>& m) noexcept { moment_ = m; }
    void set_center(const geom::Vec3<SceneFrame>& c) noexcept { center_ = c; }

private:
    geom::Vec3<SceneFrame> moment_;
    geom::Vec3<SceneFrame> center_;
};

} // namespace manta::fields
