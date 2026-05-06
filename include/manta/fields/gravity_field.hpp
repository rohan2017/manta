#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// Tag IDs identifying disturbance kinds for cross-process replication. Stock
// tags occupy [1, manta::fields::USER_BASE). User-defined factories use
// values >= USER_BASE.
namespace gravity_tags {
    constexpr std::uint16_t UNIFORM       = 1;
    constexpr std::uint16_t POINT_MASS    = 2;
    constexpr std::uint16_t POINT_MASS_J2 = 3;
}

// Gravity as a superposition of disturbances. Each Disturbance is a
// (tag, origin, lambda, params) tuple where the lambda returns the
// gravitational acceleration contribution at a given offset; the
// (tag, params) pair lets the disturbance be re-bound on a different
// process via the factory registry inherited from DisturbanceField.
class GravityField; // forward (Disturbance forward references the field)

struct GravityDisturbance {
    using Vec       = geom::Vec3<SceneFrame>;
    using DeltaG    = std::function<Vec(const Vec& offset)>;
    using Influence = std::function<bool(const Vec& offset)>;

    Vec           origin = Vec::zero();
    DeltaG        delta_g;
    Influence     in_influence;

    // Replication metadata. tag == USER_TAG means local-only.
    std::uint16_t tag    = USER_TAG;
    Params        params{};

    // Layout-stable POD param structs (the wire format). Keep these in
    // sync with the factory builders in `gravity_field.cpp` (or the
    // inline helpers below); the static_asserts pin the size budget.
    struct UniformParams     { Real gx, gy, gz; };
    struct PointMassParams   { Real ox, oy, oz, mu; };
    struct PointMassJ2Params { Real ox, oy, oz, mu, j2, eq_radius, ax, ay, az; };

    // Uniform gravity everywhere (e.g. flat-earth -9.81 ẑ).
    static GravityDisturbance uniform(Vec g) {
        GravityDisturbance d;
        d.tag    = gravity_tags::UNIFORM;
        UniformParams up{Real(g.x()), Real(g.y()), Real(g.z())};
        std::memcpy(d.params.data(), &up, sizeof(up));
        d.delta_g = [g](const Vec&) noexcept { return g; };
        return d;
    }

    // Inverse-square gravity from a point mass at `origin`. `mu` is GM.
    static GravityDisturbance point_mass(Vec origin, Real mu) {
        GravityDisturbance d;
        d.origin = origin;
        d.tag    = gravity_tags::POINT_MASS;
        PointMassParams pm{Real(origin.x()), Real(origin.y()), Real(origin.z()), mu};
        std::memcpy(d.params.data(), &pm, sizeof(pm));
        d.delta_g = [mu](const Vec& off) noexcept {
            Real r2 = off.raw().squaredNorm();
            if (r2 < Real(1e-12f)) return Vec::zero();
            Real r  = std::sqrt(r2);
            return Vec::from_raw(off.raw() * (-mu / (r2 * r)));
        };
        return d;
    }

    // Point mass with J2 oblateness perturbation.
    static GravityDisturbance point_mass_j2(Vec  origin,
                                            Real mu,
                                            Real j2_coeff,
                                            Real equatorial_radius,
                                            Vec  polar_axis = Vec{Real(0), Real(0), Real(1)}) {
        GravityDisturbance d;
        d.origin = origin;
        d.tag    = gravity_tags::POINT_MASS_J2;
        PointMassJ2Params pp{
            Real(origin.x()), Real(origin.y()), Real(origin.z()),
            mu, j2_coeff, equatorial_radius,
            Real(polar_axis.x()), Real(polar_axis.y()), Real(polar_axis.z()),
        };
        std::memcpy(d.params.data(), &pp, sizeof(pp));
        d.delta_g = [mu, j2_coeff, equatorial_radius, polar_axis](const Vec& off) noexcept {
            const auto& r = off.raw();
            Real r2 = r.squaredNorm();
            if (r2 < Real(1e-12f)) return Vec::zero();
            Real r_norm = std::sqrt(r2);
            Real inv_r3 = Real(1) / (r2 * r_norm);
            Eigen::Matrix<Real, 3, 1> g = r * (-mu * inv_r3);
            Real inv_r2 = Real(1) / r2;
            Real z_p    = r.dot(polar_axis.raw());
            Real z_p2   = z_p * z_p;
            Real factor = Real(1.5) * j2_coeff
                        * (equatorial_radius * equatorial_radius) * inv_r2;
            Real five_z2_over_r2 = Real(5) * z_p2 * inv_r2;
            Eigen::Matrix<Real, 3, 1> r_eq = r - polar_axis.raw() * z_p;
            g += r_eq * (-mu * inv_r3 * factor * (five_z2_over_r2 - Real(1)))
               + polar_axis.raw() * (-mu * inv_r3 * factor * z_p
                                     * (five_z2_over_r2 - Real(3)));
            return Vec::from_raw(g);
        };
        return d;
    }
};

class GravityField : public DisturbanceField<GravityField, GravityDisturbance> {
public:
    using Vec         = GravityDisturbance::Vec;
    using Disturbance = GravityDisturbance;

    static_assert(sizeof(Disturbance::UniformParams)     <= kParamsBytes);
    static_assert(sizeof(Disturbance::PointMassParams)   <= kParamsBytes);
    static_assert(sizeof(Disturbance::PointMassJ2Params) <= kParamsBytes);

    GravityField() = default;
    explicit GravityField(Vec uniform_g) {
        add(Disturbance::uniform(uniform_g), PERSISTENT);
    }

    Vec state_at(const Vec& pos) const noexcept {
        Eigen::Matrix<Real, 3, 1> sum = Eigen::Matrix<Real, 3, 1>::Zero();
        for (const auto& e : entries_) {
            Vec off = Vec::from_raw(pos.raw() - e.d.origin.raw());
            if (e.d.in_influence && !e.d.in_influence(off)) continue;
            if (!e.d.delta_g) continue;
            sum += e.d.delta_g(off).raw();
        }
        return Vec::from_raw(sum);
    }

    // Stock factory registry — used by DisturbanceField::factory_map() to
    // populate the tag→factory table on first use.
    static std::unordered_map<std::uint16_t, DisturbanceFactory> stock_factories() {
        std::unordered_map<std::uint16_t, DisturbanceFactory> m;
        m[gravity_tags::UNIFORM] = [](const Params& p) {
            Disturbance::UniformParams up;
            std::memcpy(&up, p.data(), sizeof(up));
            return Disturbance::uniform(Vec{up.gx, up.gy, up.gz});
        };
        m[gravity_tags::POINT_MASS] = [](const Params& p) {
            Disturbance::PointMassParams pm;
            std::memcpy(&pm, p.data(), sizeof(pm));
            return Disturbance::point_mass(Vec{pm.ox, pm.oy, pm.oz}, pm.mu);
        };
        m[gravity_tags::POINT_MASS_J2] = [](const Params& p) {
            Disturbance::PointMassJ2Params pp;
            std::memcpy(&pp, p.data(), sizeof(pp));
            return Disturbance::point_mass_j2(
                Vec{pp.ox, pp.oy, pp.oz},
                pp.mu, pp.j2, pp.eq_radius,
                Vec{pp.ax, pp.ay, pp.az});
        };
        return m;
    }
};

} // namespace manta::fields
