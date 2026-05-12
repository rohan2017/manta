#pragma once

// CollisionField — collision modeled as a superposition of geometric
// disturbances (sphere lists and infinite half-spaces). Unlike gravity/
// magnetic/fluid where a sample-at-a-point query (`state_at`) is enough,
// collision is BILATERAL: the force on body A depends on both A's
// geometry AND every other body B's geometry plus the relative
// velocity at the contact. So this field exposes a different
// aggregator, `net_wrench_on(self)`, that walks every OTHER disturbance,
// computes pairwise sphere-vs-sphere or sphere-vs-plane contact, and
// returns the integrated (force, torque) on `self`.
//
// Contact dynamics: a critically-damped PD per contact pair, plus
// Coulomb friction. Spring + damper are along the contact normal n̂:
//
//     F_n   = max(0,  k_eff·overlap  -  d_eff·(v_rel·n̂)) · n̂
//     F_t   = -clamp(d_eff·|v_t|, μ_eff·|F_n|) · v̂_t
//     wrench at part origin = (F_n + F_t,  r_contact × (F_n + F_t))
//
// where k_eff / d_eff are harmonic means of the two surfaces' k/d, and
// μ_eff = min of the two surfaces' μ. `overlap` is the penetration
// depth along the contact normal.
//
// Replication: only single-sphere and infinite-plane "stock" factories
// replicate over the disturbance-sync wire. Multi-shape colliders carry
// USER_TAG (local-only) — the geometry doesn't fit the 96-byte Params
// budget and isn't worth a variable-length encoding right now.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

namespace collision_tags {
    constexpr std::uint16_t SINGLE_SPHERE   = 1;
    constexpr std::uint16_t INFINITE_PLANE  = 2;
}

class CollisionField; // forward

struct CollisionDisturbance {
    using Vec = geom::Vec3<SceneFrame>;

    enum class ShapeKind : std::uint8_t { Sphere, Plane };

    // Geometric primitive in scene frame. `offset` is the absolute
    // scene-frame position of the primitive's reference point (sphere
    // center, or a point on the plane). `normal` is only meaningful for
    // Plane (unit-length, pointing away from the half-space the plane
    // "fills"). The Collider part rotates each tick's shapes into scene
    // frame before adding, so consumers can use them directly.
    struct Shape {
        ShapeKind kind   = ShapeKind::Sphere;
        Vec       offset = Vec::zero();    // scene-frame
        Vec       normal = Vec{MFloat(0), MFloat(0), MFloat(1)};  // plane only
        MFloat    radius = MFloat(0);      // sphere only
    };

    Vec               origin       = Vec::zero();       // scene-frame
    Vec               linear_vel   = Vec::zero();       // scene-frame
    Vec               angular_vel  = Vec::zero();       // scene-frame
    std::vector<Shape> shapes;

    // Material properties for this disturbance's surfaces. Combined via
    // harmonic mean (k, d) and min (μ) with the other surface in a
    // contact pair.
    MFloat k_normal    = MFloat(1.0e6);    // N/m
    MFloat d_normal    = MFloat(5.0e3);    // N·s/m
    MFloat mu_static   = MFloat(0.7);      // unitless (Coulomb)
    MFloat mu_kinetic  = MFloat(0.5);      // unitless

    // Identity for self-exclusion: a Collider part that re-adds its
    // disturbance each tick passes its `this` pointer here so
    // `net_wrench_on` skips self-vs-self.
    void* owner = nullptr;

    // Replication metadata. USER_TAG means local-only.
    std::uint16_t tag = USER_TAG;
    Params        params{};

    // Stock POD param layouts. Keep in sync with the factory helpers.
    struct SingleSphereParams {
        MFloat ox, oy, oz;
        MFloat radius;
        MFloat k, d, mu_s, mu_k;
    };
    struct InfinitePlaneParams {
        MFloat ox, oy, oz;
        MFloat nx, ny, nz;
        MFloat k, d, mu_s, mu_k;
    };

    // ---- Stock factories (replication-eligible). ----

    static CollisionDisturbance single_sphere(Vec center,
                                              MFloat radius,
                                              MFloat k = MFloat(1.0e6),
                                              MFloat d = MFloat(5.0e3),
                                              MFloat mu_s = MFloat(0.7),
                                              MFloat mu_k = MFloat(0.5)) {
        CollisionDisturbance c;
        c.origin = center;
        c.shapes = {Shape{ShapeKind::Sphere, center, Vec::zero(), radius}};
        c.k_normal = k; c.d_normal = d; c.mu_static = mu_s; c.mu_kinetic = mu_k;
        c.tag = collision_tags::SINGLE_SPHERE;
        SingleSphereParams sp{
            MFloat(center.x()), MFloat(center.y()), MFloat(center.z()),
            radius, k, d, mu_s, mu_k,
        };
        std::memcpy(c.params.data(), &sp, sizeof(sp));
        return c;
    }

    static CollisionDisturbance infinite_plane(Vec point_on_plane,
                                               Vec normal,
                                               MFloat k = MFloat(1.0e6),
                                               MFloat d = MFloat(5.0e3),
                                               MFloat mu_s = MFloat(0.7),
                                               MFloat mu_k = MFloat(0.5)) {
        // Normalize the normal — required for sphere-plane signed-distance.
        const auto& nr = normal.raw();
        MFloat nlen = std::sqrt(nr.squaredNorm());
        if (nlen <= MFloat(0)) nlen = MFloat(1);
        Vec n_unit = Vec::from_raw(nr / nlen);

        CollisionDisturbance c;
        c.origin = point_on_plane;
        c.shapes = {Shape{ShapeKind::Plane, point_on_plane, n_unit, MFloat(0)}};
        c.k_normal = k; c.d_normal = d; c.mu_static = mu_s; c.mu_kinetic = mu_k;
        c.tag = collision_tags::INFINITE_PLANE;
        InfinitePlaneParams ip{
            MFloat(point_on_plane.x()), MFloat(point_on_plane.y()), MFloat(point_on_plane.z()),
            MFloat(n_unit.x()), MFloat(n_unit.y()), MFloat(n_unit.z()),
            k, d, mu_s, mu_k,
        };
        std::memcpy(c.params.data(), &ip, sizeof(ip));
        return c;
    }
};

class CollisionField
    : public DisturbanceField<CollisionField, CollisionDisturbance> {
public:
    using Vec         = CollisionDisturbance::Vec;
    using Disturbance = CollisionDisturbance;
    using Shape       = CollisionDisturbance::Shape;

    static_assert(sizeof(Disturbance::SingleSphereParams)  <= kParamsBytes);
    static_assert(sizeof(Disturbance::InfinitePlaneParams) <= kParamsBytes);

    CollisionField() = default;

    struct Wrench {
        Vec force  = Vec::zero();
        Vec torque = Vec::zero();    // about the `self.origin`
    };

    // Sum (force, torque) on `self` from every other disturbance in the
    // field. `self.owner` is used to skip self-vs-self. Torque is taken
    // about `self.origin`.
    Wrench net_wrench_on(const Disturbance& self) const noexcept {
        Wrench total;
        for (const auto& e : entries_) {
            if (&e.d == &self)              continue;
            if (e.d.owner && e.d.owner == self.owner) continue;
            accumulate_pair(self, e.d, total);
        }
        return total;
    }

    // Lookup a disturbance by handle, for the in-place-mutation pattern
    // a Collider uses to refresh its pose/velocity each tick without
    // add/remove churn. Returns nullptr if the handle expired.
    Disturbance* find(Handle h) noexcept {
        if (!h) return nullptr;
        for (auto& e : entries_) if (e.id == h.id) return &e.d;
        return nullptr;
    }

    // Stock factory registry — populates the tag→factory table for
    // cross-process replication of single-sphere / infinite-plane
    // disturbances. Multi-shape colliders stay USER_TAG and local-only.
    static std::unordered_map<std::uint16_t, DisturbanceFactory> stock_factories() {
        std::unordered_map<std::uint16_t, DisturbanceFactory> m;
        m[collision_tags::SINGLE_SPHERE] = [](const Params& p) {
            Disturbance::SingleSphereParams sp;
            std::memcpy(&sp, p.data(), sizeof(sp));
            return Disturbance::single_sphere(
                Vec{sp.ox, sp.oy, sp.oz}, sp.radius,
                sp.k, sp.d, sp.mu_s, sp.mu_k);
        };
        m[collision_tags::INFINITE_PLANE] = [](const Params& p) {
            Disturbance::InfinitePlaneParams ip;
            std::memcpy(&ip, p.data(), sizeof(ip));
            return Disturbance::infinite_plane(
                Vec{ip.ox, ip.oy, ip.oz}, Vec{ip.nx, ip.ny, ip.nz},
                ip.k, ip.d, ip.mu_s, ip.mu_k);
        };
        return m;
    }

private:
    // ---- Pair-contact accumulator. ----
    //
    // Walks every shape on `a` against every shape on `b`, computes the
    // single-contact wrench at the deepest penetration point, and adds
    // to `out`. Sphere-vs-sphere and sphere-vs-plane are supported;
    // plane-vs-plane is a no-op.
    static void accumulate_pair(const Disturbance& a, const Disturbance& b,
                                Wrench& out) noexcept {
        const MFloat eps_v = MFloat(1e-6);   // velocity dead-zone for tangent direction
        const MFloat eps_d = MFloat(1e-12);  // distance dead-zone

        const MFloat k_eff  = harmonic_mean(a.k_normal, b.k_normal);
        const MFloat d_eff  = harmonic_mean(a.d_normal, b.d_normal);
        const MFloat mu_s   = std::min(a.mu_static,  b.mu_static);
        const MFloat mu_k   = std::min(a.mu_kinetic, b.mu_kinetic);

        for (const auto& sa : a.shapes) {
            for (const auto& sb : b.shapes) {
                Vec contact_pt;
                Vec normal_a_from_b;          // unit, pointing INTO a (repulsive on a)
                MFloat overlap = MFloat(0);

                if (sa.kind == Disturbance::ShapeKind::Sphere &&
                    sb.kind == Disturbance::ShapeKind::Sphere) {
                    if (!sphere_sphere(sa, sb, contact_pt, normal_a_from_b, overlap,
                                       eps_d)) continue;
                } else if (sa.kind == Disturbance::ShapeKind::Sphere &&
                           sb.kind == Disturbance::ShapeKind::Plane) {
                    if (!sphere_plane(sa, sb, contact_pt, normal_a_from_b, overlap,
                                      eps_d)) continue;
                } else if (sa.kind == Disturbance::ShapeKind::Plane &&
                           sb.kind == Disturbance::ShapeKind::Sphere) {
                    Vec n_b_from_a;
                    if (!sphere_plane(sb, sa, contact_pt, n_b_from_a, overlap, eps_d))
                        continue;
                    normal_a_from_b = Vec::from_raw(-n_b_from_a.raw());
                } else {
                    continue;   // plane-vs-plane: no contact
                }

                // Velocities at contact point (rigid body): v + ω × r.
                Vec v_at_contact_a = velocity_at(a, contact_pt);
                Vec v_at_contact_b = velocity_at(b, contact_pt);
                Vec v_rel = Vec::from_raw(v_at_contact_a.raw() - v_at_contact_b.raw());

                MFloat v_normal_into_a = v_rel.raw().dot(normal_a_from_b.raw());

                // Normal: spring repulsion + damping. Clamp to non-attractive.
                MFloat f_n_mag = k_eff * overlap - d_eff * v_normal_into_a;
                if (f_n_mag < MFloat(0)) continue;     // no contact, no force
                Vec f_normal = Vec::from_raw(normal_a_from_b.raw() * f_n_mag);

                // Tangent friction. Project v_rel onto the plane normal to n̂.
                Vec v_tangent = Vec::from_raw(
                    v_rel.raw() - normal_a_from_b.raw() * v_normal_into_a);
                MFloat v_t_mag = std::sqrt(v_tangent.raw().squaredNorm());

                Vec f_friction = Vec::zero();
                if (v_t_mag > eps_v) {
                    // Kinetic friction, clamped by μ·|F_n|. The friction
                    // opposes v_tangent (which is body-A relative to body-B).
                    MFloat mu_t = (v_t_mag > eps_v * MFloat(10)) ? mu_k : mu_s;
                    MFloat f_t_mag = std::min(d_eff * v_t_mag, mu_t * f_n_mag);
                    f_friction = Vec::from_raw(-v_tangent.raw() * (f_t_mag / v_t_mag));
                }

                Vec f_total = Vec::from_raw(f_normal.raw() + f_friction.raw());

                // Torque on `a` about a.origin: r × F where r = contact_pt - a.origin.
                Vec r_a = Vec::from_raw(contact_pt.raw() - a.origin.raw());
                Vec tau = Vec::from_raw(r_a.raw().cross(f_total.raw()));

                out.force  = Vec::from_raw(out.force.raw()  + f_total.raw());
                out.torque = Vec::from_raw(out.torque.raw() + tau.raw());
            }
        }
    }

    // Sphere-vs-sphere contact: both shapes are Spheres with absolute
    // (scene-frame) `offset` centers. Returns false if no penetration.
    static bool sphere_sphere(const Shape& sa, const Shape& sb,
                              Vec& contact_pt, Vec& normal_a_from_b,
                              MFloat& overlap, MFloat eps_d) noexcept {
        Vec d = Vec::from_raw(sa.offset.raw() - sb.offset.raw());
        MFloat dist2 = d.raw().squaredNorm();
        MFloat r_sum = sa.radius + sb.radius;
        if (dist2 >= r_sum * r_sum) return false;     // no penetration
        MFloat dist = std::sqrt(dist2);
        if (dist < eps_d) {
            // Coincident centers — pick an arbitrary normal.
            normal_a_from_b = Vec{MFloat(1), MFloat(0), MFloat(0)};
            overlap = r_sum;
        } else {
            normal_a_from_b = Vec::from_raw(d.raw() / dist);
            overlap = r_sum - dist;
        }
        // Contact point: midway between the two surface points along n̂.
        Vec pa_surface = Vec::from_raw(sa.offset.raw() - normal_a_from_b.raw() * sa.radius);
        Vec pb_surface = Vec::from_raw(sb.offset.raw() + normal_a_from_b.raw() * sb.radius);
        contact_pt = Vec::from_raw((pa_surface.raw() + pb_surface.raw()) * MFloat(0.5));
        return true;
    }

    // Sphere-vs-plane contact: `sa` is a sphere, `sb` is a plane.
    // Returns false if the sphere is on the +normal side and not touching.
    static bool sphere_plane(const Shape& sphere, const Shape& plane,
                             Vec& contact_pt, Vec& normal_a_from_b,
                             MFloat& overlap, MFloat /*eps_d*/) noexcept {
        // Signed distance from sphere center to plane along the plane normal.
        Vec to_center = Vec::from_raw(sphere.offset.raw() - plane.offset.raw());
        MFloat signed_d = to_center.raw().dot(plane.normal.raw());
        if (signed_d >= sphere.radius) return false;     // sphere clear of plane
        // Normal pointing INTO the sphere (from plane → sphere).
        normal_a_from_b = plane.normal;
        overlap = sphere.radius - signed_d;
        Vec foot = Vec::from_raw(sphere.offset.raw() - plane.normal.raw() * signed_d);
        contact_pt = foot;
        return true;
    }

    // Rigid-body velocity at a scene-frame point: v_origin + ω × (r - origin).
    static Vec velocity_at(const Disturbance& d, const Vec& point) noexcept {
        Vec r = Vec::from_raw(point.raw() - d.origin.raw());
        Eigen::Matrix<MFloat, 3, 1> cross = d.angular_vel.raw().cross(r.raw());
        return Vec::from_raw(d.linear_vel.raw() + cross);
    }

    static MFloat harmonic_mean(MFloat a, MFloat b) noexcept {
        if (a <= MFloat(0) || b <= MFloat(0)) return MFloat(0);
        return MFloat(2) * a * b / (a + b);
    }
};

} // namespace manta::fields
