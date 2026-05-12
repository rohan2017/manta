#pragma once

// Collider — a part that contributes a CollisionDisturbance to a
// registered CollisionField and applies the reaction wrench back onto
// itself each tick.
//
// On attach the Collider adds a PERSISTENT disturbance to the field
// (multi-shape, USER_TAG — local-only, not replicated). Each tick its
// `update()` pose/velocity-refreshes the disturbance in place, queries
// `CollisionField::net_wrench_on(self)`, and applies the (force, torque)
// result via `apply_force_at` / `apply_torque`.
//
// For the Jet shadow used by the EKF (Scalar != MFloat), update() is a
// no-op: collision is treated as a non-tracked input and contributes
// zero to the predict Jacobian. Re-add later if contact dynamics need
// to participate in autodiff.
//
// Required field: CollisionField.

#include <type_traits>
#include <utility>
#include <vector>

#include "../../core/craft.hpp"
#include "../../core/features.hpp"
#include "../../core/types.hpp"
#include "../../fields/collision_field.hpp"
#include "../../geom/casts.hpp"
#include "../../geom/vec3.hpp"

namespace manta::parts {

template <class Scalar = MFloat>
class ColliderT : public PartT<Scalar> {
public:
    using Vec      = geom::Vec3<PartFrame, Scalar>;
    using SceneVec = fields::CollisionDisturbance::Vec;

    // A single shape in PART frame. Used by the constructor to describe
    // the collider's local geometry; each tick `update()` rotates these
    // into scene frame before refreshing the disturbance.
    struct ShapeDesc {
        fields::CollisionDisturbance::ShapeKind kind =
            fields::CollisionDisturbance::ShapeKind::Sphere;
        // Sphere: `offset` is the center in part frame; `radius` is the
        // sphere radius. Plane: `offset` is a point on the plane in
        // part frame; `normal` is the plane normal in part frame.
        geom::Vec3<PartFrame> offset = geom::Vec3<PartFrame>::zero();
        geom::Vec3<PartFrame> normal = geom::Vec3<PartFrame>{
            MFloat(0), MFloat(0), MFloat(1)};
        MFloat radius = MFloat(0);
    };

    // Single-sphere constructor — the common case for a small craft.
    ColliderT(std::string name, MFloat radius,
              MFloat k_normal   = MFloat(1.0e6),
              MFloat d_normal   = MFloat(5.0e3),
              MFloat mu_static  = MFloat(0.7),
              MFloat mu_kinetic = MFloat(0.5))
        : PartT<Scalar>(std::move(name)),
          shapes_{{ShapeDesc{
              fields::CollisionDisturbance::ShapeKind::Sphere,
              geom::Vec3<PartFrame>::zero(),
              geom::Vec3<PartFrame>{MFloat(0), MFloat(0), MFloat(1)},
              radius,
          }}},
          k_normal_(k_normal),
          d_normal_(d_normal),
          mu_static_(mu_static),
          mu_kinetic_(mu_kinetic) {}

    // General constructor — a list of part-frame shapes for non-trivial
    // collider hulls (e.g. four spheres at a quadcopter's rotor tips).
    ColliderT(std::string name,
              std::vector<ShapeDesc> shapes,
              MFloat k_normal   = MFloat(1.0e6),
              MFloat d_normal   = MFloat(5.0e3),
              MFloat mu_static  = MFloat(0.7),
              MFloat mu_kinetic = MFloat(0.5))
        : PartT<Scalar>(std::move(name)),
          shapes_(std::move(shapes)),
          k_normal_(k_normal),
          d_normal_(d_normal),
          mu_static_(mu_static),
          mu_kinetic_(mu_kinetic) {}

    MFloat k_normal()   const noexcept { return k_normal_; }
    MFloat d_normal()   const noexcept { return d_normal_; }
    MFloat mu_static()  const noexcept { return mu_static_; }
    MFloat mu_kinetic() const noexcept { return mu_kinetic_; }

    const std::vector<ShapeDesc>& shapes() const noexcept { return shapes_; }

    // Reaction wrench from the most recent update(), in scene frame.
    // Useful for tests and debug logging.
    SceneVec last_force()  const noexcept { return last_force_; }
    SceneVec last_torque() const noexcept { return last_torque_; }

    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_COLLISION_FIELD,
        "Collider requires a CollisionField on the world. "
        "Register one with World.add_field(CollisionField()), or "
        "remove this part from the craft.");

    void update() override {
        // Jet path: no-op. The EKF doesn't model contact in predict.
        if constexpr (!std::is_same_v<Scalar, MFloat>) {
            return;
        } else {
            auto* fp = this->template field_or_null<fields::CollisionField>();
            if (!fp) return;
            auto& field = *fp;

            // Lazily add the disturbance the first time we run, then
            // mutate it in place every subsequent tick.
            if (!handle_) {
                fields::CollisionDisturbance d;
                d.k_normal   = k_normal_;
                d.d_normal   = d_normal_;
                d.mu_static  = mu_static_;
                d.mu_kinetic = mu_kinetic_;
                d.owner      = this;
                // Initial pose snapshot — replaced immediately below.
                refresh_disturbance(d);
                handle_ = field.add(std::move(d), fields::PERSISTENT);
            }
            auto* d = field.find(handle_);
            if (!d) return;                  // handle expired unexpectedly
            refresh_disturbance(*d);

            // Query net wrench on this disturbance from every other one.
            auto w = field.net_wrench_on(*d);
            last_force_  = w.force;
            last_torque_ = w.torque;

            // Apply on the part. Scene-frame force → part-frame via the
            // part's orientation; torque transforms the same way. Ori
            // doesn't expose a Vec3 operator; do it through the raw
            // Eigen quaternion (q^-1 · v rotates scene → part).
            auto q_scene_from_part_raw =
                this->template orientation<SceneFrame>().raw();
            auto q_part_from_scene_raw = q_scene_from_part_raw.conjugate();

            auto f_scene_raw = w.force.raw().template cast<Scalar>();
            auto t_scene_raw = w.torque.raw().template cast<Scalar>();
            auto f_part_raw  = q_part_from_scene_raw * f_scene_raw;
            auto t_part_raw  = q_part_from_scene_raw * t_scene_raw;

            // The contact-induced torque is already baked into w.torque
            // about a.origin (= part origin in scene), so apply at the
            // part origin without re-adding the moment arm.
            this->apply_force_at(geom::Vec3<PartFrame, Scalar>::from_raw(f_part_raw));
            this->apply_torque(geom::Vec3<PartFrame, Scalar>::from_raw(t_part_raw));
        }
    }

private:
    // Pull the current scene-frame pose + velocity from the part's
    // kinematic cache and rotate each part-frame shape into scene frame.
    void refresh_disturbance(fields::CollisionDisturbance& d) noexcept {
        auto p_scene_typed = this->template position<SceneFrame>();
        auto v_scene_typed = this->template velocity<SceneFrame>();
        auto w_scene_typed = this->template angular_velocity<SceneFrame>();
        auto q_scene_from_part_raw =
            this->template orientation<SceneFrame>().raw();

        d.origin       = geom::cast_to_real(p_scene_typed);
        d.linear_vel   = geom::cast_to_real(v_scene_typed);
        d.angular_vel  = geom::cast_to_real(w_scene_typed);

        d.shapes.clear();
        d.shapes.reserve(shapes_.size());
        for (const auto& s : shapes_) {
            // Rotate part-frame offset/normal into scene frame via the
            // raw Eigen quaternion, then translate by the part's
            // scene-frame origin.
            auto off_part_raw = s.offset.raw().template cast<Scalar>();
            auto n_part_raw   = s.normal.raw().template cast<Scalar>();
            auto off_scene_raw = q_scene_from_part_raw * off_part_raw;
            auto n_scene_raw   = q_scene_from_part_raw * n_part_raw;

            SceneVec off_real = SceneVec::from_raw(
                off_scene_raw.template cast<MFloat>());
            SceneVec n_real = SceneVec::from_raw(
                n_scene_raw.template cast<MFloat>());
            SceneVec abs_off = SceneVec::from_raw(d.origin.raw() + off_real.raw());

            d.shapes.push_back(fields::CollisionDisturbance::Shape{
                s.kind, abs_off, n_real, s.radius,
            });
        }
    }

    std::vector<ShapeDesc> shapes_;
    MFloat                 k_normal_;
    MFloat                 d_normal_;
    MFloat                 mu_static_;
    MFloat                 mu_kinetic_;
    fields::CollisionField::Handle handle_{};
    SceneVec               last_force_{};
    SceneVec               last_torque_{};
};

using Collider = ColliderT<MFloat>;

} // namespace manta::parts
