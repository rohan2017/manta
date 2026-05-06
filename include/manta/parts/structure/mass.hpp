#pragma once

#include <type_traits>

#include "../../core/craft.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/templated_query.hpp"

namespace manta::parts {

// A lump of mass with full 3×3 MOI tensor. Templated on Scalar.
//
// If a `fields::GravityField` is registered on the world (or craft), the
// part queries it at its own CoM each tick and applies `m · g` to itself.
// If no GravityField is registered, the part is force-free — pure mass+MOI
// contribution to the craft's aggregated dynamics.
//
// Set `apply_gravity=false` at construction to opt out (e.g. for ballast
// inside a sealed compartment whose buoyancy is already modeled, or for
// estimator crafts that don't represent gravity).
template <class Scalar = Real>
class MassT : public PartT<Scalar> {
public:
    // Full constructor: explicit MOI tensor.
    MassT(std::string name, Scalar mass,
          const geom::Mat3<PartFrame, PartFrame, Scalar>& moi,
          bool apply_gravity = true)
        : PartT<Scalar>(std::move(name)), apply_gravity_(apply_gravity) {
        this->set_mass(mass);
        this->set_moi(moi);
    }

    // Point-mass shorthand: zero MOI tensor. Equivalent to the deleted
    // `PointMass`. Useful for ballast / lumped-CoM modeling where rotational
    // inertia is contributed by other parts (or none at all).
    MassT(std::string name, Scalar mass, bool apply_gravity = true)
        : PartT<Scalar>(std::move(name)), apply_gravity_(apply_gravity) {
        this->set_mass(mass);
        this->set_moi(geom::Mat3<PartFrame, PartFrame, Scalar>::zero());
    }

    bool apply_gravity() const noexcept       { return apply_gravity_; }
    void set_apply_gravity(bool b) noexcept   { apply_gravity_ = b; }

    // GravityField is OPTIONAL augmentation: the macro gates compilation,
    // and `field_or_null` gives a graceful no-op at runtime when the
    // craft's world hasn't registered one (or this craft is unattached).
    void update() override {
        if constexpr (MANTA_PART_AUGMENTS_FIELD(MANTA_HAS_GRAVITY_FIELD)) {
            if (!apply_gravity_) return;
            const auto* gf = this->template field_or_null<fields::GravityField>();
            if (!gf) return;

            // CoM in scene frame, then a Scalar-templated field query so
            // Jet crafts pick up ∂g/∂pos. Real crafts get a no-op fast path.
            auto com_part  = this->get_com();
            auto com_scene = this->scene_to_part().apply_position(com_part);
            auto g_scene_v = fields::state_at_templated<Scalar>(*gf, com_scene);

            auto q_part_from_scene =
                this->template orientation<SceneFrame>().raw().conjugate();
            auto g_part = geom::Vec3<PartFrame, Scalar>::from_raw(
                q_part_from_scene * g_scene_v.raw());

            this->apply_force_at(g_part * this->get_mass(), com_part);
        }
    }

private:
    bool apply_gravity_;
};

using Mass = MassT<Real>;

} // namespace manta::parts
