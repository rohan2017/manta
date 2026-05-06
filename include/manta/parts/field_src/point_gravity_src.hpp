#pragma once

#include <type_traits>

#include "../../core/craft.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../geom/casts.hpp"

namespace manta::parts {

// PointGravitySrc — a part that *contributes* an inverse-square gravity
// disturbance to a registered GravityField, centered at the part's own
// scene-frame position. Useful for self-gravitating crafts (large asteroids,
// space stations) or to model a moving gravitating body without parking it
// in a Planet.
//
// The part has no mass or MOI of its own; only its `grav_mass` (in kg)
// determines the strength of the disturbance via μ = G · grav_mass.
//
// Each tick the previous tick's disturbance is removed and a fresh one is
// added (lifetime defaults to 1) so the source tracks the part's motion. If
// the part is fixed in scene frame, the lifetime can be configured to
// PERSISTENT to avoid the per-tick churn (set via `set_persistent(true)`).
//
// Required fields: GravityField.
template <class Scalar = MFloat>
class PointGravitySrcT : public PartT<Scalar> {
public:
    static constexpr MFloat kBigG = MFloat(6.67430e-11f);   // m^3·kg^-1·s^-2

    explicit PointGravitySrcT(std::string name, Scalar grav_mass)
        : PartT<Scalar>(std::move(name)), grav_mass_(grav_mass) {}

    Scalar grav_mass() const noexcept            { return grav_mass_; }
    void   set_grav_mass(Scalar m) noexcept      { grav_mass_ = m; }

    bool persistent() const noexcept             { return persistent_; }
    void set_persistent(bool b) noexcept         { persistent_ = b; }

    // PointGravitySrc REQUIRES a GravityField — its job is to add
    // disturbances to one. Both Python codegen and the static_assert
    // below catch a missing field.
    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_GRAVITY_FIELD,
        "PointGravitySrc requires a GravityField on the world. "
        "Register one with World.add_field(GravityField()), or "
        "remove this part from the craft.");

    void update() override {
        // PointGravitySrc REQUIRES a GravityField (build-time static_assert).
        // Use field_or_null at runtime so unattached test crafts and
        // multi-world setups omitting the field no-op gracefully.
        auto* gp = this->template field_or_null<fields::GravityField>();
        if (!gp) return;
        auto& g = *gp;

        // Position of the source in scene frame.
        auto p_scaled = this->template position<SceneFrame>();
        auto origin   = geom::cast_to_real(p_scaled);

        MFloat mu = kBigG * geom::strip_to_real(grav_mass_);

        // For non-persistent sources we let the previous tick's disturbance
        // expire on its own (lifetime=1), keeping update() symmetric across
        // ticks. For persistent sources the user is asserting the part won't
        // move materially, so we add once and never re-add.
        if (persistent_) {
            if (!added_persistent_) {
                handle_ = g.add(fields::GravityField::Disturbance::point_mass(origin, mu),
                                fields::PERSISTENT);
                added_persistent_ = true;
            }
        } else {
            g.add(fields::GravityField::Disturbance::point_mass(origin, mu),
                  /*lifetime=*/1);
        }
    }

private:
    Scalar grav_mass_;
    bool   persistent_      = false;
    bool   added_persistent_= false;
    fields::GravityField::Handle handle_{};
};

using PointGravitySrc = PointGravitySrcT<MFloat>;

} // namespace manta::parts
