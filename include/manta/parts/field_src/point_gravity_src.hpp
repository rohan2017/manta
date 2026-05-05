#pragma once

#include <type_traits>

#include "../../core/craft.hpp"
#include "../../fields/gravity_field.hpp"

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
template <class Scalar = Real>
class PointGravitySrcT : public PartT<Scalar> {
public:
    static constexpr Real kBigG = Real(6.67430e-11f);   // m^3·kg^-1·s^-2

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
        auto* gf = this->field_ptr(typeid(fields::GravityField));
        if (!gf) return;
        auto& g = *static_cast<fields::GravityField*>(gf);

        // Position of the source in scene frame.
        auto p_scaled = this->template position<SceneFrame>();
        Eigen::Matrix<Real, 3, 1> p_real;
        if constexpr (std::is_floating_point_v<Scalar>) {
            p_real = p_scaled.raw().template cast<Real>();
        } else {
            for (int i = 0; i < 3; ++i) p_real(i) = Real(p_scaled.raw()(i).a);
        }
        auto origin = geom::Vec3<SceneFrame>::from_raw(p_real);

        Real mu;
        if constexpr (std::is_floating_point_v<Scalar>) mu = Real(kBigG * grav_mass_);
        else                                            mu = Real(kBigG * grav_mass_.a);

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

using PointGravitySrc = PointGravitySrcT<Real>;

} // namespace manta::parts
