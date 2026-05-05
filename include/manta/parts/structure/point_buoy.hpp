#pragma once

#include <type_traits>

#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/templated_query.hpp"

namespace manta::parts {

// Single-point buoyancy. F = -ρ(p_scene) · V · g_part at the part origin,
// where ρ is the local fluid density and g is the local gravitational
// acceleration. Both fields are queried via the Real-bridge (treating their
// outputs as constant inputs from the autodiff perspective).
//
// Required fields: FluidField, GravityField.
template <class Scalar = Real>
class PointBuoyT : public PartT<Scalar> {
public:
    PointBuoyT(std::string name, Scalar volume)
        : PartT<Scalar>(std::move(name)), volume_(volume) {}

    Scalar volume() const noexcept { return volume_; }
    void   set_volume(Scalar v) noexcept { volume_ = v; }

    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_FLUID_FIELD,
        "PointBuoy requires a FluidField on the world. Register one "
        "with World.add_field(FluidField(...)), or remove the buoy.");
    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_GRAVITY_FIELD,
        "PointBuoy requires a GravityField on the world (buoyancy "
        "force is ρV·g). Register one with "
        "World.add_field(GravityField(...)), or remove the buoy.");

    void update() override {
        if (volume_ <= Scalar(0)) return;
        auto& fluid = this->template field<fields::FluidField>();
        auto& gf    = this->template field<fields::GravityField>();

        // Density comes from the FluidField — currently treated as a constant
        // input from the Jet perspective (no autodiff through ρ). Switch to
        // templated FluidField queries when a use case demands ∂ρ/∂pos.
        auto p_scaled = this->template position<SceneFrame>();
        Eigen::Matrix<Real, 3, 1> p_real;
        if constexpr (std::is_floating_point_v<Scalar>) {
            p_real = p_scaled.raw().template cast<Real>();
        } else {
            for (int i = 0; i < 3; ++i) p_real(i) = Real(p_scaled.raw()(i).a);
        }
        auto state = fluid.state_at(geom::Vec3<SceneFrame>::from_raw(p_real));
        Scalar rho = Scalar(state.density);

        // Gravity uses the templated query so Jet crafts pick up ∂g/∂pos.
        auto g_scene_v = fields::state_at_templated<Scalar>(gf, p_scaled);

        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();
        auto g_part = geom::Vec3<PartFrame, Scalar>::from_raw(
            q_part_from_scene * g_scene_v.raw());

        this->apply_force_at(g_part * (-rho * volume_));
    }

private:
    Scalar volume_;
};

using PointBuoy = PointBuoyT<Real>;

} // namespace manta::parts
