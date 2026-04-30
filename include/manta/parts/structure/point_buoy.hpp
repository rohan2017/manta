#pragma once

#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../fields/gravity_field.hpp"

namespace manta::parts {

// Single-point buoyancy. F = -ρ(p_scene) * V * g_part at the part origin.
// FluidField and GravityField are both Real-typed; we cast at the boundary
// so this part works in templated estimator crafts (treating the field
// values as constant inputs for autodiff purposes).
//
// Required fields: FluidField, GravityField.
template <class Scalar = Real>
class PointBuoyT : public PartT<Scalar> {
public:
    PointBuoyT(std::string name, Scalar volume)
        : PartT<Scalar>(std::move(name)), volume_(volume) {}

    Scalar volume() const noexcept { return volume_; }
    void   set_volume(Scalar v) noexcept { volume_ = v; }

    void update() override {
        if (volume_ <= Scalar(0)) return;
        auto& fluid = this->template field<fields::FluidField>();
        auto& gf    = this->template field<fields::GravityField>();

        // Bridge templated position → Real for the field query.
        auto p_scaled = this->template position<SceneFrame>();
        Eigen::Matrix<Real, 3, 1> p_real;
        if constexpr (std::is_floating_point_v<Scalar>) {
            p_real = p_scaled.raw().template cast<Real>();
        } else {
            for (int i = 0; i < 3; ++i) p_real(i) = Real(p_scaled.raw()(i).a);
        }
        auto state = fluid.state_at(geom::Vec3<SceneFrame>::from_raw(p_real));
        Scalar rho = Scalar(state.density);

        Eigen::Matrix<Scalar, 3, 1> g_scene = gf.g().raw().template cast<Scalar>();
        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();
        auto g_part = geom::Vec3<PartFrame, Scalar>::from_raw(
            q_part_from_scene * g_scene);

        this->apply_force_at(g_part * (-rho * volume_));
    }

private:
    Scalar volume_;
};

using PointBuoy = PointBuoyT<Real>;

} // namespace manta::parts
