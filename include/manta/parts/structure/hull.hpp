#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>
#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/sea_surface.hpp"

namespace manta::parts {

// Templated buoyant body — see header notes below for the algorithm.
//
// Templated on Scalar so the part can participate in templated estimator
// crafts. FluidField, SeaSurface, and GravityField are themselves not
// Scalar-templated; Hull casts the values queried from them to Scalar at
// the boundary, treating them as constant inputs for autodiff purposes.
template <class Scalar = Real>
class HullT : public PartT<Scalar> {
public:
    HullT(std::string name,
          Scalar total_volume,
          std::vector<geom::Vec3<PartFrame, Scalar>> sample_points,
          Scalar water_density = Scalar(1000.0f),
          Scalar air_density   = Scalar(1.225f))
        : PartT<Scalar>(std::move(name))
        , volume_(total_volume)
        , samples_(std::move(sample_points))
        , water_density_(water_density)
        , air_density_(air_density) {}

    void   set_surface_smoothing(Scalar h) noexcept { surface_smoothing_ = h; }
    Scalar surface_smoothing()       const noexcept { return surface_smoothing_; }

    void update() override {
        if (samples_.empty() || volume_ <= Scalar(0)) return;

        auto& fluid = this->template field<fields::FluidField>();
        auto& gf    = this->template field<fields::GravityField>();
        // SeaSurface is optional — field_ptr returns nullptr if unregistered.
        const auto* surface = dynamic_cast<const fields::SeaSurface*>(
            this->field_ptr(typeid(fields::SeaSurface)));

        const Scalar v_per = volume_ / Scalar(static_cast<float>(samples_.size()));
        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();
        Eigen::Matrix<Scalar, 3, 1> g_scene_e = gf.g().raw().template cast<Scalar>();
        auto g_part = geom::Vec3<PartFrame, Scalar>::from_raw(
            q_part_from_scene * g_scene_e);

        for (const auto& p_part : samples_) {
            // Sample's scene-frame position. Bridge to Real for FluidField /
            // SeaSurface queries (they're not Scalar-templated yet).
            auto p_scene_scaled = this->scene_to_part().apply_position(p_part);
            Eigen::Matrix<Real, 3, 1> p_scene_real;
            if constexpr (std::is_floating_point_v<Scalar>) {
                p_scene_real = p_scene_scaled.raw().template cast<Real>();
            } else {
                for (int i = 0; i < 3; ++i)
                    p_scene_real(i) = Real(p_scene_scaled.raw()(i).a);
            }
            auto p_scene_real_v = geom::Vec3<SceneFrame>::from_raw(p_scene_real);

            Scalar rho;
            if (surface) {
                Real depth_real = -surface->height_above_surface(p_scene_real_v);
                Scalar depth = Scalar(depth_real);
                Scalar h     = surface_smoothing_;
                Scalar t     = (depth + Scalar(0.5f) * h) / h;
                t = t < Scalar(0) ? Scalar(0) : (t > Scalar(1) ? Scalar(1) : t);
                Scalar s = t * t * (Scalar(3) - Scalar(2) * t);
                rho = air_density_ + s * (water_density_ - air_density_);
            } else {
                rho = Scalar(fluid.state_at(p_scene_real_v).density);
            }

            this->apply_force_at(g_part * (-rho * v_per), p_part);
        }
    }

    Scalar volume() const noexcept { return volume_; }
    const std::vector<geom::Vec3<PartFrame, Scalar>>& sample_points() const noexcept { return samples_; }
    Scalar water_density() const noexcept { return water_density_; }
    Scalar air_density()   const noexcept { return air_density_; }

private:
    Scalar                                       volume_;
    std::vector<geom::Vec3<PartFrame, Scalar>>   samples_;
    Scalar                                       water_density_;
    Scalar                                       air_density_;
    Scalar                                       surface_smoothing_ = Scalar(0.05f);
};

using Hull = HullT<Real>;

} // namespace manta::parts
