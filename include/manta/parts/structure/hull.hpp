#pragma once

#include <algorithm>
#include <vector>
#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/ocean_atmos_field.hpp"

namespace manta::parts {

// A buoyant body modeled as a set of equal-volume sample points. Each tick,
// for every sample point i:
//   1. Transform p_i (PartFrame) to scene frame.
//   2. Query the fluid density at that scene-frame position.
//   3. Compute buoyant contribution: F_i = -ρ_i * (V/N) * g_part
//      (Archimedes: weight of displaced fluid, opposing gravity)
//   4. Apply F_i at p_i (in part frame) — the offset gives the torque needed
//      for roll/pitch righting under partial submersion.
//
// REQUIRED fields (registered on Craft or World):
//   * fields::FluidField    — provides density via state_at(position)
//   * fields::GravityField  — provides g vector
//
// AUGMENTATION (auto-detected at runtime via dynamic_cast on the registered
// FluidField instance):
//   * If the registered FluidField is actually an OceanAtmosField, each
//     sample's effective density is smoothstep-blended between water and air
//     across `surface_smoothing` (default 5 cm) using the field's signed
//     `height_above_sea_level`. Removes the discrete-step force as the body
//     crosses the surface.
//   * Otherwise the raw density from FluidField::state_at is used (hard step
//     for any field that has a discontinuous density profile).
class Hull : public Part {
public:
    Hull(std::string name,
         Real total_volume,
         std::vector<geom::Vec3<PartFrame>> sample_points)
        : Part(std::move(name))
        , volume_(total_volume)
        , samples_(std::move(sample_points)) {}

    void set_surface_smoothing(Real h) noexcept { surface_smoothing_ = h; }
    Real surface_smoothing() const noexcept { return surface_smoothing_; }

    void update() override {
        if (samples_.empty() || volume_ <= Real(0)) return;

        auto& fluid = field<fields::FluidField>();
        auto& gf    = field<fields::GravityField>();
        const auto* ocean = dynamic_cast<const fields::OceanAtmosField*>(&fluid);

        const Real v_per = volume_ / Real(static_cast<float>(samples_.size()));
        auto q_part_from_scene = orientation<SceneFrame>().raw().conjugate();
        auto g_part = geom::Vec3<PartFrame>::from_raw(
            q_part_from_scene * gf.g().raw());

        for (const auto& p_part : samples_) {
            auto p_scene = scene_to_part().apply_position(p_part);

            Real rho;
            if (ocean) {
                // Smoothstep blend across the air-water interface.
                Real depth = -ocean->height_above_sea_level(p_scene);  // +ve below surface
                Real h     = surface_smoothing_;
                Real t     = (depth + Real(0.5f) * h) / h;
                t = std::clamp(t, Real(0), Real(1));
                Real s = t * t * (Real(3) - Real(2) * t);
                rho = ocean->air_density() + s * (ocean->water_density() - ocean->air_density());
            } else {
                rho = fluid.state_at(p_scene).density;
            }

            apply_force_at(g_part * (-rho * v_per), p_part);
        }
    }

    Real                                    volume()        const noexcept { return volume_; }
    const std::vector<geom::Vec3<PartFrame>>& sample_points() const noexcept { return samples_; }

private:
    Real                              volume_;
    std::vector<geom::Vec3<PartFrame>>  samples_;
    Real                              surface_smoothing_ = Real(0.05f);
};

} // namespace manta::parts
