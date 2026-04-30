#pragma once

#include <algorithm>
#include <vector>
#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/sea_surface.hpp"

namespace manta::parts {

// A buoyant body modeled as a set of equal-volume sample points. Each tick,
// for every sample point i:
//   1. Transform p_i (PartFrame) to scene frame.
//   2. Query fluid density at that scene-frame position.
//   3. Compute buoyant contribution: F_i = -ρ_i * (V/N) * g_part
//      (Archimedes: weight of displaced fluid, opposing gravity)
//   4. Apply F_i at p_i (in part frame) — the offset gives the torque needed
//      for roll/pitch righting under partial submersion.
//
// REQUIRED fields:
//   * fields::FluidField    — provides density via state_at(position)
//   * fields::GravityField  — provides g vector
//
// OPTIONAL surface-blend augmentation:
//   If a `fields::SeaSurface` is also registered, each sample's effective
//   density is smoothstep-blended between water and air across
//   `surface_smoothing` (default 5 cm), using the surface's signed
//   `height_above_surface(pos)`. Removes the discrete-step force as the
//   body crosses the surface. When no SeaSurface is registered, the raw
//   density from `FluidField::state_at` is used (hard step at whatever
//   boundary the fluid field has, if any).
//
// Querying the optional field via Craft::field_ptr (returns nullptr when
// unregistered) keeps Hull free of any specific surface implementation.
class Hull : public Part {
public:
    Hull(std::string name,
         Real total_volume,
         std::vector<geom::Vec3<PartFrame>> sample_points,
         Real water_density = Real(1000.0f),
         Real air_density   = Real(1.225f))
        : Part(std::move(name))
        , volume_(total_volume)
        , samples_(std::move(sample_points))
        , water_density_(water_density)
        , air_density_(air_density) {}

    void set_surface_smoothing(Real h) noexcept { surface_smoothing_ = h; }
    Real surface_smoothing() const noexcept { return surface_smoothing_; }

    void update() override {
        if (samples_.empty() || volume_ <= Real(0)) return;

        auto& fluid = field<fields::FluidField>();
        auto& gf    = field<fields::GravityField>();
        // SeaSurface is optional. field_ptr returns nullptr if unregistered.
        const auto* surface = dynamic_cast<const fields::SeaSurface*>(
            field_ptr(typeid(fields::SeaSurface)));

        const Real v_per = volume_ / Real(static_cast<float>(samples_.size()));
        auto q_part_from_scene = orientation<SceneFrame>().raw().conjugate();
        auto g_part = geom::Vec3<PartFrame>::from_raw(
            q_part_from_scene * gf.g().raw());

        for (const auto& p_part : samples_) {
            auto p_scene = scene_to_part().apply_position(p_part);

            Real rho;
            if (surface) {
                // Smoothstep blend across the air-water interface using the
                // SeaSurface's signed height. Hull stores the water+air
                // density values directly (matches Earth's typical pair);
                // the FluidField is queried only as a fallback.
                Real depth = -surface->height_above_surface(p_scene);  // +ve below
                Real h     = surface_smoothing_;
                Real t     = (depth + Real(0.5f) * h) / h;
                t = std::clamp(t, Real(0), Real(1));
                Real s = t * t * (Real(3) - Real(2) * t);
                rho = air_density_ + s * (water_density_ - air_density_);
            } else {
                rho = fluid.state_at(p_scene).density;
            }

            apply_force_at(g_part * (-rho * v_per), p_part);
        }
    }

    Real                                    volume()        const noexcept { return volume_; }
    const std::vector<geom::Vec3<PartFrame>>& sample_points() const noexcept { return samples_; }
    Real                                    water_density() const noexcept { return water_density_; }
    Real                                    air_density()   const noexcept { return air_density_; }

private:
    Real                              volume_;
    std::vector<geom::Vec3<PartFrame>>  samples_;
    Real                              water_density_;
    Real                              air_density_;
    Real                              surface_smoothing_ = Real(0.05f);
};

} // namespace manta::parts
