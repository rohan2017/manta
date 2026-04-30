#pragma once

#include "fluid_field.hpp"

namespace manta::fields {

// Two-layer fluid: water below the sea surface, air above. Sea surface is a
// horizontal plane at z = sea_level (flat-ocean approximation; no waves or
// curvature for now). Currents and winds default to zero but can be set
// independently.
//
// Smoothness across the boundary is intended to be handled by the consumer
// (e.g. Hull samples multiple points at varying depths). This field returns
// a hard step in density at the surface.
class OceanAtmosField : public FluidField {
public:
    OceanAtmosField(Real sea_level     = Real(0),
                    Real water_density = Real(1000.0f),
                    Real air_density   = Real(1.225f)) noexcept
        : sea_level_(sea_level)
        , water_density_(water_density)
        , air_density_(air_density) {}

    FluidState state_at(const geom::Vec3<SceneFrame>& pos) const noexcept override {
        if (pos.z() < sea_level_) {
            return {water_density_, current_};
        }
        return {air_density_, wind_};
    }

    // Signed altitude above sea level. Positive in air, negative underwater.
    Real height_above_sea_level(const geom::Vec3<SceneFrame>& pos) const noexcept {
        return pos.z() - sea_level_;
    }

    void set_sea_level    (Real z)                              noexcept { sea_level_     = z; }
    void set_water_density(Real d)                              noexcept { water_density_ = d; }
    void set_air_density  (Real d)                              noexcept { air_density_   = d; }
    void set_current      (const geom::Vec3<SceneFrame>& v)       noexcept { current_       = v; }
    void set_wind         (const geom::Vec3<SceneFrame>& v)       noexcept { wind_          = v; }

    Real sea_level()     const noexcept { return sea_level_; }
    Real water_density() const noexcept { return water_density_; }
    Real air_density()   const noexcept { return air_density_; }
    const geom::Vec3<SceneFrame>& current() const noexcept { return current_; }
    const geom::Vec3<SceneFrame>& wind()    const noexcept { return wind_; }

private:
    Real                 sea_level_;
    Real                 water_density_;
    Real                 air_density_;
    geom::Vec3<SceneFrame> current_;
    geom::Vec3<SceneFrame> wind_;
};

} // namespace manta::fields
