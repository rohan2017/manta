#pragma once

#include "fluid_field.hpp"

namespace manta::fields {

// Constant density, constant velocity everywhere. Mostly useful for tests
// (immerse a body in "infinite ocean") and quick sanity checks.
class UniformFluidField : public FluidField {
public:
    UniformFluidField(Real density,
                      geom::Vec3<SceneFrame> velocity = geom::Vec3<SceneFrame>::zero()) noexcept
        : density_(density), velocity_(velocity) {}

    FluidState state_at(const geom::Vec3<SceneFrame>&) const noexcept override {
        return {density_, velocity_};
    }

    void set_density (Real d)                            noexcept { density_  = d; }
    void set_velocity(const geom::Vec3<SceneFrame>& v)     noexcept { velocity_ = v; }
    Real density() const noexcept { return density_; }

private:
    Real                 density_;
    geom::Vec3<SceneFrame> velocity_;
};

} // namespace manta::fields
