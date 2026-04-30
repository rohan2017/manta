#pragma once

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// Snapshot of fluid state at a point. Density is volumetric mass density in
// kg/m^3; velocity is the bulk fluid velocity (current/wind) in scene frame.
struct FluidState {
    Real                 density;   // kg/m^3
    geom::Vec3<SceneFrame> velocity;  // m/s in scene frame
};

// Abstract base for fluid environments. Subclasses provide spatially-varying
// density (e.g. atmosphere altitude profile) and bulk velocity (winds,
// currents). Parts that interact with a fluid (Hull buoyancy, AeroSurface
// drag, etc.) query state_at() each tick at their sample points.
class FluidField : public Field {
public:
    virtual FluidState state_at(const geom::Vec3<SceneFrame>& pos) const noexcept = 0;
};

} // namespace manta::fields
