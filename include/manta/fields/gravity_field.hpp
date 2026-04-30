#pragma once

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

// Uniform gravitational field. Query g() in your part's update() and apply
// the resulting force: apply_force_at(craft().field<GravityField>().g_part() * mass_).
//
// The gravity vector is stored in SceneFrame. Convenience helper g_scene() for
// direct use and g_part(part) for the part-frame vector.
class GravityField : public Field {
public:
    // g defaults to -9.81 m/s^2 in the -z direction (NED convention: z points down
    // in many aerospace sims, but here we use z-up so gravity is -z).
    explicit GravityField(geom::Vec3<SceneFrame> g =
                              geom::Vec3<SceneFrame>{Real(0), Real(0), Real(-9.81f)}) noexcept
        : g_(g) {}

    // Gravity vector in scene frame.
    const geom::Vec3<SceneFrame>& g() const noexcept { return g_; }

    void set_g(const geom::Vec3<SceneFrame>& g) noexcept { g_ = g; }

private:
    geom::Vec3<SceneFrame> g_;
};

} // namespace manta::fields
