#include <algorithm>
#include <cassert>

#include "manta/core/craft.hpp"
#include "manta/core/planet.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"

namespace manta {

void Scene::add_craft(CraftT<Real>& c) {
    c.world_ = world_;  // propagate world pointer so parts can resolve global fields
    c.scene_ = this;
    crafts_.push_back(&c);
}

void Scene::add_craft(CraftT<Real>& c, const InitialState& init) {
    add_craft(c);
    c.set_position   (init.position);
    c.set_orientation(init.orientation);
    c.set_vel_linear (init.vel_linear);
    c.set_vel_angular(init.vel_angular);
}

void Scene::remove_craft(CraftT<Real>& c) {
    auto it = std::find(crafts_.begin(), crafts_.end(), &c);
    if (it != crafts_.end()) {
        c.world_ = nullptr;
        c.scene_ = nullptr;
        crafts_.erase(it);
    }
}

void Scene::update(float dt) {
    // Refresh world_to_scene_ from the planet anchor (if any). Without a
    // planet, the chain collapses to planet_to_scene_ reinterpreted as a
    // static WorldFrame→SceneFrame link (planet IS world). With a planet,
    // we compose the planet's KinematicLink<World, Planet> with the static
    // PlanetFrame→SceneFrame link so any rotation/translation rate of the
    // planet is carried into world_to_scene_.
    if (planet_) {
        world_to_scene_ = planet_->world_to_planet() * planet_to_scene_;
    } else {
        // Reinterpret the static link's data as living in WorldFrame.
        world_to_scene_ = geom::KinematicLink<WorldFrame, SceneFrame>{
            geom::Vec3<WorldFrame>::from_raw(planet_to_scene_.position().raw()),
            geom::Ori<WorldFrame>{planet_to_scene_.orientation().raw()},
            geom::Vec3<WorldFrame>::zero(),
            geom::Vec3<SceneFrame>::zero(),
        };
    }

    // Phase 1 — kinematic pass for ALL crafts → globally consistent
    //           scene_to_part_ caches.
    for (CraftT<Real>* c : crafts_) c->kinematic_pass();
    // Phase 2 — per-part update + wrench aggregation + accel computation.
    for (CraftT<Real>* c : crafts_) c->sense_and_aggregate();
    // Phase 3 — each craft advances its own state independently.
    for (CraftT<Real>* c : crafts_) c->integrate(Real(dt));
}

World& Scene::world() {
    assert(world_ && "Scene is not attached to a World");
    return *world_;
}

const World& Scene::world() const {
    assert(world_ && "Scene is not attached to a World");
    return *world_;
}

// CraftT cross-class methods (field_ptr / scene / world) are defined inline
// at the bottom of craft.hpp; no explicit instantiation needed here.

} // namespace manta
