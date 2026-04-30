#include <algorithm>
#include <cassert>

#include "manta/core/craft.hpp"
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
