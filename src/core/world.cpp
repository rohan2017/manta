#include "manta/core/world.hpp"

namespace manta {

Scene& World::create_scene(const geom::StaticLink<PlanetFrame, SceneFrame>& origin) {
    auto s = std::make_unique<Scene>();
    s->world_ = this;
    s->planet_to_scene_ = origin;
    scenes_.push_back(std::move(s));
    return *scenes_.back();
}

void World::update() {
    Real t = Real(clock_.time());
    Real dt_r = Real(clock_.dt());
    float dt = clock_.dt();

    // Planet phase: advance each planet's frame transform + disturbance state
    // BEFORE the craft phase so crafts see consistent planet poses.
    for (auto& p : planets_) {
        p->update(t, dt_r);
    }

    // Craft phase: all scenes update their crafts.
    for (auto& scene : scenes_) {
        scene->update(dt);
    }

    // Field phase: each field drains buffered contributions and updates.
    for (auto& [ti, f] : fields_) {
        f->update();
    }

    clock_.advance();
}

} // namespace manta
