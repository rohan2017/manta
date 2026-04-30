#pragma once

#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include "scene.hpp"
#include "../fields/field.hpp"
#include "../geom/static_link.hpp"
#include "../sim/sim_clock.hpp"

namespace manta {

// Top-level simulation container. Owns scenes and holds (non-owning) pointers
// to fields. Orchestrates the per-tick update: crafts run first (parallel in
// the future), then fields drain contributions and update, then the clock
// advances.
class World {
public:
    World() noexcept = default;

    // --- Scene management ---
    // Creates a new scene owned by this world. Optionally position the scene
    // origin in PlanetFrame.
    Scene& create_scene(const geom::StaticLink<PlanetFrame, SceneFrame>& origin =
                            geom::StaticLink<PlanetFrame, SceneFrame>::identity());

    const std::vector<std::unique_ptr<Scene>>& scenes() const noexcept { return scenes_; }

    // --- Global field registry ---
    // Fields are NOT owned by World; the caller manages their lifetime.
    template<typename FieldT>
    void register_field(FieldT& f) {
        fields_[std::type_index(typeid(FieldT))] = &f;
    }

    template<typename FieldT>
    FieldT& field() {
        return *static_cast<FieldT*>(field_ptr(typeid(FieldT)));
    }
    template<typename FieldT>
    const FieldT& field() const {
        return *static_cast<const FieldT*>(field_ptr(typeid(FieldT)));
    }

    fields::Field* field_ptr(const std::type_info& ti) const {
        auto it = fields_.find(std::type_index(ti));
        return (it != fields_.end()) ? it->second : nullptr;
    }

    // --- Clock ---
    SimClock&       clock()       noexcept { return clock_; }
    const SimClock& clock() const noexcept { return clock_; }

    // --- Main update ---
    // One full simulation tick using clock_.dt():
    //   1. All scenes update their crafts (craft phase).
    //   2. All registered fields call update() (field phase).
    //   3. Clock advances by dt.
    void update();

private:
    std::vector<std::unique_ptr<Scene>>           scenes_;
    std::unordered_map<std::type_index, fields::Field*> fields_;
    SimClock                                       clock_;
};

} // namespace manta
