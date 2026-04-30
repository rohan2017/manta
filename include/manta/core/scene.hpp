#pragma once

#include <vector>
#include "../geom/ori.hpp"
#include "../geom/static_link.hpp"
#include "../geom/vec3.hpp"
#include "frame.hpp"

namespace manta {

template <class Scalar> class CraftT;
class World;
class Planet;

// Initial conditions applied to a craft when it joins a scene. Default values
// are the natural identity: at origin, axis-aligned, at rest.
//
// The state lives at the boundary (Scene::add_craft), not on the Craft. Crafts
// don't own initial state — they just expose runtime mutation via
// Craft::set_position / set_orientation / set_vel_*.
struct InitialState {
    geom::Vec3<SceneFrame> position    = geom::Vec3<SceneFrame>::zero();
    geom::Ori<SceneFrame>  orientation = geom::Ori<SceneFrame>::identity();
    geom::Vec3<SceneFrame> vel_linear  = geom::Vec3<SceneFrame>::zero();
    geom::Vec3<CraftFrame> vel_angular = geom::Vec3<CraftFrame>::zero();
};

// A floating-origin reference frame shared by a set of nearby crafts. Each
// craft's scene_to_craft_ transform is relative to this scene's origin.
//
// For now the floating origin is a static translation in PlanetFrame. Auto-
// rebasing (shifting the origin when the centroid drifts) is deferred.
class Scene {
public:
    Scene() noexcept = default;

    // Craft lifecycle. Craft ownership stays with the caller; the Scene holds
    // a non-owning pointer. Calling add_craft wires the craft's world_ pointer.
    //
    // The two-argument overload applies the InitialState to the craft's
    // scene-frame kinematic state at add time — this is the codegen-preferred
    // path (initial conditions don't live on the Craft object, they're
    // declared at the boundary).
    //
    // The one-argument overload leaves the craft's existing kinematic state
    // untouched. Useful for runtime mutation patterns (call set_position /
    // set_vel_* on the craft before or after add_craft).
    void add_craft(CraftT<Real>& c);
    void add_craft(CraftT<Real>& c, const InitialState& init);
    void remove_craft(CraftT<Real>& c);

    const std::vector<CraftT<Real>*>& crafts() const noexcept { return crafts_; }

    // Call once per tick. Drives all craft updates with the given dt.
    void update(float dt);

    // Floating origin: position of the scene in PlanetFrame.
    const geom::StaticLink<PlanetFrame, SceneFrame>& planet_to_scene() const noexcept {
        return planet_to_scene_;
    }
    void set_planet_to_scene(const geom::StaticLink<PlanetFrame, SceneFrame>& sl) noexcept {
        planet_to_scene_ = sl;
    }

    World& world();
    const World& world() const;
    bool has_world() const noexcept { return world_ != nullptr; }

    // Planet anchor (optional). When non-null, the scene's planet_to_scene_
    // transform is interpreted as living inside this planet's PlanetFrame;
    // the kinematic chain becomes World → Planet → Scene → Craft → Part.
    // When null (default), the scene parents directly to World identity.
    void          set_planet(Planet* p) noexcept { planet_ = p; }
    Planet*       planet()       noexcept { return planet_; }
    const Planet* planet() const noexcept { return planet_; }
    bool          has_planet() const noexcept { return planet_ != nullptr; }

private:
    friend class World;

    std::vector<CraftT<Real>*> crafts_;
    World*              world_         = nullptr;
    Planet*             planet_        = nullptr;
    geom::StaticLink<PlanetFrame, SceneFrame> planet_to_scene_ =
        geom::StaticLink<PlanetFrame, SceneFrame>::identity();
};

} // namespace manta
