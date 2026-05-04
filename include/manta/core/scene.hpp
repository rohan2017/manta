#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "../geom/kinematic_link.hpp"
#include "../geom/ori.hpp"
#include "../geom/static_link.hpp"
#include "../geom/vec3.hpp"
#include "frame.hpp"
#include "planet.hpp"
#include "types.hpp"

namespace manta {

template <class Scalar> class CraftT;
template <class Scalar> class WorldT;

// Initial conditions applied to a craft when it joins a scene. Default values
// are the natural identity: at origin, axis-aligned, at rest.
//
// The state lives at the boundary (Scene::add_craft), not on the Craft. Crafts
// don't own initial state — they just expose runtime mutation via
// Craft::set_position / set_orientation / set_vel_*.
//
// Stored as Real-typed (the codegen and user-facing API speak in Real); the
// templated Scene path casts these into Scalar at add time.
struct InitialState {
    geom::Vec3<SceneFrame> position    = geom::Vec3<SceneFrame>::zero();
    geom::Ori<SceneFrame>  orientation = geom::Ori<SceneFrame>::identity();
    geom::Vec3<SceneFrame> vel_linear  = geom::Vec3<SceneFrame>::zero();
    geom::Vec3<CraftFrame> vel_angular = geom::Vec3<CraftFrame>::zero();
};

// A floating-origin reference frame shared by a set of nearby crafts. Each
// craft's scene_to_craft_ transform is relative to this scene's origin.
//
// SceneT is templated on the same Scalar as the crafts it holds. The Real
// instantiation drives the sim; a Jet instantiation drives the EKF's
// Jacobian step, with planets' world_to_planet link cast into Jet (their
// motion is a non-estimated input, so the cast carries zero gradient — i.e.
// the EKF treats planet pose as constant within a tick).
template <class Scalar>
class SceneT {
public:
    SceneT() noexcept = default;

    // Craft lifecycle. Craft ownership stays with the caller; the Scene holds
    // a non-owning pointer. add_craft wires the craft's world_/scene_ pointers.
    void add_craft(CraftT<Scalar>& c) {
        c.world_ = world_;
        c.scene_ = this;
        crafts_.push_back(&c);
    }

    void add_craft(CraftT<Scalar>& c, const InitialState& init) {
        add_craft(c);
        c.set_position(geom::Vec3<SceneFrame, Scalar>::from_raw(
            init.position.raw().template cast<Scalar>()));
        Eigen::Quaternion<Scalar> q_init(
            Scalar(init.orientation.raw().w()),
            Scalar(init.orientation.raw().x()),
            Scalar(init.orientation.raw().y()),
            Scalar(init.orientation.raw().z()));
        c.set_orientation(geom::Ori<SceneFrame, Scalar>{q_init});
        c.set_vel_linear(geom::Vec3<SceneFrame, Scalar>::from_raw(
            init.vel_linear.raw().template cast<Scalar>()));
        c.set_vel_angular(geom::Vec3<CraftFrame, Scalar>::from_raw(
            init.vel_angular.raw().template cast<Scalar>()));
    }

    void remove_craft(CraftT<Scalar>& c) {
        auto it = std::find(crafts_.begin(), crafts_.end(), &c);
        if (it != crafts_.end()) {
            c.world_ = nullptr;
            c.scene_ = nullptr;
            crafts_.erase(it);
        }
    }

    const std::vector<CraftT<Scalar>*>& crafts() const noexcept { return crafts_; }

    // Call once per tick. Drives all crafts in three phases (kinematic /
    // sense+aggregate / integrate) so cross-craft couplings see globally
    // consistent state.
    void update(Scalar dt) {
        // Refresh world_to_scene_ from the planet anchor (if any). Planets
        // are Real-typed; we cast their KinematicLink into Scalar component-
        // wise. For Scalar=Jet, the cast carries zero gradients — the EKF
        // treats planet pose as a non-estimated input.
        if (planet_) {
            using KL_real = geom::KinematicLink<WorldFrame, PlanetFrame, Real>;
            using SL_ps   = geom::StaticLink<PlanetFrame, SceneFrame, Scalar>;
            const KL_real& wp = planet_->world_to_planet();

            geom::KinematicLink<WorldFrame, PlanetFrame, Scalar> wp_s{
                geom::Vec3<WorldFrame, Scalar>::from_raw(
                    wp.position().raw().template cast<Scalar>()),
                geom::Ori<WorldFrame, Scalar>{Eigen::Quaternion<Scalar>(
                    Scalar(wp.orientation().raw().w()),
                    Scalar(wp.orientation().raw().x()),
                    Scalar(wp.orientation().raw().y()),
                    Scalar(wp.orientation().raw().z()))},
                geom::Vec3<WorldFrame, Scalar>::from_raw(
                    wp.vel_linear().raw().template cast<Scalar>()),
                geom::Vec3<PlanetFrame, Scalar>::from_raw(
                    wp.vel_angular().raw().template cast<Scalar>()),
                geom::Vec3<WorldFrame, Scalar>::from_raw(
                    wp.acc_linear().raw().template cast<Scalar>()),
                geom::Vec3<PlanetFrame, Scalar>::from_raw(
                    wp.acc_angular().raw().template cast<Scalar>()),
            };
            // Compose with the static planet→scene link (same identity at
            // either Scalar — it's just a translation).
            SL_ps ps_s = static_link_planet_to_scene<Scalar>();
            world_to_scene_ = wp_s * ps_s;
        } else {
            // No planet anchor: the static planet→scene link reinterpreted
            // as living in WorldFrame (the planet IS the world).
            world_to_scene_ = geom::KinematicLink<WorldFrame, SceneFrame, Scalar>{
                geom::Vec3<WorldFrame, Scalar>::from_raw(
                    planet_to_scene_real_.position().raw().template cast<Scalar>()),
                geom::Ori<WorldFrame, Scalar>{Eigen::Quaternion<Scalar>(
                    Scalar(planet_to_scene_real_.orientation().raw().w()),
                    Scalar(planet_to_scene_real_.orientation().raw().x()),
                    Scalar(planet_to_scene_real_.orientation().raw().y()),
                    Scalar(planet_to_scene_real_.orientation().raw().z()))},
                geom::Vec3<WorldFrame, Scalar>::zero(),
                geom::Vec3<SceneFrame, Scalar>::zero(),
            };
        }

        // Phase 1 — kinematic pass for ALL crafts.
        for (CraftT<Scalar>* c : crafts_) c->kinematic_pass();
        // Phase 2 — per-part update + wrench aggregation + accel computation.
        for (CraftT<Scalar>* c : crafts_) c->sense_and_aggregate();
        // Phase 3 — each craft advances its own state independently.
        for (CraftT<Scalar>* c : crafts_) c->integrate(dt);
    }

    // Floating origin: position of the scene in PlanetFrame.
    const geom::StaticLink<PlanetFrame, SceneFrame>& planet_to_scene() const noexcept {
        return planet_to_scene_real_;
    }
    void set_planet_to_scene(const geom::StaticLink<PlanetFrame, SceneFrame>& sl) noexcept {
        planet_to_scene_real_ = sl;
    }

    WorldT<Scalar>& world() {
        assert(world_ && "Scene is not attached to a World");
        return *world_;
    }
    const WorldT<Scalar>& world() const {
        assert(world_ && "Scene is not attached to a World");
        return *world_;
    }
    bool has_world() const noexcept { return world_ != nullptr; }

    // Planet anchor (optional). Planets are Real-typed and shared between
    // value and Jet Worlds — they aren't part of the EKF state.
    void          set_planet(Planet* p) noexcept { planet_ = p; }
    Planet*       planet()       noexcept { return planet_; }
    const Planet* planet() const noexcept { return planet_; }
    bool          has_planet() const noexcept { return planet_ != nullptr; }

    // Full WorldFrame → SceneFrame kinematic link, refreshed each tick.
    const geom::KinematicLink<WorldFrame, SceneFrame, Scalar>& world_to_scene() const noexcept {
        return world_to_scene_;
    }

private:
    template <class S>
    geom::StaticLink<PlanetFrame, SceneFrame, S> static_link_planet_to_scene() const noexcept {
        return geom::StaticLink<PlanetFrame, SceneFrame, S>{
            geom::Vec3<PlanetFrame, S>::from_raw(
                planet_to_scene_real_.position().raw().template cast<S>()),
            geom::Ori<PlanetFrame, S>{Eigen::Quaternion<S>(
                S(planet_to_scene_real_.orientation().raw().w()),
                S(planet_to_scene_real_.orientation().raw().x()),
                S(planet_to_scene_real_.orientation().raw().y()),
                S(planet_to_scene_real_.orientation().raw().z()))},
        };
    }

    friend class WorldT<Scalar>;

    std::vector<CraftT<Scalar>*>                          crafts_;
    WorldT<Scalar>*                                       world_      = nullptr;
    Planet*                                               planet_     = nullptr;
    geom::StaticLink<PlanetFrame, SceneFrame>             planet_to_scene_real_ =
        geom::StaticLink<PlanetFrame, SceneFrame>::identity();
    geom::KinematicLink<WorldFrame, SceneFrame, Scalar>   world_to_scene_;
};

// Real instantiation alias.
using Scene = SceneT<Real>;

} // namespace manta
