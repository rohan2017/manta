#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "../geom/casts.hpp"
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

// The full rigid-body state of a craft expressed in its scene's frame.
// One canonical struct shared by Scene::add_craft (boundary), Craft state
// setters (runtime mutation), and the EKF (filter belief).
template <class Scalar = MFloat>
struct CraftStateT {
    geom::Vec3<SceneFrame, Scalar> position    = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Ori<SceneFrame, Scalar>  orientation = geom::Ori<SceneFrame, Scalar>::identity();
    geom::Vec3<SceneFrame, Scalar> vel_linear  = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Vec3<CraftFrame, Scalar> vel_angular = geom::Vec3<CraftFrame, Scalar>::zero();
};
using CraftState = CraftStateT<MFloat>;

// Boundary alias: `InitialState` is the value-typed CraftState used by
// Scene::add_craft. Defining it as an alias keeps a single canonical
// state shape; the per-Scalar Scene path lifts the value fields into
// Scalar at add time.
using InitialState = CraftStateT<MFloat>;

// A floating-origin reference frame shared by a set of nearby crafts. Each
// craft's scene_to_craft_ transform is relative to this scene's origin.
//
// SceneT is templated on the same Scalar as the crafts it holds. The MFloat
// instantiation drives the sim; a Jet instantiation drives the EKF's
// Jacobian step, with planets' world_to_planet link cast into Jet (their
// motion is a non-estimated input, so the cast carries zero gradient — i.e.
// the EKF treats planet pose as constant within a tick).
//
// Lifetime: Scene holds non-owning craft pointers. Either side can be
// destroyed first — Scene's dtor unbinds remaining crafts (clears their
// scene_/world_ pointers), and CraftT's dtor removes itself from its
// owning Scene.
template <class Scalar>
class SceneT {
public:
    SceneT() noexcept = default;
    ~SceneT() {
        // Unbind any remaining crafts so a craft outliving its Scene
        // observes scene_ == nullptr instead of a dangling pointer.
        for (CraftT<Scalar>* c : crafts_) {
            if (c) { c->scene_ = nullptr; c->world_ = nullptr; }
        }
    }
    SceneT(const SceneT&) = delete;
    SceneT& operator=(const SceneT&) = delete;

    // Craft lifecycle. Craft ownership stays with the caller; the Scene holds
    // a non-owning pointer. add_craft wires the craft's world_/scene_ pointers.
    void add_craft(CraftT<Scalar>& c) {
        c.world_ = world_;
        c.scene_ = this;
        crafts_.push_back(&c);
    }

    void add_craft(CraftT<Scalar>& c, const InitialState& init) {
        add_craft(c);
        typename CraftT<Scalar>::State s;
        s.position    = geom::lift_from_real<Scalar>(init.position);
        s.orientation = geom::lift_from_real<Scalar>(init.orientation);
        s.vel_linear  = geom::lift_from_real<Scalar>(init.vel_linear);
        s.vel_angular = geom::lift_from_real<Scalar>(init.vel_angular);
        c.set_state(s);
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

    // Refresh world_to_scene_, run the kinematic pass and force
    // aggregation for every craft, but do NOT advance state. Leaves each
    // craft's `acc_linear` / `acc_angular` populated so sensors can be
    // queried without integrating.
    void kinematic_and_aggregate() {
        refresh_world_to_scene();
        for (CraftT<Scalar>* c : crafts_) c->kinematic_pass();
        for (CraftT<Scalar>* c : crafts_) c->sense_and_aggregate();
    }

    // Advance every craft by `dt` using the currently-cached
    // acceleration, but do NOT re-aggregate afterward. Used by the EKF
    // predict step (after a one-shot `kinematic_and_aggregate()` at
    // x_pre): h and H are extracted from the pre-integrate cache, and
    // the next predict re-seeds and re-evaluates from scratch anyway.
    void integrate(Scalar dt) {
        for (CraftT<Scalar>* c : crafts_) c->integrate_pre_aggregate(dt);
        for (CraftT<Scalar>* c : crafts_) c->integrate_post_aggregate(dt);
    }

    // Per-tick driver. End-of-tick invariant: every craft's pos/q/v AND
    // its cached acc_linear/acc_angular are at the new state x_k —
    // sensors that read the cache (IMU specific force) report the
    // CURRENT acceleration, not the previous tick's.
    //
    // Tick 1 integrates with a default-zero cache (no kick), giving a
    // drift-only step before the first aggregate populates real
    // acceleration values for tick 2+. That introduces a one-tick lag
    // on whatever forces would have acted at t=0 (typically gravity);
    // at 1 kHz it's a 1 ms delay, negligible. Callers that need
    // pre-integrate state can prime the cache explicitly with
    // `kinematic_and_aggregate()` before the first `step()`.
    void step(Scalar dt) {
        refresh_world_to_scene();
        for (CraftT<Scalar>* c : crafts_) c->integrate_pre_aggregate(dt);
        for (CraftT<Scalar>* c : crafts_) c->kinematic_pass();
        for (CraftT<Scalar>* c : crafts_) c->sense_and_aggregate();
        for (CraftT<Scalar>* c : crafts_) c->integrate_post_aggregate(dt);
    }


private:
    // Refresh world_to_scene_ from the planet anchor (if any). Planets
    // are value-typed; we cast their KinematicLink into Scalar component-
    // wise. For Scalar=Jet, the cast carries zero gradients — the EKF
    // treats planet pose as a non-estimated input.
    void refresh_world_to_scene() {
        if (planet_) {
            // Compose value→Scene moving link with the static Planet→Scene
            // link (just a translation; same at either Scalar).
            auto wp_s = geom::cast_kinematic_link<Scalar>(planet_->world_to_planet());
            world_to_scene_ = wp_s * geom::cast_static_link<Scalar>(planet_to_scene_real_);
        } else {
            // No planet anchor: the planet IS the world. Reinterpret the
            // static planet→scene link as a kinematic link in WorldFrame.
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
    }

public:

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

    // Planet anchor (optional). Planets are value-typed and shared between
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
    friend class WorldT<Scalar>;

    std::vector<CraftT<Scalar>*>                          crafts_;
    WorldT<Scalar>*                                       world_      = nullptr;
    Planet*                                               planet_     = nullptr;
    geom::StaticLink<PlanetFrame, SceneFrame>             planet_to_scene_real_ =
        geom::StaticLink<PlanetFrame, SceneFrame>::identity();
    geom::KinematicLink<WorldFrame, SceneFrame, Scalar>   world_to_scene_;
};

// value instantiation alias.
using Scene = SceneT<MFloat>;

} // namespace manta
