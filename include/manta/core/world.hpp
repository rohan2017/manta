#pragma once

#include <memory>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "planet.hpp"
#include "scene.hpp"
#include "../fields/field.hpp"
#include "../geom/static_link.hpp"
#include "../sim/sim_clock.hpp"

namespace manta {

// Top-level simulation container, templated on Scalar.
//
// `WorldT<MFloat>` (the alias `World`) drives the sim: real-time tick loop,
// real-valued state. `WorldT<Jet>` is the Jacobian-step shadow used by
// `EKF` — same scenes/crafts/fields registered, but every per-tick
// computation runs through Ceres Jets so `predict()` extracts the exact
// state-transition Jacobian via autodiff.
//
// Planets are NOT templated. They're treated as a non-estimated input to
// the filter: their pose/rotation come from a value-valued `Planet`
// instance; the Jet-world casts the planet's KinematicLink into Jet with
// zero gradient. That's a deliberate "selective Jet propagation" choice —
// the EKF doesn't estimate planet motion.
//
// Fields are likewise not templated; field queries from inside Jet-typed
// craft code go through `state_at_templated<Scalar>` which casts/finite-
// diffs as needed.
template <class Scalar_>
class WorldT {
public:
    using Scalar = Scalar_;
    WorldT() noexcept = default;

    // --- Scene management ---
    SceneT<Scalar>& create_scene(
            const geom::StaticLink<PlanetFrame, SceneFrame>& origin =
                geom::StaticLink<PlanetFrame, SceneFrame>::identity()) {
        auto s = std::make_unique<SceneT<Scalar>>();
        s->world_ = this;
        s->planet_to_scene_real_ = origin;
        scenes_.push_back(std::move(s));
        return *scenes_.back();
    }

    const std::vector<std::unique_ptr<SceneT<Scalar>>>& scenes() const noexcept {
        return scenes_;
    }
    std::vector<std::unique_ptr<SceneT<Scalar>>>& scenes() noexcept {
        return scenes_;
    }

    // --- Planet management ---
    // Planets are constructed value-typed (they're shared between value and
    // Jet worlds). add_planet<P>(...) constructs P, calls
    // register_disturbances on this World (which value-side gives the
    // planet a chance to add its standing field disturbances), and
    // returns a reference. For a Jet World, planets aren't reconstructed
    // — the user (or codegen) shares a single Planet instance across
    // both worlds.
    template <typename P, typename... Args>
    P& add_planet(Args&&... args) {
        auto ptr = std::make_unique<P>(std::forward<Args>(args)...);
        P& ref = *ptr;
        // Only the value World invokes register_disturbances — a planet's
        // disturbances on shared fields shouldn't be added twice when the
        // codegen also constructs a Jet World on the same planet.
        if constexpr (std::is_same_v<Scalar, MFloat>) {
            ref.register_disturbances(*this);
        }
        planets_.push_back(std::move(ptr));
        return ref;
    }

    const std::vector<std::unique_ptr<Planet>>& planets() const noexcept { return planets_; }
    std::vector<std::unique_ptr<Planet>>&       planets()       noexcept { return planets_; }

    // --- Global field registry (shared between Worlds at different Scalars) ---
    //
    // IMPORTANT: register every field BEFORE adding any craft. Crafts'
    // parts capture field references during compute_params, which runs
    // soon after `Scene::add_craft`. Registering a field after that point
    // means parts already added never see it — silently broken behavior
    // (e.g. an IMU's gravity subtraction is skipped). This trap surfaces
    // the mistake at the offending register_field call instead.
    template<typename FieldT>
    void register_field(FieldT& f) {
        if (crafts_added_) {
            throw std::runtime_error(
                "WorldT::register_field: a craft was already added to this "
                "world. Register all fields BEFORE adding any craft so the "
                "crafts' parts can resolve field references during "
                "compute_params.");
        }
        fields_[std::type_index(typeid(FieldT))] = &f;
    }

    // Internal hook called by SceneT::add_craft so the World can lock its
    // field registry. Not part of the public API; users should not call
    // this directly.
    void mark_crafts_added() noexcept { crafts_added_ = true; }

    // Has any craft been added to any scene yet? Used by CraftView to
    // skip duplicate field mirroring on subsequent registrations.
    bool crafts_added() const noexcept { return crafts_added_; }

    // Mirror every field registration from `other` into this world.
    // Fields are not Scalar-templated — a single `GravityField` instance
    // can be shared across `WorldT<double>` and `WorldT<ceres::Jet<...>>`.
    // Used by the EKF's CraftView to populate the Jet shadow world from
    // the user's value-side world without making the user re-register
    // each field.
    template <class OtherScalar>
    void mirror_fields_from(const WorldT<OtherScalar>& other) {
        if (crafts_added_) {
            throw std::runtime_error(
                "WorldT::mirror_fields_from: a craft was already added to "
                "this world. Mirror fields BEFORE adding any craft.");
        }
        for (const auto& [ti, ptr] : other.fields_for_mirror()) {
            fields_[ti] = ptr;
        }
    }

    // Read-only access to the field map for cross-Scalar mirroring.
    const std::unordered_map<std::type_index, fields::Field*>&
    fields_for_mirror() const noexcept { return fields_; }

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

    // Run kinematic + aggregate for every scene without integrating,
    // advancing planets, updating fields, or ticking the clock. Used by
    // EKF measurement functors that need h(x) to query post-aggregation
    // sensor values (e.g. an IMU's body-frame inertial acceleration)
    // without changing the world's canonical state.
    void kinematic_and_aggregate() {
        for (auto& scene : scenes_) scene->kinematic_and_aggregate();
    }

    // Advance every scene's crafts using their currently-cached
    // accelerations — no re-aggregate, no clock tick, no field/planet
    // advance. Used by the EKF's `end_step()` to push the Jet world
    // from x_pre to x_post.
    void integrate(Scalar dt) {
        for (auto& scene : scenes_) scene->integrate(dt);
    }

    // --- Main step ---
    // One full simulation tick using clock_.dt():
    //   1. Planets advance (value world only — Jet world reuses the
    //      value-side planet state via Scene's casting).
    //   2. All scenes step their crafts.
    //   3. All registered fields update (value world only — fields are
    //      shared, updated once).
    //   4. Clock advances.
    void step() {
        const MFloat t    = MFloat(clock_.time());
        const MFloat dt_r = MFloat(clock_.dt());

        if constexpr (std::is_same_v<Scalar, MFloat>) {
            for (auto& p : planets_) p->update(t, dt_r);
        }

        const Scalar dt_s = Scalar(clock_.dt());
        for (auto& scene : scenes_) scene->step(dt_s);

        if constexpr (std::is_same_v<Scalar, MFloat>) {
            for (auto& [ti, f] : fields_) f->update();
            clock_.advance();
        }
    }


private:
    std::vector<std::unique_ptr<SceneT<Scalar>>> scenes_;
    std::vector<std::unique_ptr<Planet>>         planets_;
    std::unordered_map<std::type_index, fields::Field*> fields_;
    SimClock                                     clock_;
    bool                                         crafts_added_ = false;
};

// value instantiation alias — the sim's primary World type.
using World = WorldT<MFloat>;

} // namespace manta
