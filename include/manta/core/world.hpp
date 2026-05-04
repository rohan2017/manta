#pragma once

#include <memory>
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
// `WorldT<Real>` (the alias `World`) drives the sim: real-time tick loop,
// real-valued state. `WorldT<Jet>` is the Jacobian-step shadow used by
// `WorldEKF` — same scenes/crafts/fields registered, but every per-tick
// computation runs through Ceres Jets so `predict()` extracts the exact
// state-transition Jacobian via autodiff.
//
// Planets are NOT templated. They're treated as a non-estimated input to
// the filter: their pose/rotation come from a Real-valued `Planet`
// instance; the Jet-world casts the planet's KinematicLink into Jet with
// zero gradient. That's a deliberate "selective Jet propagation" choice —
// the EKF doesn't estimate planet motion.
//
// Fields are likewise not templated; field queries from inside Jet-typed
// craft code go through `state_at_templated<Scalar>` which casts/finite-
// diffs as needed.
template <class Scalar>
class WorldT {
public:
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

    // --- Planet management ---
    // Planets are constructed Real-typed (they're shared between value and
    // Jet worlds). add_planet<P>(...) constructs P, calls
    // register_disturbances on this World (which Real-side gives the
    // planet a chance to add its standing field disturbances), and
    // returns a reference. For a Jet World, planets aren't reconstructed
    // — the user (or codegen) shares a single Planet instance across
    // both worlds.
    template <typename P, typename... Args>
    P& add_planet(Args&&... args) {
        auto ptr = std::make_unique<P>(std::forward<Args>(args)...);
        P& ref = *ptr;
        // Only the Real World invokes register_disturbances — a planet's
        // disturbances on shared fields shouldn't be added twice when the
        // codegen also constructs a Jet World on the same planet.
        if constexpr (std::is_same_v<Scalar, Real>) {
            ref.register_disturbances(*this);
        }
        planets_.push_back(std::move(ptr));
        return ref;
    }

    // For the Jet-world case where the user wants to reuse an existing
    // (Real-side) Planet without owning a duplicate: register the pointer
    // for shared use. The Jet World doesn't own the planet's lifetime.
    void share_planet(Planet* p) {
        shared_planets_.push_back(p);
    }

    const std::vector<std::unique_ptr<Planet>>& planets() const noexcept { return planets_; }
    std::vector<std::unique_ptr<Planet>>&       planets()       noexcept { return planets_; }

    // --- Global field registry (shared between Worlds at different Scalars) ---
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
    //   1. Planets advance (Real world only — Jet world reuses the
    //      Real-side planet state via Scene's casting).
    //   2. All scenes update their crafts.
    //   3. All registered fields update (Real world only — fields are
    //      shared, updated once).
    //   4. Clock advances.
    void update() {
        const Real t    = Real(clock_.time());
        const Real dt_r = Real(clock_.dt());

        if constexpr (std::is_same_v<Scalar, Real>) {
            for (auto& p : planets_) p->update(t, dt_r);
        }
        // Note: a Jet World does NOT advance planet state. The Real-side
        // Planet (shared via share_planet, or via the codegen path that
        // owns the planet only on the Real side) carries the canonical
        // pose; the Jet Scene casts it.

        const Scalar dt_s = Scalar(clock_.dt());
        for (auto& scene : scenes_) {
            scene->update(dt_s);
        }

        if constexpr (std::is_same_v<Scalar, Real>) {
            for (auto& [ti, f] : fields_) f->update();
        }

        if constexpr (std::is_same_v<Scalar, Real>) {
            clock_.advance();
        }
    }

private:
    std::vector<std::unique_ptr<SceneT<Scalar>>> scenes_;
    std::vector<std::unique_ptr<Planet>>         planets_;
    std::vector<Planet*>                         shared_planets_;
    std::unordered_map<std::type_index, fields::Field*> fields_;
    SimClock                                     clock_;
};

// Real instantiation alias — the sim's primary World type.
using World = WorldT<Real>;

} // namespace manta
