#pragma once

#include "../../core/part.hpp"
#include "../../coupling/tether.hpp"

namespace manta::parts {

// One end of a Tether. A regular Part — belongs to exactly one Craft, attaches
// at the part's origin (set via set_transform on the part for a non-origin
// attach). The pair of endpoints sharing a Tether may live on the same craft
// or on different crafts within the same Scene.
//
// In update(), reads its own and the sibling's scene-frame state from the
// kinematic cache, computes the spring-damper force, and applies it. Both
// endpoints compute symmetrically; the equal-and-opposite forces fall out of
// the geometry.
class TetherEndpoint : public Part {
public:
    // is_a: which slot to register in. Use true for the first endpoint of a
    // tether, false for the second. Mostly bookkeeping; the dynamics are
    // symmetric.
    TetherEndpoint(std::string name, coupling::Tether& tether, bool is_a)
        : Part(std::move(name)), tether_(&tether) {
        if (is_a) tether.set_endpoint_a(this);
        else      tether.set_endpoint_b(this);
    }

    coupling::Tether& tether() noexcept { return *tether_; }
    const coupling::Tether& tether() const noexcept { return *tether_; }

    void update() override {
        auto* sib = (tether_->endpoint_a() == this) ? tether_->endpoint_b()
                                                    : tether_->endpoint_a();
        if (!sib) return;

        auto p_self = position<SceneFrame>();
        auto p_sib  = sib->position<SceneFrame>();
        auto v_self = velocity<SceneFrame>();
        auto v_sib  = sib->velocity<SceneFrame>();

        auto F_scene = tether_->force_on_self(p_self, p_sib, v_self, v_sib);

        // Rotate scene-frame force into this endpoint's part frame and apply
        // at the part origin (no torque about origin).
        auto q_part_from_scene = orientation<SceneFrame>().raw().conjugate();
        auto F_part = geom::Vec3<PartFrame>::from_raw(q_part_from_scene * F_scene.raw());
        apply_force_at(F_part);
    }

private:
    coupling::Tether* tether_;
};

} // namespace manta::parts
