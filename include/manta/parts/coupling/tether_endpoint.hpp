#pragma once

#include "../../core/part.hpp"
#include "../../coupling/tether.hpp"

namespace manta::parts {

// One end of a Tether. A regular Part — belongs to exactly one Craft, attaches
// at the part's origin (set via set_transform on the part for a non-origin
// attach). The pair of endpoints sharing a Tether may live on the same craft
// or on different crafts within the same Scene.
//
// Templated on Scalar; the shared TetherT<Scalar> instance must use the same
// Scalar as both endpoints.
//
// In update(), reads its own and the sibling's scene-frame state from the
// kinematic cache, computes the spring-damper force, and applies it. Both
// endpoints compute symmetrically; the equal-and-opposite forces fall out of
// the geometry.
template <class Scalar = Real>
class TetherEndpointT : public PartT<Scalar> {
public:
    // is_a: which slot to register in. Use true for the first endpoint of a
    // tether, false for the second. Mostly bookkeeping; the dynamics are
    // symmetric.
    TetherEndpointT(std::string name, coupling::TetherT<Scalar>& tether, bool is_a)
        : PartT<Scalar>(std::move(name)), tether_(&tether) {
        if (is_a) tether.set_endpoint_a(this);
        else      tether.set_endpoint_b(this);
    }

    // Unlink from the tether on destruction so the surviving side observes
    // a nullable endpoint rather than a dangling pointer.
    ~TetherEndpointT() override {
        if (!tether_) return;
        if (tether_->endpoint_a() == this) tether_->set_endpoint_a(nullptr);
        if (tether_->endpoint_b() == this) tether_->set_endpoint_b(nullptr);
    }

    coupling::TetherT<Scalar>& tether() noexcept { return *tether_; }
    const coupling::TetherT<Scalar>& tether() const noexcept { return *tether_; }

    void update() override {
        if (!tether_) return;
        auto* sib = (tether_->endpoint_a() == this) ? tether_->endpoint_b()
                                                    : tether_->endpoint_a();
        if (!sib) return;

        auto p_self = this->template position<SceneFrame>();
        auto p_sib  = sib->template position<SceneFrame>();
        auto v_self = this->template velocity<SceneFrame>();
        auto v_sib  = sib->template velocity<SceneFrame>();

        auto F_scene = tether_->force_on_self(p_self, p_sib, v_self, v_sib);

        // Rotate scene-frame force into this endpoint's part frame and apply
        // at the part origin (no torque about origin).
        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();
        Eigen::Matrix<Scalar, 3, 1> F_part_e = q_part_from_scene * F_scene.raw();
        this->apply_force_at(geom::Vec3<PartFrame, Scalar>::from_raw(F_part_e));
    }

private:
    template <class S> friend class manta::coupling::TetherT;

    coupling::TetherT<Scalar>* tether_;
};

using TetherEndpoint = TetherEndpointT<Real>;

} // namespace manta::parts

// TetherT dtor needs TetherEndpointT's tether_ field accessible. Defined
// here once both classes are complete.
namespace manta::coupling {
template <class Scalar>
inline TetherT<Scalar>::~TetherT() {
    if (a_) a_->tether_ = nullptr;
    if (b_) b_->tether_ = nullptr;
}
} // namespace manta::coupling

