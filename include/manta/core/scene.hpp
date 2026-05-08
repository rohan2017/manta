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
// 13-DOF ambient representation (4-component unit quaternion).
// One canonical struct shared by Scene::add_craft (boundary), Craft state
// setters (runtime mutation), and the EKF (reference state — see below).
template <class Scalar = MFloat>
struct CraftStateT {
    geom::Vec3<SceneFrame, Scalar> position    = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Ori<SceneFrame, Scalar>  orientation = geom::Ori<SceneFrame, Scalar>::identity();
    geom::Vec3<SceneFrame, Scalar> vel_linear  = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Vec3<CraftFrame, Scalar> vel_angular = geom::Vec3<CraftFrame, Scalar>::zero();
};
using CraftState = CraftStateT<MFloat>;

using InitialState = CraftStateT<MFloat>;

// Tangent-space representation of CraftState — the *error* state used by the
// ESKF.  12-DOF: position, orientation (axis-angle so(3)), linear velocity,
// angular velocity. The orientation block is 3-DOF instead of 4 because the
// rotation manifold is 3-dimensional; the unit-quaternion ambient
// representation has a redundant radial direction that the tangent omits.
//
// Layout matches the canonical EKF tangent index layout:
//   [δp (3) | δθ (3) | δv (3) | δω (3)]
template <class Scalar = MFloat>
struct CraftTangentT {
    static constexpr int kDim = 12;
    using Vec = Eigen::Matrix<Scalar, kDim, 1>;

    geom::Vec3<SceneFrame, Scalar> position    = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Vec3<SceneFrame, Scalar> orientation = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Vec3<SceneFrame, Scalar> vel_linear  = geom::Vec3<SceneFrame, Scalar>::zero();
    geom::Vec3<CraftFrame, Scalar> vel_angular = geom::Vec3<CraftFrame, Scalar>::zero();

    static CraftTangentT zero() noexcept { return CraftTangentT{}; }

    // Pack into a flat 12-vector in the canonical [δp | δθ | δv | δω] order.
    Vec to_vec() const noexcept {
        Vec v;
        v.template segment<3>(0) = position.raw();
        v.template segment<3>(3) = orientation.raw();
        v.template segment<3>(6) = vel_linear.raw();
        v.template segment<3>(9) = vel_angular.raw();
        return v;
    }
    static CraftTangentT from_vec(const Vec& v) noexcept {
        CraftTangentT t;
        t.position    = geom::Vec3<SceneFrame, Scalar>::from_raw(v.template segment<3>(0));
        t.orientation = geom::Vec3<SceneFrame, Scalar>::from_raw(v.template segment<3>(3));
        t.vel_linear  = geom::Vec3<SceneFrame, Scalar>::from_raw(v.template segment<3>(6));
        t.vel_angular = geom::Vec3<CraftFrame, Scalar>::from_raw(v.template segment<3>(9));
        return t;
    }
};
using CraftTangent = CraftTangentT<MFloat>;

// Boxplus retraction:  x_full = x_ref ⊞ δ.
//
// Euclidean components add linearly. Orientation uses the so(3) exp map:
//   q_full = q_ref ⊗ exp(δ_θ / 2)
//
// where exp is the unit-quaternion exponential of an axis-angle rotation
// vector. δ_θ is small for typical filter corrections; far from origin the
// retraction still preserves the manifold (always returns a unit quaternion).
template <class Scalar>
inline CraftStateT<Scalar> boxplus(const CraftStateT<Scalar>&    x,
                                   const CraftTangentT<Scalar>&  d) noexcept {
    CraftStateT<Scalar> out;
    out.position    = geom::Vec3<SceneFrame, Scalar>::from_raw(
                          x.position.raw() + d.position.raw());
    auto dq = geom::angle_axis_to_quat<Scalar>(d.orientation.raw());
    out.orientation = geom::Ori<SceneFrame, Scalar>{x.orientation.raw() * dq};
    out.vel_linear  = geom::Vec3<SceneFrame, Scalar>::from_raw(
                          x.vel_linear.raw() + d.vel_linear.raw());
    out.vel_angular = geom::Vec3<CraftFrame, Scalar>::from_raw(
                          x.vel_angular.raw() + d.vel_angular.raw());
    return out;
}

// Boxminus:  δ = a ⊟ b   so that   b ⊞ (a ⊟ b) == a.
//
// For orientation:  q_a = q_b ⊗ exp(δ_θ/2)  ⇒  δ_θ = 2·log(q_b⁻¹ ⊗ q_a).
// We compute log via the imaginary part: for a unit quaternion (w, v),
// log = atan2(|v|, w) · v/|v|; multiplying by 2 cancels the half-angle.
// Taylor-safe at q ≈ identity.
template <class Scalar>
inline CraftTangentT<Scalar> boxminus(const CraftStateT<Scalar>& a,
                                      const CraftStateT<Scalar>& b) noexcept {
    using std::atan2;
    using std::sqrt;
    CraftTangentT<Scalar> d;
    d.position    = geom::Vec3<SceneFrame, Scalar>::from_raw(
                        a.position.raw() - b.position.raw());

    auto dq = b.orientation.raw().conjugate() * a.orientation.raw();
    Eigen::Matrix<Scalar, 3, 1> v(dq.x(), dq.y(), dq.z());
    Scalar w = dq.w();
    Scalar v_norm_sq = v.squaredNorm();
    Eigen::Matrix<Scalar, 3, 1> theta;
    if (v_norm_sq > Scalar(0)) {
        Scalar v_norm = sqrt(v_norm_sq);
        // 2 * atan2(|v|, w) gives the rotation angle; v / |v| is the axis.
        Scalar k = Scalar(2) * atan2(v_norm, w) / v_norm;
        theta = k * v;
    } else {
        // Identity rotation: log = 0. First derivatives match (Taylor: 2v).
        theta = Scalar(2) * v;
    }
    d.orientation = geom::Vec3<SceneFrame, Scalar>::from_raw(theta);

    d.vel_linear  = geom::Vec3<SceneFrame, Scalar>::from_raw(
                        a.vel_linear.raw() - b.vel_linear.raw());
    d.vel_angular = geom::Vec3<CraftFrame, Scalar>::from_raw(
                        a.vel_angular.raw() - b.vel_angular.raw());
    return d;
}

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
    //
    // ORDERING NOTE: register all fields on the World *before* adding any
    // craft. A craft's parts capture field references during compute_params
    // (called downstream of add_craft); a field registered after that point
    // is silently invisible to those parts. WorldT::register_field traps
    // this with a runtime check.
    void add_craft(CraftT<Scalar>& c) {
        c.world_ = world_;
        c.scene_ = this;
        crafts_.push_back(&c);
        if (world_) world_->mark_crafts_added();
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
