#pragma once

#include <cassert>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include "articulated_part.hpp"
#include "composite_part.hpp"
#include "root_part.hpp"
#include "scene.hpp"   // sense_and_aggregate uses Scene::world_to_scene
#include "../geom/kinematic_link.hpp"
#include "../fields/field.hpp"

namespace manta {

template <class Scalar> class WorldT;

// A craft is a self-contained vehicle: a root part tree plus its own rigid-
// body state. Templated on Scalar so the same authoring artifact serves
// the sim path (Scalar = MFloat, driven by a Scene/World) and the estimator
// path (Scalar = ceres::Jet<...>, driven by a Jet-instantiated SceneT/WorldT
// from inside EKF::predict()).
//
// `using Craft = CraftT<MFloat>` below is what existing code uses. A Jet-
// instantiated WorldT<Jet> holds Jet-typed crafts via the same Scene API
// — there is no separate path for the estimator.
template <class Scalar = MFloat>
class CraftT {
public:
    using State = CraftStateT<Scalar>;

    explicit CraftT(std::string name) noexcept : name_(std::move(name)) {
        root_.craft_  = this;
        root_.parent_ = nullptr;
    }
    virtual ~CraftT();

    CraftT(const CraftT&) = delete;
    CraftT& operator=(const CraftT&) = delete;

    using RootPartT_   = CompositePartT<Scalar>;
    using SceneToCraft = geom::KinematicLink<SceneFrame, CraftFrame, Scalar>;

    // The root composite. Always returns the underlying CompositePartT so
    // callers can `craft.root().add<...>()`.
    CompositePartT<Scalar>&       root()       noexcept { return root_; }
    const CompositePartT<Scalar>& root() const noexcept { return root_; }

    const std::string& name() const noexcept { return name_; }

    // Scene-frame rigid-body state.
    const SceneToCraft& scene_to_craft() const noexcept { return scene_to_craft_; }

    // Canonical state setter. All other paths (Scene::add_craft with
    // an InitialState, the per-channel setters below, the EKF's
    // set_rigid_state) funnel through here.
    void set_state(const State& s) noexcept {
        scene_to_craft_.set_position(s.position);
        scene_to_craft_.set_orientation(s.orientation);
        scene_to_craft_.set_vel_linear(s.vel_linear);
        scene_to_craft_.set_vel_angular(s.vel_angular);
    }

    State get_state() const noexcept {
        return State{
            scene_to_craft_.position(),
            scene_to_craft_.orientation(),
            scene_to_craft_.vel_linear(),
            scene_to_craft_.vel_angular(),
        };
    }

    // Per-channel ergonomic setters. Equivalent to building a State,
    // mutating one field, and calling set_state.
    void set_position(const geom::Vec3<SceneFrame, Scalar>& p)   noexcept { scene_to_craft_.set_position(p); }
    void set_orientation(const geom::Ori<SceneFrame, Scalar>& q) noexcept { scene_to_craft_.set_orientation(q); }
    void set_vel_linear(const geom::Vec3<SceneFrame, Scalar>& v) noexcept { scene_to_craft_.set_vel_linear(v); }
    void set_vel_angular(const geom::Vec3<CraftFrame, Scalar>& w) noexcept { scene_to_craft_.set_vel_angular(w); }

    // Tune how often the integrator renormalizes the orientation quat to
    // suppress drift. n = 1 (default) renormalizes every step (safest).
    // Larger n saves a sqrt per skipped step at the cost of letting the
    // quaternion drift slightly off the unit sphere between renormalizations.
    // Reasonable values are 1–100; use the larger end only when this craft's
    // dynamics are well-conditioned and you've measured a meaningful sqrt
    // cost. The counter is per-craft so different crafts in the same world
    // can pick different trade-offs.
    void set_quat_renormalize_period(int n) noexcept {
        renorm_period_  = (n < 1) ? 1 : n;
        renorm_counter_ = 0;
    }
    int  quat_renormalize_period() const noexcept { return renorm_period_; }

    // Typed planet access via the craft's scene. Returns the scene's planet
    // dynamically-cast to `PlanetT*`, or `nullptr` if the scene has no
    // planet OR the planet isn't of the requested type. Mirrors how a Part
    // declares `requires_planet = Earth` in its descriptor and then calls
    // `craft().planet<Earth>()` from update().
    template <typename PlanetT>
    PlanetT* planet() noexcept {
        if (!scene_) return nullptr;
        return dynamic_cast<PlanetT*>(scene_->planet());
    }
    template <typename PlanetT>
    const PlanetT* planet() const noexcept {
        if (!scene_) return nullptr;
        return dynamic_cast<const PlanetT*>(scene_->planet());
    }

    // Field registry — keyed on typeid(FieldT).
    template <typename FieldT>
    void register_field(FieldT& f) {
        fields_[std::type_index(typeid(FieldT))] = &f;
    }
    template <typename FieldT>
    FieldT& field() {
        return *static_cast<FieldT*>(field_ptr(typeid(FieldT)));
    }
    template <typename FieldT>
    const FieldT& field() const {
        return *static_cast<const FieldT*>(field_ptr(typeid(FieldT)));
    }

    fields::Field* field_ptr(const std::type_info& ti) const;

    // Scene/World accessors. With templated Scene/World, a Jet-instantiated
    // craft IS attached to its Jet-instantiated parents — these accessors
    // work in both paths.
    SceneT<Scalar>&       scene();
    const SceneT<Scalar>& scene() const;
    bool has_scene() const noexcept { return scene_ != nullptr; }

    WorldT<Scalar>&       world();
    const WorldT<Scalar>& world() const;
    bool has_world() const noexcept { return world_ != nullptr; }

    // Three-phase update. Sim path (Scene-driven) calls these in barrier-
    // synchronized order across sibling crafts. Estimator path (EKF-driven)
    // can call them directly via evaluate().
    void kinematic_pass() {
        // Recompute mass/MOI/COM each tick (joint angles may have moved).
        root_.compute_params();

        auto craft_identity = geom::KinematicLink<CraftFrame, PartFrame, Scalar>::identity();
        root_.craft_to_part_ = craft_identity;
        root_.scene_to_part_ = scene_to_craft_.template reinterpret_to_frame<PartFrame>();

        for (auto& child : root_.children_) {
            kinematic_recurse(*child, craft_identity, root_.scene_to_part_);
        }
    }

    void sense_and_aggregate() {
        sense_force_recurse(root_);
        root_.aggregate_wrenches();

        Scalar m = root_.get_mass();
        if (m <= Scalar(0)) {
            scene_to_craft_.set_acc_linear(geom::Vec3<SceneFrame, Scalar>::zero());
            scene_to_craft_.set_acc_angular(geom::Vec3<CraftFrame, Scalar>::zero());
            return;
        }

        // The aggregated wrench is expressed at the root part's origin =
        // craft origin. The MOI from compute_params is about the craft COM.
        // Newton's second law gives a_COM = F/m; Euler's gives
        // I_COM·α = τ_COM − ω×(I_COM·ω) where τ_COM = τ_origin − r_COM × F.
        // We then transform back to the origin's acceleration for storage in
        // scene_to_craft_:
        //   a_origin = a_COM + α × r_OC + ω × (ω × r_OC),
        // where r_OC = origin − COM (in scene frame). Static-COM only;
        // d(r_COM)/dt for articulated mass-shifting bodies is not modeled.
        //
        // Non-inertial scene fictitious forces. When SceneFrame is non-inertial
        // (e.g. anchored to a rotating, accelerating Planet), Newton's law in
        // SceneFrame gains four pseudo-force terms per unit mass:
        //     a_translational = -a_S          (scene origin's linear accel)
        //     a_euler         = -α × r        (scene's angular accel × position)
        //     a_centrifugal   = -ω × (ω × r)
        //     a_coriolis      = -2 ω × v
        // where ω, α are the scene's angular velocity / acceleration relative
        // to the inertial WorldFrame (in scene-frame components), a_S is the
        // linear acceleration of the scene origin in WorldFrame (rotated into
        // scene frame), and r, v are the craft's scene-frame position and
        // velocity. We add the wrench-equivalent (m × a_pseudo) to F before
        // computing a_COM so the integrator (which advances state in
        // SceneFrame) produces inertially correct motion.
        //
        // All four terms are read from Scene's cached world_to_scene link,
        // whose composition automatically populates ω, α, and a_S from the
        // upstream Planet's rotation profile.
        const auto& w = root_.net_wrench();
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;
        EigenV F           = w.force().raw();
        EigenV tau_origin  = w.torque().raw();
        EigenV r_com_craft = root_.get_com().raw();    // COM in CraftFrame

        // Rotor gyroscopic torque on the body. The aggregated MOI handles
        // the I·ω part of L_body, but each joint also stores
        // I_axial·θ̇·axis of angular momentum that the body's Euler
        // equation must account for via −ω × h_rotor. Without this term,
        // a free-spinning rotor (reaction wheel, prop) wouldn't gyro-
        // stabilize its body — the body just rotates under applied
        // torques as if the rotor's stored momentum didn't exist.
        EigenV h_rotor = EigenV::Zero();
        accumulate_rotor_momentum(root_, h_rotor);

        if (scene_) {
            // omega_S, alpha_S = scene's angular velocity / acceleration
            // relative to WorldFrame, in scene-frame components.
            const auto& wts = scene_->world_to_scene();
            EigenV omega_S = wts.vel_angular().raw().template cast<Scalar>();
            EigenV alpha_S = wts.acc_angular().raw().template cast<Scalar>();
            // a_S lives in WorldFrame (the From frame of world_to_scene); rotate
            // into SceneFrame using R_WS^T = R_SW.
            Eigen::Matrix<Scalar, 3, 3> R_WS =
                wts.orientation().raw().toRotationMatrix().template cast<Scalar>();
            EigenV a_S_world = wts.acc_linear().raw().template cast<Scalar>();
            EigenV a_S_scene = R_WS.transpose() * a_S_world;

            EigenV r_scene = scene_to_craft_.position().raw();
            EigenV v_scene = scene_to_craft_.vel_linear().raw();

            EigenV a_pseudo_scene =
                  -a_S_scene
                  - alpha_S.cross(r_scene)
                  - omega_S.cross(omega_S.cross(r_scene))
                  - Scalar(2) * omega_S.cross(v_scene);

            EigenV f_extra_scene = m * a_pseudo_scene;
            EigenV f_extra_craft =
                scene_to_craft_.orientation().raw().conjugate() * f_extra_scene;
            F += f_extra_craft;
        }

        // Linear: a_COM in scene = R_(scene<-craft) · (F / m).
        EigenV a_com_scene =
            scene_to_craft_.orientation().raw() * (F / m);

        // Angular: solve I_COM · α = τ_COM − ω × (I_COM · ω) − ω × h_rotor
        // in CraftFrame. The h_rotor term gives gyroscopic precession from
        // spinning rotors stored in joints.
        const auto& I_com = root_.get_moi();
        EigenV omega_craft = scene_to_craft_.vel_angular().raw();
        EigenV tau_com     = tau_origin - r_com_craft.cross(F)
                           - omega_craft.cross(h_rotor);
        EigenV alpha_craft = EigenV::Zero();
        Scalar det = I_com.raw().determinant();
        if (det > Scalar(1e-18)) {
            EigenV I_omega = I_com.raw() * omega_craft;
            EigenV tau_eff = tau_com - omega_craft.cross(I_omega);
            alpha_craft    = I_com.raw().inverse() * tau_eff;
        }

        // Origin acceleration: a_COM + α × r_OC + ω × (ω × r_OC), all in scene.
        // omega_C / alpha_C = craft's angular velocity / acceleration (vs Scene)
        // expressed in scene-frame components.
        EigenV r_oc_craft = -r_com_craft;              // origin − COM, craft frame
        EigenV r_oc_scene = scene_to_craft_.orientation().raw() * r_oc_craft;
        EigenV alpha_C    = scene_to_craft_.orientation().raw() * alpha_craft;
        EigenV omega_C    = scene_to_craft_.orientation().raw() * omega_craft;
        EigenV a_origin_scene = a_com_scene
                              + alpha_C.cross(r_oc_scene)
                              + omega_C.cross(omega_C.cross(r_oc_scene));

        // Moving-COM correction. The r_OC formula above handles only the
        // rigid-at-this-instant centripetal/Euler terms, treating the
        // craft as if it were rigid at this tick's COM. When joints are
        // moving, the COM has body-frame velocity v_C_body and body-frame
        // acceleration a_C_body which contribute additional terms:
        //
        //   a_O = a_C + R·[α × r_OC + ω × (ω × r_OC) − 2·ω × v_C_body − a_C_body]
        //
        // We accumulate v_C_body and a_C_body by walking mass-bearing leaves
        // (their craft_to_part_ caches carry the kinematic-chain velocity and
        // acceleration components). For rigid parts both contributions are
        // zero; for parts in articulated chains the joint's rate and accel
        // propagate down via the KinematicLink composition.
        //
        // Crucially, this correction is added to a_O — NOT to the body's
        // wrench. Adding it to the wrench would pollute m_total·a_C = F_total
        // and make the system COM drift, violating conservation of linear
        // momentum for a closed craft with only internal forces.
        EigenV v_C_accum = EigenV::Zero();
        EigenV a_C_accum = EigenV::Zero();
        accumulate_leaf_com_motion(root_, v_C_accum, a_C_accum);
        if (m > Scalar(0)) {
            EigenV v_C_body = v_C_accum / m;
            EigenV a_C_body = a_C_accum / m;
            EigenV a_O_correction_body =
                -Scalar(2) * omega_craft.cross(v_C_body) - a_C_body;
            a_origin_scene +=
                scene_to_craft_.orientation().raw() * a_O_correction_body;
        }

        scene_to_craft_.set_acc_linear(
            geom::Vec3<SceneFrame, Scalar>::from_raw(a_origin_scene));
        scene_to_craft_.set_acc_angular(
            geom::Vec3<CraftFrame, Scalar>::from_raw(alpha_craft));
    }

    // ---- Integrator hooks (Verlet-ready split) ----
    //
    // The world tick is structured as:
    //   integrate_pre_aggregate(dt)
    //   kinematic_pass()
    //   sense_and_aggregate()        <-- caches a_k at the new state
    //   integrate_post_aggregate(dt)
    //
    // Today (explicit Euler): pre runs the full step using cached a_{k-1},
    // post is a no-op. The cache that pre consumes was populated either at
    // setup time (bootstrap) or by the previous tick's post-integrate
    // sense_and_aggregate, which leaves end-of-tick state self-consistent
    // (pos/q/v AND cached a both at x_k).
    //
    // Future symplectic upgrade (velocity Verlet, kick-drift-kick) drops in
    // here without touching call sites: pre becomes a half-kick + drift,
    // post becomes the second half-kick using the freshly aggregated a_k.
    // One force evaluation per tick, second-order accurate, energy-bounded
    // — orbits stay closed.
    void integrate_pre_aggregate(Scalar dt) {
        if (dt <= Scalar(0)) return;
        integrate_joints_recurse(root_, dt);
        if (root_.get_mass() <= Scalar(0)) return;
        // Per-craft renorm period: renormalize the orientation every
        // `renorm_period_` integration steps. Default 1 = every step.
        // Bumping this trades a tiny drift against sqrt cost; use
        // `set_quat_renormalize_period(n)` to opt in.
        const bool do_normalize =
            (renorm_period_ <= 1) ||
            ((++renorm_counter_ % renorm_period_) == 0);
        scene_to_craft_.update(dt, do_normalize);
    }

    void integrate_post_aggregate(Scalar /*dt*/) {
        // Explicit Euler placeholder. Verlet upgrade goes here:
        //   v_lin    += 0.5 * a_lin    * dt;
        //   v_angular += 0.5 * a_angular * dt;
    }

    // Combined (single-shot) integrate — kept for the EKF predict path
    // and for unit tests that drive a craft outside a full Scene.
    void integrate(Scalar dt) {
        integrate_pre_aggregate(dt);
        integrate_post_aggregate(dt);
    }

    // Convenience: full tick on this craft alone (no Scene barriers).
    void update(Scalar dt = Scalar(0)) {
        kinematic_pass();
        sense_and_aggregate();
        integrate(dt);
    }

    // ---- Estimator-side evaluation hook ----
    //
    // The EKF calls evaluate() with the EKF's current state vector and
    // an input vector (e.g. IMU readings already pushed via set_measurement
    // on sensor parts that the dynamics consumes). evaluate() sets the
    // craft's rigid-body state from `x_in`, runs one tick, and reads the
    // post-tick state out.
    //
    // State layout (13 entries): [px py pz | qw qx qy qz | vx vy vz | wx wy wz]
    // where (vx,vy,vz) are linear velocity in scene frame, (wx,wy,wz) are
    // angular velocity in craft frame. Quaternion is normalized after setting.
    static constexpr int kRigidStateDim = 13;
    using RigidState = Eigen::Matrix<Scalar, kRigidStateDim, 1>;

    void set_rigid_state(const RigidState& x) noexcept {
        // Quaternion always renormalized (including under autodiff): Eigen's
        // quat-vec product on non-unit q yields |q|²·R·v, which would leak a
        // spurious |q| sensitivity through the Jacobian.
        Eigen::Quaternion<Scalar> q(x(3), x(4), x(5), x(6));
        q.normalize();
        State s;
        s.position    = geom::Vec3<SceneFrame, Scalar>::from_raw(x.template segment<3>(0));
        s.orientation = geom::Ori<SceneFrame, Scalar>{q};
        s.vel_linear  = geom::Vec3<SceneFrame, Scalar>::from_raw(x.template segment<3>(7));
        s.vel_angular = geom::Vec3<CraftFrame, Scalar>::from_raw(x.template segment<3>(10));
        set_state(s);
    }

    RigidState get_rigid_state() const noexcept {
        RigidState x;
        x.template segment<3>(0) = scene_to_craft_.position().raw();
        const auto& q = scene_to_craft_.orientation().raw();
        x(3) = q.w(); x(4) = q.x(); x(5) = q.y(); x(6) = q.z();
        x.template segment<3>(7)  = scene_to_craft_.vel_linear().raw();
        x.template segment<3>(10) = scene_to_craft_.vel_angular().raw();
        return x;
    }

    RigidState evaluate(const RigidState& x_in, Scalar dt) {
        set_rigid_state(x_in);
        update(dt);
        return get_rigid_state();
    }

private:
    static void kinematic_recurse(PartT<Scalar>& part,
                                  const geom::KinematicLink<CraftFrame, PartFrame, Scalar>& parent_craft_link,
                                  const geom::KinematicLink<SceneFrame, PartFrame, Scalar>& parent_scene_link) {
        auto craft_to_mount = parent_craft_link * part.transform_;
        auto scene_to_mount = parent_scene_link * part.transform_;

        if (part.has_joint_link()) {
            auto jl = part.joint_link();
            part.craft_to_part_ = craft_to_mount * jl;
            part.scene_to_part_ = scene_to_mount * jl;
        } else {
            part.craft_to_part_ = craft_to_mount;
            part.scene_to_part_ = scene_to_mount;
        }

        auto* kids = part.children();
        if (!kids) return;
        for (auto& child : *kids) {
            kinematic_recurse(*child, part.craft_to_part_, part.scene_to_part_);
        }
    }

    static void sense_force_recurse(PartT<Scalar>& part) {
        part.wrench_accum_ = Wrench<PartFrame, Scalar>::zero();
        // Refresh state-dependent noise σ before the part runs its
        // update — so any `Vec3 + Noise` inside picks up the latest σ.
        part.update_noise_sigmas();
        part.update();
        auto* kids = part.children();
        if (!kids) return;
        for (auto& child : *kids) {
            sense_force_recurse(*child);
        }
    }

    static void integrate_joints_recurse(PartT<Scalar>& part, Scalar dt) {
        part.integrate_joint_state(dt);
        auto* kids = part.children();
        if (!kids) return;
        for (auto& child : *kids) {
            integrate_joints_recurse(*child, dt);
        }
    }

    // Walk mass-bearing leaves accumulating their contribution to the
    // system's body-frame COM velocity and acceleration:
    //   v_C_body = Σ m_i · v_com_in_body / m_total
    //   a_C_body = Σ m_i · a_com_in_body / m_total
    // Each leaf's craft_to_part_ kinematic cache holds the body-frame
    // velocity/acceleration of the part's origin (computed by the
    // KinematicLink chain from any articulated joints upstream). Combined
    // with the leaf's own com_in_part offset and angular vel/accel of the
    // part frame, this gives the leaf COM's full body-frame motion:
    //
    //   v_com_in_body = v_origin + ω_part_in_body × R_part_to_body·com_in_part
    //   a_com_in_body = a_origin + α_part_in_body × R·com
    //                            + ω_part_in_body × (ω_part_in_body × R·com)
    //
    // Used by sense_and_aggregate to apply the moving-COM correction to
    // the body origin's inertial acceleration. Composite parts (which
    // hold aggregate mass but no actual particle) are skipped — we only
    // visit leaves (children() == nullptr or empty).
    static void accumulate_leaf_com_motion(
            const PartT<Scalar>& part,
            Eigen::Matrix<Scalar, 3, 1>& v_accum,
            Eigen::Matrix<Scalar, 3, 1>& a_accum) {
        auto* kids = part.children();
        const bool is_leaf = (kids == nullptr || kids->empty());

        if (is_leaf) {
            const Scalar m_i = part.get_mass();
            if (m_i > Scalar(0)) {
                using EigenV = Eigen::Matrix<Scalar, 3, 1>;
                const auto& link = part.craft_to_part();
                EigenV com_in_part = part.get_com().raw();
                EigenV R_com       = link.orientation().raw() * com_in_part;
                EigenV omega_in_part = link.vel_angular().raw();
                EigenV alpha_in_part = link.acc_angular().raw();
                EigenV omega_in_body = link.orientation().raw() * omega_in_part;
                EigenV alpha_in_body = link.orientation().raw() * alpha_in_part;
                EigenV v_com = link.vel_linear().raw()
                             + omega_in_body.cross(R_com);
                EigenV a_com = link.acc_linear().raw()
                             + alpha_in_body.cross(R_com)
                             + omega_in_body.cross(omega_in_body.cross(R_com));
                v_accum += m_i * v_com;
                a_accum += m_i * a_com;
            }
            return;
        }

        for (const auto& child : *kids) {
            accumulate_leaf_com_motion(*child, v_accum, a_accum);
        }
    }

    // Walk the part tree summing each articulated joint's stored angular
    // momentum (I_axial · rate · axis), rotated into CraftFrame. Used by
    // sense_and_aggregate to inject the gyroscopic torque correction
    // −ω_craft × h_rotor on the body's Euler equation — what a spinning
    // rotor exerts on its mount when the body rotates.
    //
    // Exact for axially-symmetric rotors with axis-fixed-in-body joints
    // (every Motor in stock parts). For non-axisymmetric rotors with
    // mass-shifting subtrees, the time-varying inertia tensor introduces
    // additional terms not captured here — but the I·ω piece handled by
    // the body Euler already accounts for the bulk of those.
    static void accumulate_rotor_momentum(const PartT<Scalar>& part,
                                          Eigen::Matrix<Scalar, 3, 1>& accum) {
        auto h_part = part.rotor_angular_momentum_part_frame();
        if (h_part.raw().squaredNorm() > Scalar(0)) {
            accum += part.craft_to_part().orientation().raw() * h_part.raw();
        }
        auto* kids = part.children();
        if (!kids) return;
        for (const auto& child : *kids) {
            accumulate_rotor_momentum(*child, accum);
        }
    }

    friend class SceneT<Scalar>;

    std::string name_;
    RootPartT_  root_{"root"};
    SceneToCraft scene_to_craft_;
    std::unordered_map<std::type_index, fields::Field*> fields_;
    SceneT<Scalar>*  scene_ = nullptr;
    WorldT<Scalar>*  world_ = nullptr;
    // Renormalize the craft's orientation quaternion every `renorm_period_`
    // integration steps. Default 1 = every step (matches pre-tunable
    // behavior). Tunable via `set_quat_renormalize_period`.
    int renorm_period_  = 1;
    int renorm_counter_ = 0;
};

// Backwards-compat alias. The existing RootPart is used by Scalar=MFloat instances.
using Craft = CraftT<MFloat>;

} // namespace manta

// ---- Inline definitions of cross-class methods ----
// These need both PartT and CraftT (and, for field_ptr's world fallback,
// World) to be fully defined. Including world.hpp here is safe: world.hpp
// only forward-references CraftT (via scene.hpp).
#include "world.hpp"

namespace manta {

template <class Scalar>
inline CraftT<Scalar>& PartT<Scalar>::craft() {
    assert(craft_ != nullptr && "PartT::craft(): part is not attached to a Craft");
    return *craft_;
}

template <class Scalar>
inline const CraftT<Scalar>& PartT<Scalar>::craft() const {
    assert(craft_ != nullptr && "PartT::craft(): part is not attached to a Craft");
    return *craft_;
}

template <class Scalar>
inline fields::Field* PartT<Scalar>::field_ptr(const std::type_info& ti) const {
    assert(craft_ != nullptr && "PartT::field(): part is not attached to a Craft");
    return craft_->field_ptr(ti);
}

template <class Scalar>
inline fields::Field* CraftT<Scalar>::field_ptr(const std::type_info& ti) const {
    auto it = fields_.find(std::type_index(ti));
    if (it != fields_.end()) return it->second;
    if (world_) return world_->field_ptr(ti);
    return nullptr;
}

template <class Scalar>
inline SceneT<Scalar>& CraftT<Scalar>::scene() {
    assert(scene_ && "Craft is not attached to a Scene");
    return *scene_;
}
template <class Scalar>
inline const SceneT<Scalar>& CraftT<Scalar>::scene() const {
    assert(scene_ && "Craft is not attached to a Scene");
    return *scene_;
}

template <class Scalar>
inline WorldT<Scalar>& CraftT<Scalar>::world() {
    assert(world_ && "Craft is not attached to a World");
    return *world_;
}
template <class Scalar>
inline const WorldT<Scalar>& CraftT<Scalar>::world() const {
    assert(world_ && "Craft is not attached to a World");
    return *world_;
}

// Defined here (not inline in the class body) because it dereferences
// SceneT, which forward-declares CraftT. By this point in the
// translation unit both are complete.
template <class Scalar>
inline CraftT<Scalar>::~CraftT() {
    if (scene_) scene_->remove_craft(*this);
}

} // namespace manta
