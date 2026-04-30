#pragma once

#include <cassert>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include "articulated_part.hpp"
#include "composite_part.hpp"
#include "root_part.hpp"
#include "../geom/kinematic_link.hpp"
#include "../fields/field.hpp"

namespace manta {

class Scene;
class World;

// A craft is a self-contained vehicle: a root part tree plus its own rigid-
// body state. Templated on Scalar so the same authoring artifact can serve
// the sim path (Scalar = Real, driven by a Scene) and the estimator path
// (Scalar = ceres::Jet<...>, driven by an EKF via evaluate()).
//
// `using Craft = CraftT<Real>` below is what existing code uses; the Real
// instantiation is the one Scenes hold and update. Estimator-side use
// activates by instantiating CraftT<Jet> and calling its evaluate() hook
// directly — no Scene needed.
template <class Scalar = Real>
class CraftT {
public:
    explicit CraftT(std::string name) noexcept : name_(std::move(name)) {
        root_.craft_  = this;
        root_.parent_ = nullptr;
    }
    virtual ~CraftT() = default;

    using RootPartT_   = CompositePartT<Scalar>;   // RootPart is just a marker subclass
    using SceneToCraft = geom::KinematicLink<SceneFrame, CraftFrame, Scalar>;

    // The root composite. We use CompositePartT<Scalar> directly here to keep
    // the templated path uniform; the existing `RootPart` alias (= a tagged
    // subclass for Real) keeps working.
    PartT<Scalar>&       root_part()       noexcept { return root_; }
    const PartT<Scalar>& root_part() const noexcept { return root_; }

    // The legacy accessor returns the underlying composite as RootPart-like.
    // For Scalar=Real this is `RootPart&` via the alias chain.
    auto&       root()       noexcept { return root_; }
    const auto& root() const noexcept { return root_; }

    const std::string& name() const noexcept { return name_; }

    // Scene-frame rigid-body state.
    const SceneToCraft& scene_to_craft() const noexcept { return scene_to_craft_; }

    // Initial-condition setters.
    void set_position(const geom::Vec3<SceneFrame, Scalar>& p)  noexcept { scene_to_craft_.set_position(p); }
    void set_orientation(const geom::Ori<SceneFrame, Scalar>& q) noexcept { scene_to_craft_.set_orientation(q); }
    void set_vel_linear(const geom::Vec3<SceneFrame, Scalar>& v) noexcept { scene_to_craft_.set_vel_linear(v); }
    void set_vel_angular(const geom::Vec3<CraftFrame, Scalar>& w) noexcept { scene_to_craft_.set_vel_angular(w); }

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

    // Scene/World accessors are only meaningful when Scalar == Real; for the
    // estimator path (Jet), the craft isn't attached to a Scene/World.
    Scene&       scene();
    const Scene& scene() const;
    bool has_scene() const noexcept { return scene_ != nullptr; }

    World&       world();
    const World& world() const;
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

        const auto& w = root_.net_wrench();

        using EigenV = Eigen::Matrix<Scalar, 3, 1>;
        EigenV F_craft = w.force().raw();
        EigenV a_scene = scene_to_craft_.orientation().raw() * F_craft / m;
        scene_to_craft_.set_acc_linear(geom::Vec3<SceneFrame, Scalar>::from_raw(a_scene));

        const auto& I = root_.get_moi();
        EigenV omega = scene_to_craft_.vel_angular().raw();
        EigenV tau   = w.torque().raw();
        EigenV alpha = EigenV::Zero();
        Scalar det = I.raw().determinant();
        if (det > Scalar(1e-18)) {
            EigenV I_omega  = I.raw() * omega;
            EigenV tau_eff  = tau - omega.cross(I_omega);
            alpha           = I.raw().inverse() * tau_eff;
        }
        scene_to_craft_.set_acc_angular(geom::Vec3<CraftFrame, Scalar>::from_raw(alpha));
    }

    void integrate(Scalar dt) {
        if (dt <= Scalar(0)) return;
        integrate_joints_recurse(root_, dt);
        if (root_.get_mass() <= Scalar(0)) return;
        scene_to_craft_.update(dt);
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
        scene_to_craft_.set_position(geom::Vec3<SceneFrame, Scalar>::from_raw(x.template segment<3>(0)));
        Eigen::Quaternion<Scalar> q(x(3), x(4), x(5), x(6));
        q.normalize();
        scene_to_craft_.set_orientation(geom::Ori<SceneFrame, Scalar>{q});
        scene_to_craft_.set_vel_linear(geom::Vec3<SceneFrame, Scalar>::from_raw(x.template segment<3>(7)));
        scene_to_craft_.set_vel_angular(geom::Vec3<CraftFrame, Scalar>::from_raw(x.template segment<3>(10)));
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

        if (auto* a = dynamic_cast<ArticulatedPartT<Scalar>*>(&part)) {
            auto jl = a->joint_link();
            part.craft_to_part_ = craft_to_mount * jl;
            part.scene_to_part_ = scene_to_mount * jl;
        } else {
            part.craft_to_part_ = craft_to_mount;
            part.scene_to_part_ = scene_to_mount;
        }

        auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&part);
        if (!cp) return;
        for (auto& child : cp->children_) {
            kinematic_recurse(*child, part.craft_to_part_, part.scene_to_part_);
        }
    }

    static void sense_force_recurse(PartT<Scalar>& part) {
        part.wrench_accum_ = Wrench<PartFrame, Scalar>::zero();
        part.update();
        auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&part);
        if (!cp) return;
        for (auto& child : cp->children_) {
            sense_force_recurse(*child);
        }
    }

    static void integrate_joints_recurse(PartT<Scalar>& part, Scalar dt) {
        if (auto* a = dynamic_cast<ArticulatedPartT<Scalar>*>(&part)) {
            a->integrate_joint(dt);
        }
        if (auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&part)) {
            for (auto& child : cp->children_) {
                integrate_joints_recurse(*child, dt);
            }
        }
    }

    friend class Scene;

    std::string name_;
    RootPartT_  root_{"root"};
    SceneToCraft scene_to_craft_;
    std::unordered_map<std::type_index, fields::Field*> fields_;
    Scene*  scene_ = nullptr;
    World*  world_ = nullptr;
};

// Backwards-compat alias. The existing RootPart is used by Scalar=Real instances.
using Craft = CraftT<Real>;

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
inline Scene& CraftT<Scalar>::scene() {
    assert(scene_ && "Craft is not attached to a Scene");
    return *scene_;
}
template <class Scalar>
inline const Scene& CraftT<Scalar>::scene() const {
    assert(scene_ && "Craft is not attached to a Scene");
    return *scene_;
}

template <class Scalar>
inline World& CraftT<Scalar>::world() {
    assert(world_ && "Craft is not attached to a World");
    return *world_;
}
template <class Scalar>
inline const World& CraftT<Scalar>::world() const {
    assert(world_ && "Craft is not attached to a World");
    return *world_;
}

} // namespace manta
