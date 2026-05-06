#pragma once

#include "composite_part.hpp"
#include "../geom/kinematic_link.hpp"

namespace manta {

// A composite part with a 1-DOF revolute joint between its parent's mount
// frame (where transform_ leaves us) and its own joint-output frame.
//
// Templated on Scalar — the non-templated `ArticulatedPart` alias below
// (= ArticulatedPartT<MFloat>) is what existing user code uses.
template <class Scalar = MFloat>
class ArticulatedPartT : public CompositePartT<Scalar> {
public:
    explicit ArticulatedPartT(std::string name,
                              geom::Vec3<PartFrame, Scalar> axis,
                              Scalar initial_angle = Scalar(0),
                              Scalar initial_rate  = Scalar(0)) noexcept
        : CompositePartT<Scalar>(std::move(name)),
          axis_(geom::Vec3<PartFrame, Scalar>::from_raw(axis.raw().normalized())),
          angle_(initial_angle),
          rate_(initial_rate) {}

    const geom::Vec3<PartFrame, Scalar>& joint_axis() const noexcept { return axis_; }

    Scalar angle() const noexcept { return angle_; }
    Scalar rate()  const noexcept { return rate_;  }
    Scalar accel() const noexcept { return accel_; }

    void set_angle(Scalar a) noexcept { angle_ = a; }
    void set_rate (Scalar r) noexcept { rate_  = r; }

    // Build the kinematic link from mount frame to joint-output frame using
    // current angle/rate. The kinematic pass composes this with the part's
    // static mount transform to get parent_link → joint output.
    geom::KinematicLink<PartFrame, PartFrame, Scalar>
    joint_link() const noexcept override {
        using KL = geom::KinematicLink<PartFrame, PartFrame, Scalar>;
        using V  = geom::Vec3<PartFrame, Scalar>;

        const auto orientation =
            geom::Ori<PartFrame, Scalar>::from_axis_angle(axis_, angle_);

        // vel_angular lives in the To frame; since the joint rotates about its
        // own axis at rate θ̇, the angular velocity vector is θ̇*axis in both
        // frames component-wise.
        const auto omega = V::from_raw(rate_ * axis_.raw());
        return KL{V::zero(), orientation, V::zero(), omega,
                  V::zero(), V::zero()};
    }
    bool has_joint_link() const noexcept override { return true; }

    // Subclass hook (Motor, FreeHinge, ...).
    virtual void resolve(const Wrench<PartFrame, Scalar>& child_total,
                         Wrench<PartFrame, Scalar>&       parent_out,
                         Scalar&                          joint_accel_out) = 0;

    // aggregate_wrenches override. Composite gathers child wrenches into
    // wrench_accum_; we then call resolve() to split that into a child
    // total (consumed by the joint) and a parent reaction expressed in
    // the joint-output frame, finally rotated into the mount frame.
    void aggregate_wrenches() override {
        using V      = geom::Vec3<PartFrame, Scalar>;
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;

        CompositePartT<Scalar>::aggregate_wrenches();
        const Wrench<PartFrame, Scalar> child_total = this->wrench_accum_;

        Wrench<PartFrame, Scalar> parent_reaction =
            Wrench<PartFrame, Scalar>::zero();
        Scalar joint_accel = Scalar(0);
        resolve(child_total, parent_reaction, joint_accel);
        accel_ = joint_accel;

        // Express parent_reaction in the mount frame: jo→mount is R(axis, +angle).
        auto q_jo_to_mount = Eigen::Quaternion<Scalar>{
            Eigen::AngleAxis<Scalar>{angle_, axis_.raw()}};
        EigenV F_mount = q_jo_to_mount * parent_reaction.force().raw();
        EigenV T_mount = q_jo_to_mount * parent_reaction.torque().raw();
        this->wrench_accum_ = Wrench<PartFrame, Scalar>{
            V::from_raw(F_mount), V::from_raw(T_mount)
        };
    }

    // compute_params override. Composite leaves COM and MOI in the joint-
    // output frame about the composite COM; we rotate both into the mount
    // frame so the parent assembly sees them as if attached statically.
    void compute_params() override {
        using V      = geom::Vec3<PartFrame, Scalar>;
        using EigenM = Eigen::Matrix<Scalar, 3, 3>;

        CompositePartT<Scalar>::compute_params();

        auto q_jo_to_mount = Eigen::Quaternion<Scalar>{
            Eigen::AngleAxis<Scalar>{angle_, axis_.raw()}};
        EigenM R = q_jo_to_mount.toRotationMatrix();

        this->com_ = V::from_raw(R * this->com_.raw());
        this->moi_ = geom::Mat3<PartFrame, PartFrame, Scalar>{
            R * this->moi_.raw() * R.transpose()};
    }

    virtual void integrate_joint(Scalar dt) noexcept {
        if (dt <= Scalar(0)) return;
        angle_ += rate_ * dt + Scalar(0.5) * accel_ * dt * dt;
        rate_  += accel_ * dt;
    }

    // PartT hook used by Craft's integrator pass.
    void integrate_joint_state(Scalar dt) noexcept override { integrate_joint(dt); }

protected:
    geom::Vec3<PartFrame, Scalar> axis_;
    Scalar angle_ = Scalar(0);
    Scalar rate_  = Scalar(0);
    Scalar accel_ = Scalar(0);
};

// Backwards-compat alias.
using ArticulatedPart = ArticulatedPartT<MFloat>;

} // namespace manta
