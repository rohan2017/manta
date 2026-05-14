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
    // current angle/rate/accel. The kinematic pass composes this with the
    // part's static mount transform to get parent_link → joint output.
    //
    // acc_angular carries the joint's angular acceleration (accel_ · axis,
    // using last tick's accel since the current tick's accel_ is set inside
    // aggregate_wrenches AFTER kinematic_pass). This propagates through the
    // KinematicLink composition so downstream parts (Mass children, IMUs)
    // pick up the joint's angular acceleration in their acc_linear /
    // acc_angular caches. Same 1-tick-lag tradeoff as the rest of the
    // explicit-Euler joint integrator.
    geom::KinematicLink<PartFrame, PartFrame, Scalar>
    joint_link() const noexcept override {
        using KL = geom::KinematicLink<PartFrame, PartFrame, Scalar>;
        using V  = geom::Vec3<PartFrame, Scalar>;

        const auto orientation =
            geom::Ori<PartFrame, Scalar>::from_axis_angle(axis_, angle_);

        // vel_angular and acc_angular live in the To frame; since the joint
        // rotates about its own axis at rate θ̇ with acceleration θ̈, the
        // vectors are θ̇·axis and θ̈·axis respectively (axis is invariant).
        const auto omega = V::from_raw(rate_  * axis_.raw());
        const auto alpha = V::from_raw(accel_ * axis_.raw());
        return KL{V::zero(), orientation, V::zero(), omega,
                  V::zero(), alpha};
    }
    bool has_joint_link() const noexcept override { return true; }

    // Subclass hook (Motor, FreeHinge, ...). Inputs and outputs are all
    // expressed in MOUNT-frame components (the parent's side of the
    // joint, where the parent's compute_params rolled up COM/MOI).
    // `omega_mount` is the absolute angular velocity of the mount frame
    // (relative to scene), in mount components — used by subclasses that
    // want to inject Coriolis/centrifugal coupling on the joint.
    virtual void resolve(const Wrench<PartFrame, Scalar>& child_total,
                         const geom::Vec3<PartFrame, Scalar>& omega_mount,
                         Wrench<PartFrame, Scalar>&       parent_out,
                         Scalar&                          joint_accel_out) = 0;

    // aggregate_wrenches override. Composite gathers child wrenches into
    // wrench_accum_ in the joint-output frame; we rotate to mount frame,
    // compute the mount-frame absolute angular velocity, and call
    // resolve() with everything in mount frame. The resolve's output
    // parent_out is already in mount frame so it goes straight into
    // wrench_accum_ for the upstream walk.
    void aggregate_wrenches() override {
        using V      = geom::Vec3<PartFrame, Scalar>;
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;

        CompositePartT<Scalar>::aggregate_wrenches();

        // q_jo_to_mount: rotate vectors from joint-output frame to mount frame.
        auto q_jo_to_mount = Eigen::Quaternion<Scalar>{
            Eigen::AngleAxis<Scalar>{angle_, axis_.raw()}};

        // Rotate child_total (in joint-output) into mount frame.
        const auto& jo = this->wrench_accum_;
        EigenV F_child_mount = q_jo_to_mount * jo.force().raw();
        EigenV T_child_mount = q_jo_to_mount * jo.torque().raw();
        const Wrench<PartFrame, Scalar> child_total_mount{
            V::from_raw(F_child_mount), V::from_raw(T_child_mount)
        };

        // Mount-frame absolute angular velocity. scene_to_part_.vel_angular()
        // is ω of the joint-output frame relative to scene, in joint-output
        // components. ω_mount = ω_part − rate·axis (axis is invariant under
        // the joint rotation), then rotate to mount components.
        EigenV omega_part_jo  = this->scene_to_part_.vel_angular().raw();
        EigenV omega_mount_jo = omega_part_jo - rate_ * axis_.raw();
        EigenV omega_mount_mount = q_jo_to_mount * omega_mount_jo;

        Wrench<PartFrame, Scalar> parent_reaction =
            Wrench<PartFrame, Scalar>::zero();
        Scalar joint_accel = Scalar(0);
        resolve(child_total_mount,
                V::from_raw(omega_mount_mount),
                parent_reaction,
                joint_accel);
        accel_ = joint_accel;

        // parent_reaction is already in mount frame; pass it up unmodified.
        this->wrench_accum_ = parent_reaction;
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

    // Angular momentum from the joint rotation: I_axial · rate · axis.
    // I_axial is the rotor subtree's moment of inertia about the joint
    // axis (computed via axis·I_com·axis + m·d_perp² parallel-axis shift,
    // matching Motor::resolve). The Craft sums (rotated to CraftFrame)
    // these across the tree to produce h_rotor for the body's
    // gyroscopic torque correction.
    geom::Vec3<PartFrame, Scalar>
    rotor_angular_momentum_part_frame() const noexcept override {
        using V      = geom::Vec3<PartFrame, Scalar>;
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;
        EigenV axis = axis_.raw();
        EigenV com  = this->com_.raw();
        Scalar d_perp_sq = com.squaredNorm()
                         - (com.dot(axis)) * (com.dot(axis));
        Scalar I_axial = axis.dot(this->moi_.raw() * axis)
                       + this->mass_ * d_perp_sq;
        return V::from_raw(I_axial * rate_ * axis);
    }

protected:
    geom::Vec3<PartFrame, Scalar> axis_;
    Scalar angle_ = Scalar(0);
    Scalar rate_  = Scalar(0);
    Scalar accel_ = Scalar(0);
};

// Backwards-compat alias.
using ArticulatedPart = ArticulatedPartT<MFloat>;

} // namespace manta
