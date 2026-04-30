#pragma once

#include "composite_part.hpp"
#include "../geom/kinematic_link.hpp"

namespace manta {

// A composite part with a 1-DOF revolute joint between its parent's mount
// frame (where transform_ leaves us) and its own joint-output frame.
//
// Templated on Scalar — the non-templated `ArticulatedPart` alias below
// (= ArticulatedPartT<Real>) is what existing user code uses.
template <class Scalar = Real>
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
    geom::KinematicLink<PartFrame, PartFrame, Scalar> joint_link() const noexcept {
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

    // Subclass hook (Motor, FreeHinge, ...).
    virtual void resolve(const Wrench<PartFrame, Scalar>& child_total,
                         Wrench<PartFrame, Scalar>&       parent_out,
                         Scalar&                          joint_accel_out) = 0;

    // aggregate_wrenches override.
    void aggregate_wrenches() override {
        using V      = geom::Vec3<PartFrame, Scalar>;
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;

        this->wrench_accum_ = Wrench<PartFrame, Scalar>::zero();
        for (auto& child_ptr : this->children_) {
            PartT<Scalar>& child = *child_ptr;
            if (auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&child)) {
                cp->aggregate_wrenches();
            }
            Wrench<PartFrame, Scalar> cw = child.drain_wrench();
            V force_p  = child.transform_.rotate(cw.force());
            V torque_p = child.transform_.rotate(cw.torque())
                       + child.transform_.position().cross(force_p);
            this->wrench_accum_ += Wrench<PartFrame, Scalar>{force_p, torque_p};
        }

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

    // compute_params override.
    void compute_params() override {
        using V      = geom::Vec3<PartFrame, Scalar>;
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;
        using EigenM = Eigen::Matrix<Scalar, 3, 3>;

        CompositePartT<Scalar>::compute_params();

        // MOI about the joint origin in the joint-output frame.
        EigenV com_jo = this->com_.raw();
        EigenM I_origin = this->moi_.raw()
                        + this->mass_ * (com_jo.squaredNorm() * EigenM::Identity()
                                         - com_jo * com_jo.transpose());
        moi_origin_jo_ = geom::Mat3<PartFrame, PartFrame, Scalar>{I_origin};

        // Rotate com/moi from joint-output frame to mount frame.
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

    const geom::Mat3<PartFrame, PartFrame, Scalar>& moi_about_origin_jo() const noexcept {
        return moi_origin_jo_;
    }

protected:
    geom::Vec3<PartFrame, Scalar> axis_;
    Scalar angle_ = Scalar(0);
    Scalar rate_  = Scalar(0);
    Scalar accel_ = Scalar(0);

    geom::Mat3<PartFrame, PartFrame, Scalar> moi_origin_jo_ =
        geom::Mat3<PartFrame, PartFrame, Scalar>::zero();
};

// Backwards-compat alias.
using ArticulatedPart = ArticulatedPartT<Real>;

} // namespace manta
