#include "manta/parts/articulation/motor.hpp"
#include "manta/core/craft.hpp"

#include <algorithm>

namespace manta::parts {

Motor::Motor(std::string name,
             geom::Vec3<PartFrame> axis,
             Real stall_torque,
             Real damping)
    : ArticulatedPart(std::move(name), axis),
      stall_torque_(stall_torque),
      damping_(damping) {}

void Motor::resolve(const Wrench<PartFrame>& child_total,
                    Wrench<PartFrame>&       parent_out,
                    Real&                    joint_accel_out) {
    using V      = geom::Vec3<PartFrame>;
    using EigenV = Eigen::Matrix<Real, 3, 1>;

    // Decompose child torque into axial (along joint axis) and perpendicular
    // (transmitted as constraint reaction to the parent).
    EigenV axis = axis_.raw();           // unit
    EigenV tau  = child_total.torque().raw();
    Real   tau_axial = tau.dot(axis);
    EigenV tau_perp  = tau - tau_axial * axis;

    // Actuator torque about the axis. In Saturating mode the actuator pushes
    // the joint with τ_cmd (clamped to ±stall); in Passive mode there is no
    // actuator torque and the joint reacts only to tau_axial + damping.
    Real tau_actuator = Real(0);
    if (mode_ == Mode::Saturating && stall_torque_ > Real(0)) {
        tau_actuator = std::clamp(torque_cmd_, -stall_torque_, stall_torque_);
    } else if (mode_ == Mode::Saturating) {
        tau_actuator = torque_cmd_;          // unbounded if stall not configured
    }

    // Optional viscous damping (resists rate; takes joint torque from joint).
    Real tau_damping = -damping_ * rate_;

    // Net axial torque on the joint output side: child's axial torque +
    // actuator + damping. The joint integrator advances using this.
    Real tau_net_axial = tau_axial + tau_actuator + tau_damping;

    // Axial moment of inertia about the JOINT ORIGIN (not COM) about the
    // joint axis. ArticulatedPart::compute_params has populated this in the
    // joint-output frame.
    Real I_axial = axis.dot(moi_about_origin_jo().raw() * axis);
    if (I_axial < Real(1e-18)) {
        // Subtree has no axial inertia (or compute_params not called).
        // Treat joint as locked: no acceleration, full torque to parent.
        joint_accel_out = Real(0);
        parent_out = Wrench<PartFrame>{
            child_total.force(),
            V::from_raw(tau)
        };
        return;
    }

    joint_accel_out = tau_net_axial / I_axial;

    // Parent reaction wrench (in joint-output frame):
    //   force: full child force (joint is rigid in translation)
    //   torque: perpendicular component (constraint reaction) + actuator
    //           reaction along the axis (Newton's third — the actuator
    //           that drives the joint with +τ_actuator pushes the parent
    //           with −τ_actuator).
    EigenV tau_parent = tau_perp + (-tau_actuator) * axis;
    parent_out = Wrench<PartFrame>{
        child_total.force(),
        V::from_raw(tau_parent)
    };
}

} // namespace manta::parts
