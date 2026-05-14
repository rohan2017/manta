#include "manta/parts/articulation/motor.hpp"
#include "manta/core/craft.hpp"

#include <algorithm>

namespace manta::parts {

Motor::Motor(std::string name,
             geom::Vec3<PartFrame> axis,
             MFloat stall_torque,
             MFloat damping)
    : ArticulatedPart(std::move(name), axis),
      stall_torque_(stall_torque),
      damping_(damping) {}

// All inputs and outputs are in MOUNT-frame components — see
// ArticulatedPart::aggregate_wrenches, which rotates child_total to mount
// before calling resolve and leaves the returned parent_out as-is.
void Motor::resolve(const Wrench<PartFrame>& child_total,
                    const geom::Vec3<PartFrame>& omega_mount,
                    Wrench<PartFrame>&       parent_out,
                    MFloat&                    joint_accel_out) {
    using V      = geom::Vec3<PartFrame>;
    using EigenV = Eigen::Matrix<MFloat, 3, 1>;
    using EigenM = Eigen::Matrix<MFloat, 3, 3>;

    // Decompose child torque into axial (along joint axis) and perpendicular
    // (transmitted as constraint reaction to the parent).
    EigenV axis = axis_.raw();           // unit; invariant under joint rotation
    EigenV tau  = child_total.torque().raw();
    MFloat   tau_axial = tau.dot(axis);
    EigenV tau_perp  = tau - tau_axial * axis;

    // Actuator torque about the axis. In Saturating mode the actuator pushes
    // the joint with τ_cmd (clamped to ±stall); in Passive mode there is no
    // actuator torque and the joint reacts only to tau_axial + damping.
    MFloat tau_actuator = MFloat(0);
    if (mode_ == Mode::Saturating && stall_torque_ > MFloat(0)) {
        tau_actuator = std::clamp(torque_cmd_, -stall_torque_, stall_torque_);
    } else if (mode_ == Mode::Saturating) {
        tau_actuator = torque_cmd_;          // unbounded if stall not configured
    }

    // Optional viscous damping (resists rate; takes joint torque from joint).
    MFloat tau_damping = -damping_ * rate_;

    // Axial moment of inertia about the JOINT ORIGIN, projected on the
    // joint axis. Built from the standard COM-based MOI via parallel axis.
    // `get_com()` and `get_moi()` are in MOUNT frame (post-rotation in
    // ArticulatedPart::compute_params); axis is invariant.
    EigenV com         = get_com().raw();
    MFloat d_perp_sq   = com.squaredNorm() - (com.dot(axis)) * (com.dot(axis));
    EigenM I_com_mount = get_moi().raw();
    MFloat I_axial     = axis.dot(I_com_mount * axis) + get_mass() * d_perp_sq;

    if (I_axial < MFloat(1e-18)) {
        // Subtree has no axial inertia (or compute_params not called).
        // Treat joint as locked: no acceleration, full torque to parent.
        joint_accel_out = MFloat(0);
        parent_out = Wrench<PartFrame>{
            child_total.force(),
            V::from_raw(tau)
        };
        return;
    }

    // Full inertia tensor about joint origin (parallel-axis lift from COM):
    //   I_joint = I_com + m · (|r|²·I − r·r^T)
    // Used for the Coriolis term below.
    EigenM I_joint_mount = I_com_mount
                         + get_mass() * (com.squaredNorm() * EigenM::Identity()
                                       - com * com.transpose());

    // Rotor absolute angular velocity in mount components:
    //   ω_rotor = ω_mount + θ̇ · axis.
    EigenV omega_m     = omega_mount.raw();
    EigenV omega_rotor = omega_m + rate_ * axis;

    // Coriolis joint torque from base rotation through the rotor's inertia:
    //   τ_corio_axial = −[ω_rotor × (I_joint · ω_rotor)] · axis
    // Vanishes identically for fully axisymmetric rotors (where I·v = I_axial·v
    // for v along axis); non-zero for asymmetric blade/disk geometries. The
    // α_mount cross-coupling term (−(I·α_mount)·axis) is omitted — handling
    // it requires either iteration or a small joint+root linear solve, both
    // of which are deferred per the current design.
    MFloat tau_corio = -omega_rotor.cross(I_joint_mount * omega_rotor).dot(axis);

    // Net axial torque on the joint output side.
    MFloat tau_net_axial = tau_axial + tau_actuator + tau_damping + tau_corio;
    joint_accel_out = tau_net_axial / I_axial;

    // Moving-COM reaction is intentionally NOT injected as a force here.
    //
    // For a free-floating craft with internal joint motion, conservation of
    // linear momentum says the SYSTEM COM doesn't drift — the body origin
    // wobbles around it. Any force added to parent_out flows into the
    // body's Newton equation as m_total·a_C = F_total, which would make
    // a_C non-zero and the COM drift.
    //
    // The correct fix lives at the body-origin level:
    //   a_O = a_C + R·[α × r_OC + ω × (ω × r_OC) − 2·ω × v_C_body − a_C_body]
    // Today's Craft::sense_and_aggregate handles α × r_OC and ω × (ω × r_OC)
    // using the current-tick r_C (which IS re-sampled each tick from joint
    // angles), and applies the moving-COM correction (−2·ω × v_C_body −
    // a_C_body) directly to a_origin_scene via a leaf walk.
    //
    // Parent reaction wrench (in mount frame):
    //   force: pass-through of the child subtree's external force only.
    //   torque: perpendicular component (constraint reaction) + actuator
    //           reaction along the axis (Newton's third).
    EigenV tau_parent = tau_perp + (-tau_actuator) * axis;
    parent_out = Wrench<PartFrame>{
        child_total.force(),
        V::from_raw(tau_parent)
    };
}

} // namespace manta::parts
