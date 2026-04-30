#pragma once

#include "../../core/articulated_part.hpp"

namespace manta::parts {

// A 1-DOF motor — a revolute joint with an actuator that can apply a
// commanded torque about the joint axis.
//
// Operating modes:
//   - Passive   : no actuator torque. Joint accelerates under whatever axial
//                 component of the child wrench it sees, plus optional
//                 viscous damping.
//   - Saturating: actuator drives the joint with a commanded torque, clamped
//                 to ±stall_torque. The same torque (signed) also reaches
//                 the parent as the motor's reaction (Newton's third).
//
// In both modes the perpendicular component of the child torque, plus the
// full child force, is transmitted to the parent as a constraint reaction.
class Motor : public ArticulatedPart {
public:
    enum class Mode { Passive, Saturating };

    explicit Motor(std::string name,
                   geom::Vec3<PartFrame> axis,
                   Real stall_torque = Real(0),
                   Real damping      = Real(0));

    void set_torque(Real tau_cmd) noexcept {
        torque_cmd_ = tau_cmd;
        mode_ = Mode::Saturating;
    }
    void set_passive() noexcept { mode_ = Mode::Passive; }

    Mode mode()         const noexcept { return mode_;        }
    Real torque_cmd()   const noexcept { return torque_cmd_;  }
    Real stall_torque() const noexcept { return stall_torque_; }
    Real damping()      const noexcept { return damping_;     }

    void resolve(const Wrench<PartFrame>& child_total,
                 Wrench<PartFrame>&       parent_out,
                 Real&                    joint_accel_out) override;

private:
    // Cached during resolve — used only for reporting/tests.
    Mode mode_         = Mode::Passive;
    Real torque_cmd_   = Real(0);
    Real stall_torque_ = Real(0);
    Real damping_      = Real(0);
};

} // namespace manta::parts
