#pragma once

#include <cmath>
#include "thruster.hpp"

namespace manta::parts {

// A thruster mounted on a 2-axis gimbal. Same conventions as before:
// pitch about part-frame y, yaw about part-frame x.
//
// Templated on Scalar; trig calls use unqualified `sin`/`cos` so ADL picks
// the right overload (`std::sin/cos` for float/double, `ceres::sin/cos` for
// Jet).
template <class Scalar = Real>
class GimbaledThrusterT : public ThrusterT<Scalar> {
public:
    explicit GimbaledThrusterT(std::string name,
                               Scalar max_thrust,
                               Scalar max_angle_rad = Scalar(0.15f),
                               geom::Vec3<PartFrame, Scalar> base_direction =
                                   {Scalar(0), Scalar(0), Scalar(1)})
        : ThrusterT<Scalar>(std::move(name), max_thrust, base_direction)
        , base_direction_(base_direction)
        , max_angle_(max_angle_rad) {}

    void set_gimbal(Scalar pitch, Scalar yaw) noexcept {
        pitch_ = clamp_angle(pitch);
        yaw_   = clamp_angle(yaw);
    }

    Scalar pitch() const noexcept { return pitch_; }
    Scalar yaw()   const noexcept { return yaw_; }
    Scalar max_angle() const noexcept { return max_angle_; }

    geom::Vec3<PartFrame, Scalar> deflected_direction() const noexcept {
        using std::cos;
        using std::sin;
        const Scalar cy = cos(yaw_),   sy = sin(yaw_);
        const Scalar cp = cos(pitch_), sp = sin(pitch_);
        const auto& v = base_direction_.raw();
        Eigen::Matrix<Scalar, 3, 1> v1{v(0), cy * v(1) - sy * v(2), sy * v(1) + cy * v(2)};
        Eigen::Matrix<Scalar, 3, 1> v2{cp * v1(0) + sp * v1(2), v1(1), -sp * v1(0) + cp * v1(2)};
        return geom::Vec3<PartFrame, Scalar>::from_raw(v2);
    }

    void update() override {
        if (this->throttle() <= Scalar(0)) return;
        auto dir = deflected_direction();
        this->apply_force_at(dir * (this->max_thrust() * this->throttle()));
    }

private:
    Scalar clamp_angle(Scalar a) const noexcept {
        return a < -max_angle_ ? -max_angle_ : (a > max_angle_ ? max_angle_ : a);
    }

    geom::Vec3<PartFrame, Scalar> base_direction_;
    Scalar max_angle_;
    Scalar pitch_ = Scalar(0);
    Scalar yaw_   = Scalar(0);
};

using GimbaledThruster = GimbaledThrusterT<Real>;

} // namespace manta::parts
