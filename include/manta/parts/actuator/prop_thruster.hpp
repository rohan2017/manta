#pragma once

#include "thruster.hpp"

namespace manta::parts {

// Thruster with propeller reaction torque.
// Applies counter-torque along the thrust direction: τ = kt * thrust * sign.
// sign: +1 if cw, -1 if CCW.
template <class Scalar = Real>
class PropThrusterT : public ThrusterT<Scalar> {
public:
    explicit PropThrusterT(std::string name,
                           Scalar max_thrust,
                           Scalar kt = Scalar(0.02f),
                           bool cw = false,
                           geom::Vec3<PartFrame, Scalar> dir =
                               {Scalar(0), Scalar(0), Scalar(1)})
        : ThrusterT<Scalar>(std::move(name), max_thrust, dir)
        , kt_(kt)
        , cw_(cw) {}

    void update() override {
        ThrusterT<Scalar>::update();
        Scalar thrust = this->max_thrust() * this->throttle();
        if (thrust > Scalar(0)) {
            Scalar sign = cw_ ? Scalar(1) : Scalar(-1);
            this->apply_torque(this->direction() * (thrust * kt_ * sign));
        }
    }

    Scalar kt() const noexcept { return kt_; }
    bool   cw() const noexcept { return cw_; }

private:
    Scalar kt_;
    bool   cw_;
};

using PropThruster = PropThrusterT<Real>;

} // namespace manta::parts
