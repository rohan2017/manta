#pragma once

#include <algorithm>
#include "../../core/part.hpp"

namespace manta::parts {

// A thruster that applies a force along a configurable direction in its own
// part frame, scaled by throttle in [0, 1].
template <class Scalar = Real>
class ThrusterT : public PartT<Scalar> {
public:
    explicit ThrusterT(std::string name,
                       Scalar max_thrust,
                       geom::Vec3<PartFrame, Scalar> direction =
                           {Scalar(0), Scalar(0), Scalar(1)})
        : PartT<Scalar>(std::move(name))
        , max_thrust_(max_thrust)
        , direction_(direction) {}

    void set_throttle(Scalar t) noexcept {
        throttle_ = t < Scalar(0) ? Scalar(0) : (t > Scalar(1) ? Scalar(1) : t);
    }

    Scalar                                  throttle()   const noexcept { return throttle_; }
    Scalar                                  max_thrust() const noexcept { return max_thrust_; }
    const geom::Vec3<PartFrame, Scalar>&    direction()  const noexcept { return direction_; }

    void update() override {
        if (throttle_ > Scalar(0)) {
            this->apply_force_at(direction_ * (max_thrust_ * throttle_));
        }
    }

private:
    Scalar                          max_thrust_;
    geom::Vec3<PartFrame, Scalar>   direction_;
    Scalar                          throttle_ = Scalar(0);
};

using Thruster = ThrusterT<Real>;

} // namespace manta::parts
