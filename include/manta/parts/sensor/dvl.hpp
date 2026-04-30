#pragma once

#include "../../core/noise.hpp"
#include "../../core/part.hpp"

namespace manta::parts {

struct DvlNoiseParams {
    float velocity_sigma = 0.0f;
};

// Doppler Velocity Log — body-frame linear velocity sensor.
template <class Scalar = Real>
class DVLT : public PartT<Scalar> {
public:
    explicit DVLT(std::string name, DvlNoiseParams p = DvlNoiseParams{})
        : PartT<Scalar>(std::move(name))
        , noise_{p.velocity_sigma} {}

    void update() override {
        last_vel_ = this->velocity_body() + noise_;
    }

    void set_measurement(const geom::Vec3<PartFrame, Scalar>& velocity) noexcept {
        last_vel_ = velocity;
    }

    const geom::Vec3<PartFrame, Scalar>& last_velocity() const noexcept { return last_vel_; }
    Noise<WhiteGaussian>&                 noise()              noexcept { return noise_; }

private:
    Noise<WhiteGaussian>           noise_;
    geom::Vec3<PartFrame, Scalar>  last_vel_;
};

using DVL = DVLT<Real>;

} // namespace manta::parts
