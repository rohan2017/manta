#pragma once

#include "../../core/noise.hpp"
#include "../../core/part.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

struct DvlNoiseParams {
    float velocity_sigma = 0.0f;
};

// Doppler Velocity Log — body-frame linear velocity sensor.
//
// `rate_hz` caps the effective sample rate; default 0 = every tick.
// `consume_fresh()` is the EKF-side one-shot probe (see imu.hpp).
template <class Scalar = Real>
class DVLT : public PartT<Scalar> {
public:
    explicit DVLT(std::string name,
                  DvlNoiseParams p = DvlNoiseParams{},
                  Real rate_hz     = Real(0))
        : PartT<Scalar>(std::move(name))
        , noise_{p.velocity_sigma}
        , gate_{rate_hz} {}

    void update() override {
        Real dt = (this->craft_ && this->craft_->has_world()) ? this->craft().world().clock().dt() : Real(0);
        if (!gate_.tick(dt)) return;
        last_vel_ = this->velocity_body() + noise_;
        fresh_ = true;
    }

    void set_measurement(const geom::Vec3<PartFrame, Scalar>& velocity) noexcept {
        last_vel_ = velocity;
        fresh_ = true;
    }

    bool consume_fresh() noexcept { bool was = fresh_; fresh_ = false; return was; }
    bool fresh() const noexcept { return fresh_; }

    const geom::Vec3<PartFrame, Scalar>& last_velocity() const noexcept { return last_vel_; }
    Noise<WhiteGaussian>&                 noise()              noexcept { return noise_; }

private:
    Noise<WhiteGaussian>           noise_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_vel_;
};

using DVL = DVLT<Real>;

} // namespace manta::parts
