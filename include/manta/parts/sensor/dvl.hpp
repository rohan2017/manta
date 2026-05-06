#pragma once

#include "../../core/noise.hpp"
#include "../../core/noise_registry.hpp"
#include "../../core/part.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

// Doppler Velocity Log — body-frame linear velocity sensor.
//
// `rate_hz` caps the effective sample rate; default 0 = every tick.
// `consume_fresh()` is the EKF-side one-shot probe (see imu.hpp).
template <class Scalar = MFloat>
class DVLT : public PartT<Scalar> {
public:
    explicit DVLT(std::string name,
                  float velocity_sigma = 0.0f,
                  MFloat  rate_hz        = MFloat(0))
        : PartT<Scalar>(std::move(name))
        , noise_{velocity_sigma}
        , gate_{rate_hz} {}

    void update() override {
        MFloat dt = (this->craft_ && this->craft_->has_world()) ? this->craft().world().clock().dt() : MFloat(0);
        if (!gate_.tick(dt)) return;
        last_vel_ = this->velocity_body() + noise_;
        fresh_ = true;
    }

    void set_measurement(const geom::Vec3<PartFrame, Scalar>& velocity) noexcept {
        last_vel_ = velocity;
        fresh_ = true;
    }

    void register_noise(NoiseRegistry& r) override {
        if (noise_.sigma() >= 0.0f) r.register_white_3d(noise_);
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

using DVL = DVLT<MFloat>;

} // namespace manta::parts
