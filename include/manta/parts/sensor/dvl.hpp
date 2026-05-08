#pragma once

#include "../../core/noise.hpp"
#include "../../core/noise_registry.hpp"
#include "../../core/part.hpp"
#include "../../estimation/measurement.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

// Doppler Velocity Log — body-frame linear velocity sensor.
//
// `rate_hz` caps the effective sample rate; default 0 = every tick.
// EKF integration is via the `velocity` `Measurement` member; z comes
// from a separate `Reading<3>` source the user passes to
// `ekf.measure(...)`.
template <class Scalar = MFloat>
class DVLT : public PartT<Scalar> {
public:
    explicit DVLT(std::string name,
                  float velocity_sigma = 0.0f,
                  MFloat  rate_hz        = MFloat(0))
        : PartT<Scalar>(std::move(name))
        , noise_{velocity_sigma}
        , gate_{rate_hz}
        , velocity{make_measurement<Scalar, 3>(
            "velocity", &last_vel_.raw(), noise_.sigma_ptr(), &fresh_)}
    {
        this->measurements_.push_back(&velocity);
    }

    void update() override {
        MFloat dt = (this->craft_ && this->craft_->has_world())
                  ? this->craft().world().clock().dt() : MFloat(0);
        fresh_ = gate_.tick(dt);
        if (!fresh_) return;
        last_vel_ = this->velocity_body() + noise_;
    }

    void register_noise(NoiseRegistry& r) override {
        if (noise_.sigma() >= 0.0f) r.register_white_3d(noise_);
    }

    bool fresh() const noexcept { return fresh_; }

    const geom::Vec3<PartFrame, Scalar>& last_velocity() const noexcept { return last_vel_; }
    Noise<WhiteGaussian>&                 noise()              noexcept { return noise_; }

    Measurement velocity;

    // ---- Legacy bridges (will be removed in Phase 5c) ----
    void set_measurement(const geom::Vec3<PartFrame, Scalar>& v) noexcept {
        last_vel_ = v; fresh_ = true;
    }
    bool consume_fresh() noexcept { bool was = fresh_; fresh_ = false; return was; }

private:
    Noise<WhiteGaussian>           noise_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_vel_;
};

using DVL = DVLT<MFloat>;

} // namespace manta::parts
