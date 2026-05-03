#pragma once

#include "../../core/noise.hpp"
#include "../../core/part.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

struct ImuNoiseParams {
    float accel_sigma = 0.0f;
    float gyro_sigma  = 0.0f;
};

// An inertial measurement unit. Reads the kinematic-pass acceleration and
// angular velocity caches and records them each tick, with optional
// white-noise injection.
//
// `rate_hz` (default 0 = unrated, refreshes every tick) caps the sensor's
// effective sample rate. When set, the IMU's `update()` only refreshes
// `last_accel_` / `last_gyro_` once per `1/rate_hz` of sim time. The
// sensor exposes `consume_fresh()` for one-shot consumption: returns true
// on the tick a new reading was just produced and clears the freshness
// bit. The plain `last_accel()` / `last_gyro()` getters continue to
// return the most recent value regardless of staleness.
template <class Scalar = Real>
class IMUT : public PartT<Scalar> {
public:
    explicit IMUT(std::string name,
                  ImuNoiseParams p = ImuNoiseParams{},
                  Real rate_hz     = Real(0))
        : PartT<Scalar>(std::move(name))
        , accel_noise_{p.accel_sigma}
        , gyro_noise_{p.gyro_sigma}
        , gate_{rate_hz} {}

    void update() override {
        Real dt = (this->craft_ && this->craft_->has_world()) ? this->craft().world().clock().dt() : Real(0);
        if (!gate_.tick(dt)) return;
        last_accel_ = this->acceleration_body() + accel_noise_;
        last_gyro_  = this->angular_velocity_body() + gyro_noise_;
        fresh_ = true;
    }

    // External-measurement seam (estimator path). Always marks the reading
    // as fresh — bypasses the rate gate.
    //
    // Three forms:
    //   * set_measurement(accel, gyro) — full 6-channel inject.
    //   * set_measurement_accel(accel) — just the accelerometer half.
    //   * set_measurement_gyro(gyro)   — just the gyro half.
    //
    // The split forms exist so single-channel `connect()` calls can wire
    // sim → est cleanly without needing to bundle six floats into one
    // signal: `w.connect(sim.imu.last_accel, est.imu.set_measurement_accel)`.
    void set_measurement(const geom::Vec3<PartFrame, Scalar>& accel,
                         const geom::Vec3<PartFrame, Scalar>& gyro) noexcept {
        last_accel_ = accel;
        last_gyro_  = gyro;
        fresh_ = true;
    }

    void set_measurement_accel(const geom::Vec3<PartFrame, Scalar>& accel) noexcept {
        last_accel_ = accel;
        fresh_ = true;
    }

    void set_measurement_gyro(const geom::Vec3<PartFrame, Scalar>& gyro) noexcept {
        last_gyro_ = gyro;
        fresh_ = true;
    }

    // One-shot freshness probe — returns true exactly once per refresh.
    // Intended for the EKF/UKF wiring; everything else should use the plain
    // last_accel() / last_gyro() getters.
    bool consume_fresh() noexcept {
        bool was = fresh_; fresh_ = false; return was;
    }
    bool fresh() const noexcept { return fresh_; }

    const geom::Vec3<PartFrame, Scalar>& last_accel() const noexcept { return last_accel_; }
    const geom::Vec3<PartFrame, Scalar>& last_gyro()  const noexcept { return last_gyro_; }

    Noise<WhiteGaussian>& accel_noise() noexcept { return accel_noise_; }
    Noise<WhiteGaussian>& gyro_noise()  noexcept { return gyro_noise_; }

private:
    Noise<WhiteGaussian>           accel_noise_;
    Noise<WhiteGaussian>           gyro_noise_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_accel_;
    geom::Vec3<PartFrame, Scalar>  last_gyro_;
};

using IMU = IMUT<Real>;

} // namespace manta::parts
