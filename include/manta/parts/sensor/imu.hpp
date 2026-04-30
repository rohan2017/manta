#pragma once

#include "../../core/noise.hpp"
#include "../../core/part.hpp"

namespace manta::parts {

struct ImuNoiseParams {
    float accel_sigma = 0.0f;
    float gyro_sigma  = 0.0f;
};

// An inertial measurement unit. Reads the kinematic-pass acceleration and
// angular velocity caches and records them each tick, with optional
// white-noise injection.
//
// Templated on Scalar; the `IMU` alias = IMUT<Real> is what existing user
// code uses.
template <class Scalar = Real>
class IMUT : public PartT<Scalar> {
public:
    explicit IMUT(std::string name, ImuNoiseParams p = ImuNoiseParams{})
        : PartT<Scalar>(std::move(name))
        , accel_noise_{p.accel_sigma}
        , gyro_noise_{p.gyro_sigma} {}

    void update() override {
        last_accel_ = this->acceleration_body() + accel_noise_;
        last_gyro_  = this->angular_velocity_body() + gyro_noise_;
    }

    // External-measurement seam (estimator path).
    void set_measurement(const geom::Vec3<PartFrame, Scalar>& accel,
                         const geom::Vec3<PartFrame, Scalar>& gyro) noexcept {
        last_accel_ = accel;
        last_gyro_  = gyro;
    }

    const geom::Vec3<PartFrame, Scalar>& last_accel() const noexcept { return last_accel_; }
    const geom::Vec3<PartFrame, Scalar>& last_gyro()  const noexcept { return last_gyro_; }

    Noise<WhiteGaussian>& accel_noise() noexcept { return accel_noise_; }
    Noise<WhiteGaussian>& gyro_noise()  noexcept { return gyro_noise_; }

private:
    Noise<WhiteGaussian>           accel_noise_;
    Noise<WhiteGaussian>           gyro_noise_;
    geom::Vec3<PartFrame, Scalar>  last_accel_;
    geom::Vec3<PartFrame, Scalar>  last_gyro_;
};

using IMU = IMUT<Real>;

} // namespace manta::parts
