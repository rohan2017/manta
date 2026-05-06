#pragma once

#include "../../core/noise.hpp"
#include "../../core/part.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/templated_query.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

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
template <class Scalar = MFloat>
class IMUT : public PartT<Scalar> {
public:
    // σ values default to 0 (sample no noise on the value path; register
    // with EKF as a zero-variance slot, which is wasteful but harmless).
    // Pass σ < 0 to disable a channel entirely — neither sampled by sim
    // nor registered with the EKF. The codegen uses σ < 0 as the "skip"
    // sentinel so unspecified noise channels don't bloat NumNoiseSlots /
    // BiasDim.
    //
    // gyro_bias_sigma is the random-walk diffusion coefficient for the
    // gyro bias: at runtime the bias drifts via N(0, σ²·dt) per tick on
    // the sim path, and the EKF estimates it as a 3-DOF augmented state
    // when registered.
    explicit IMUT(std::string name,
                  float accel_sigma     = 0.0f,
                  float gyro_sigma      = 0.0f,
                  MFloat rate_hz        = MFloat(0),
                  float gyro_bias_sigma = -1.0f)
        : PartT<Scalar>(std::move(name))
        , accel_noise_{accel_sigma}
        , gyro_noise_{gyro_sigma}
        , gyro_bias_{gyro_bias_sigma}
        , gate_{rate_hz} {}

    // Body-frame specific force — what a real accelerometer reports.
    //
    // Specific force = (housing_inertial_accel − gravity), so a craft in
    // free fall reads zero (housing accelerates at exactly g, sensor mass
    // doesn't deflect) and a stationary craft reads −g_body (≈ +9.81 ẑ
    // in body frame at q=identity). The kinematic `acceleration_body()`
    // accessor returns body-frame inertial accel including gravity's
    // contribution (since `sense_and_aggregate` sums gravity into the
    // wrench), so we subtract `R(q)^T · g_scene` to recover specific
    // force.
    //
    // GravityField is OPTIONAL augmentation. The macro gates compilation;
    // at runtime, an unattached test craft or a world without gravity
    // gracefully returns the raw kinematic body acceleration.
    geom::Vec3<PartFrame, Scalar> specific_force_body() const {
        auto a_body = this->acceleration_body();
        if constexpr (MANTA_PART_AUGMENTS_FIELD(MANTA_HAS_GRAVITY_FIELD)) {
            const auto* g_field = this->template field_or_null<fields::GravityField>();
            if (!g_field) return a_body;
            auto pos_scene = this->scene_to_part().position();
            auto g_scene   = fields::state_at_templated<Scalar>(*g_field, pos_scene);
            auto g_body    = this->scene_to_part().rotate_inverse(g_scene);
            return geom::Vec3<PartFrame, Scalar>::from_raw(
                a_body.raw() - g_body.raw());
        } else {
            return a_body;
        }
    }

    void update() override {
        MFloat dt = (this->craft_ && this->craft_->has_world()) ? this->craft().world().clock().dt() : MFloat(0);
        if (!gate_.tick(dt)) return;
        // gyro reading: ω + white noise + RW bias. The RW bias's `+`
        // operator reads the EKF's current estimate (via state3()) on
        // the Jet path and the sim's drifting truth on the value path.
        last_accel_ = this->specific_force_body() + accel_noise_;
        last_gyro_  = this->angular_velocity_body() + gyro_noise_ + gyro_bias_;
        fresh_ = true;
    }

    // Conditional registration: any channel with σ ≥ 0 is registered.
    // Codegen passes σ < 0 to suppress channels the user didn't enable
    // on the descriptor.
    void register_noise(NoiseRegistry& r) override {
        if (accel_noise_.sigma() >= 0.0f) r.register_white_3d(accel_noise_);
        if (gyro_noise_.sigma()  >= 0.0f) r.register_white_3d(gyro_noise_);
        if (gyro_bias_.sigma()   >= 0.0f) r.register_random_walk_3d(gyro_bias_);
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
    Noise<RandomWalk>&    gyro_bias()   noexcept { return gyro_bias_; }
    const Noise<WhiteGaussian>& accel_noise() const noexcept { return accel_noise_; }
    const Noise<WhiteGaussian>& gyro_noise()  const noexcept { return gyro_noise_; }
    const Noise<RandomWalk>&    gyro_bias()   const noexcept { return gyro_bias_; }

private:
    Noise<WhiteGaussian>           accel_noise_;
    Noise<WhiteGaussian>           gyro_noise_;
    Noise<RandomWalk>              gyro_bias_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_accel_;
    geom::Vec3<PartFrame, Scalar>  last_gyro_;
};

using IMU = IMUT<MFloat>;

} // namespace manta::parts
