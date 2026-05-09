#pragma once

#include "../../core/noise.hpp"
#include "../../core/noise_registry.hpp"
#include "../../core/part.hpp"
#include "../../estimation/measurement.hpp"
#include "../../fields/gravity_field.hpp"
#include "../../fields/templated_query.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

// ---------------------------------------------------------------------
// IMU — full Kalibr 4-parameter noise model
// ---------------------------------------------------------------------
//
// References Kalibr's standard IMU noise model:
//   https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
//
// Per-axis measurement model:
//
//     ω̃(t) = ω(t) + b_g(t) + n_g(t)
//     ã(t) = a(t) + b_a(t) + n_a(t)
//
// where
//
//   * n_g ~ N(0, σ_g²)   gyro white noise.
//   * n_a ~ N(0, σ_a²)   accel white noise.
//   * ḃ_g, ḃ_a — bias random walks (RW driver σ_bg, σ_ba).
//
// EKF/UKF integration is via the published `accel` and `gyro`
// `Measurement` members. They point at the IMU's internal h(x) cache
// (populated by `update()` each tick), the relevant noise σ field
// (used for R), and the part's freshness flag (set by the rate gate).
//
// Roles:
//   * On a *sim* craft (value-typed instance), `update()` runs h(x)
//     plus white-noise sampling plus bias offset — producing a
//     synthetic noisy reading. Other parts of the system (sim
//     telemetry, `connect()` plumbing) read `last_accel()` etc.
//   * On a *Jet shadow* craft (Jet-typed instance bound to an EKF),
//     `update()` runs h(x) too — but on the Jet path the operator+
//     for Noise injects no value-side sample (so the .a channel is
//     pure h(x)) and instead populates the .v channel with the
//     noise-input gain L that the EKF reads back for auto-Q.
//
// `update()` does not accept external readings — z comes from a
// separate `Reading<Dim>` source passed to `ekf.measure(...)`.
template <class Scalar = MFloat>
class IMUT : public PartT<Scalar> {
public:
    // σ < 0 ⇒ skip; σ = 0 ⇒ register zero-variance; σ > 0 ⇒ register.
    explicit IMUT(std::string name,
                  float accel_sigma      = -1.0f,
                  float gyro_sigma       = -1.0f,
                  float accel_bias_sigma = -1.0f,
                  float gyro_bias_sigma  = -1.0f,
                  MFloat rate_hz         = MFloat(0))
        : PartT<Scalar>(std::move(name))
        , accel_noise_{accel_sigma}
        , gyro_noise_{gyro_sigma}
        , accel_bias_{accel_bias_sigma}
        , gyro_bias_{gyro_bias_sigma}
        , gate_{rate_hz}
        , accel{make_measurement<Scalar, 3>(
            "accel", &last_accel_.raw(), accel_noise_.sigma_ptr(), &fresh_)}
        , gyro {make_measurement<Scalar, 3>(
            "gyro",  &last_gyro_.raw(),  gyro_noise_.sigma_ptr(),  &fresh_)}
    {
        this->measurements_.push_back(&accel);
        this->measurements_.push_back(&gyro);
        this->random_walks_.push_back({"accel_bias", &accel_bias_});
        this->random_walks_.push_back({"gyro_bias",  &gyro_bias_});
    }

    // ---- h(x) helpers (also used by user code; not part of the EKF
    //      interface — the EKF reads through the Measurement objects). ----

    // Body-frame specific force (housing accel − gravity).
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
        MFloat dt = (this->craft_ && this->craft_->has_world())
                  ? this->craft().world().clock().dt() : MFloat(0);
        fresh_ = gate_.tick(dt);
        if (!fresh_) return;
        // Kalibr model: measurement = truth + white_noise + bias.
        // On the sim (value-typed) instance: noise samples + bias state
        // both flow through operator+ and produce a noisy reading.
        // On the Jet-typed shadow: operator+ injects auto-Q L gain in
        // place of the noise sample, leaving .a as pure h(x) and .v as
        // the noise-input Jacobian.
        last_accel_ = this->specific_force_body() + accel_noise_ + accel_bias_;
        last_gyro_  = this->angular_velocity_body() + gyro_noise_ + gyro_bias_;
    }

    // Drift simulated true biases. Sim role only — call from your tick
    // loop if you want bias drift in the synthetic reading. The
    // Jet-side instance's bias state is tracked separately by the EKF.
    void advance_biases(float dt) noexcept {
        accel_bias_.advance(dt);
        gyro_bias_.advance(dt);
    }

    void register_noise(NoiseRegistry& r) override {
        if (accel_noise_.sigma() >= 0.0f) r.register_white_3d(accel_noise_);
        if (gyro_noise_.sigma()  >= 0.0f) r.register_white_3d(gyro_noise_);
        if (accel_bias_.sigma()  >= 0.0f) r.register_random_walk(accel_bias_);
        if (gyro_bias_.sigma()   >= 0.0f) r.register_random_walk(gyro_bias_);
    }

    // Accessors for downstream non-EKF consumers (sim telemetry, viewer).
    const geom::Vec3<PartFrame, Scalar>& last_accel() const noexcept { return last_accel_; }
    const geom::Vec3<PartFrame, Scalar>& last_gyro()  const noexcept { return last_gyro_;  }

    // Did this IMU produce a fresh reading on the last update()?
    bool fresh() const noexcept { return fresh_; }


    Noise<WhiteGaussian>& accel_noise() noexcept { return accel_noise_; }
    Noise<WhiteGaussian>& gyro_noise()  noexcept { return gyro_noise_; }
    Noise<RandomWalk<3>>& accel_bias()  noexcept { return accel_bias_; }
    Noise<RandomWalk<3>>& gyro_bias()   noexcept { return gyro_bias_; }
    const Noise<WhiteGaussian>& accel_noise() const noexcept { return accel_noise_; }
    const Noise<WhiteGaussian>& gyro_noise()  const noexcept { return gyro_noise_; }
    const Noise<RandomWalk<3>>& accel_bias()  const noexcept { return accel_bias_; }
    const Noise<RandomWalk<3>>& gyro_bias()   const noexcept { return gyro_bias_; }

    // Public Measurement handles — pass to ekf.measure(...).
    MeasurementHandle<3> accel;
    MeasurementHandle<3> gyro;

private:
    Noise<WhiteGaussian>           accel_noise_;
    Noise<WhiteGaussian>           gyro_noise_;
    Noise<RandomWalk<3>>           accel_bias_;
    Noise<RandomWalk<3>>           gyro_bias_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_accel_;
    geom::Vec3<PartFrame, Scalar>  last_gyro_;
};

using IMU = IMUT<MFloat>;

} // namespace manta::parts
