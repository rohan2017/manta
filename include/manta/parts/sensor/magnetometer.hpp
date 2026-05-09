#pragma once

#include <type_traits>

#include "../../core/noise.hpp"
#include "../../core/noise_registry.hpp"
#include "../../core/part.hpp"
#include "../../estimation/measurement.hpp"
#include "../../fields/mag_field.hpp"
#include "../../fields/templated_query.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

// 3-axis magnetometer. Each tick (modulo the rate gate), queries the
// registered MagField at the part's scene-frame position, rotates the
// result into part frame, samples white-Gaussian noise, and stores the
// value. Exposes the result as the public `b` Measurement member; pass
// it to `ekf.measure(...)` along with a Reading source for z.
template <class Scalar = MFloat>
class MagnetometerT : public PartT<Scalar> {
public:
    // Hard-required field. The Python codegen also validates this at
    // config time; the static_assert below is defense-in-depth for any
    // path that bypasses the codegen.
    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_MAG_FIELD,
        "Magnetometer requires a MagField on the world. Register one "
        "with World.add_field(MagField(...)) (or DipoleMagField / "
        "future IGRF subclasses), or remove the Magnetometer part.");

    explicit MagnetometerT(std::string name,
                           float sigma   = 0.0f,    // Tesla
                           MFloat  rate_hz = MFloat(0))
        : PartT<Scalar>(std::move(name))
        , noise_{WhiteGaussian{sigma}}
        , gate_{rate_hz}
        , b{make_measurement<Scalar, 3>(
            "b", &last_b_.raw(), noise_.sigma_ptr(), &fresh_)}
    {
        this->measurements_.push_back(&b);
    }

    void update() override {
        MFloat dt = (this->craft_ && this->craft_->has_world())
                  ? this->craft().world().clock().dt() : MFloat(0);
        fresh_ = gate_.tick(dt);
        if (!fresh_) return;
        // MagField is REQUIRED at build time. Use field_or_null at runtime
        // so an unattached test craft, or a craft whose world (in a
        // multi-world TU) deliberately omitted the field, reports zero
        // rather than crashing.
        const auto* mf = this->template field_or_null<fields::MagField>();
        if (!mf) {
            last_b_ = geom::Vec3<PartFrame, Scalar>::zero();
            return;
        }

        auto p_scaled  = this->template position<SceneFrame>();
        auto b_scene_v = fields::state_at_templated<Scalar>(*mf, p_scaled);

        auto q_part_from_scene =
            this->template orientation<SceneFrame>().raw().conjugate();
        auto b_part = geom::Vec3<PartFrame, Scalar>::from_raw(
            q_part_from_scene * b_scene_v.raw());

        last_b_ = b_part + noise_;
    }

    void register_noise(NoiseRegistry& r) override {
        if (noise_.sigma() >= 0.0f) r.register_white_3d(noise_);
    }

    bool fresh() const noexcept { return fresh_; }

    const geom::Vec3<PartFrame, Scalar>& last_b() const noexcept { return last_b_; }

    Noise<WhiteGaussian>&       noise()       noexcept { return noise_; }
    const Noise<WhiteGaussian>& noise() const noexcept { return noise_; }

    MeasurementHandle<3> b;


private:
    Noise<WhiteGaussian>           noise_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_b_;
};

using Magnetometer = MagnetometerT<MFloat>;

} // namespace manta::parts
