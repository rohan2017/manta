#pragma once

#include <type_traits>

#include "../../core/noise.hpp"
#include "../../core/part.hpp"
#include "../../fields/mag_field.hpp"
#include "../../fields/templated_query.hpp"
#include "../../sim/rate_gate.hpp"

namespace manta::parts {

struct MagnetometerNoiseParams {
    float sigma = 0.0f;   // Tesla
};

// A 3-axis magnetometer. Each tick, queries the registered MagField at the
// part's scene-frame position, rotates the result into part frame, optionally
// adds white-Gaussian noise, and stores the value.
//
// Required fields: MagField (any concrete subclass — DipoleMagField, or a
// future IGRF model). The field is queried via the abstract base typeid.
//
// Like IMU/DVL, the field itself is NOT scalar-templated — values are
// queried at Real precision, cast to Scalar at the boundary, and treated as
// constants for autodiff (zero derivative w.r.t. craft state). Acceptable
// for short-horizon estimator predict steps where ∂B/∂x is negligible.
//
// External-measurement seam (`set_measurement`) lets an estimator-side
// craft be driven from real sensor data over Zenoh, mirroring the IMU/DVL
// convention.
template <class Scalar = Real>
class MagnetometerT : public PartT<Scalar> {
public:
    explicit MagnetometerT(std::string name,
                           MagnetometerNoiseParams p = MagnetometerNoiseParams{},
                           Real rate_hz = Real(0))
        : PartT<Scalar>(std::move(name))
        , noise_{WhiteGaussian{p.sigma}}
        , gate_{rate_hz} {}

    void update() override {
        Real dt = (this->craft_ && this->craft_->has_world()) ? this->craft().world().clock().dt() : Real(0);
        if (!gate_.tick(dt)) return;

        const auto* mf_base = this->field_ptr(typeid(fields::MagField));
        if (!mf_base) {
            last_b_ = geom::Vec3<PartFrame, Scalar>::zero();
            fresh_ = true;
            return;
        }
        const auto* mf = static_cast<const fields::MagField*>(mf_base);

        auto p_scaled = this->template position<SceneFrame>();
        auto b_scene_v = fields::state_at_templated<Scalar>(*mf, p_scaled);

        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();
        auto b_part = geom::Vec3<PartFrame, Scalar>::from_raw(
            q_part_from_scene * b_scene_v.raw());

        last_b_ = b_part + noise_;
        fresh_ = true;
    }

    // External-measurement seam (estimator path). Always marks the reading
    // as fresh — bypasses the rate gate.
    void set_measurement(const geom::Vec3<PartFrame, Scalar>& b) noexcept {
        last_b_ = b;
        fresh_ = true;
    }

    bool consume_fresh() noexcept { bool was = fresh_; fresh_ = false; return was; }
    bool fresh() const noexcept { return fresh_; }

    const geom::Vec3<PartFrame, Scalar>& last_b() const noexcept { return last_b_; }

    Noise<WhiteGaussian>& noise() noexcept { return noise_; }

private:
    Noise<WhiteGaussian>           noise_;
    sim::RateGate                  gate_;
    bool                           fresh_ = false;
    geom::Vec3<PartFrame, Scalar>  last_b_;
};

using Magnetometer = MagnetometerT<Real>;

} // namespace manta::parts
