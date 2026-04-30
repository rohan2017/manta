#pragma once

#include "../../core/craft.hpp"
#include "../../fields/point_gravity_field.hpp"

namespace manta::parts {

// Applies inverse-square gravity from a PointGravityField.
//
// Templated on Scalar; the field itself is non-templated, so gravity is
// treated as a constant input (no autodiff derivative w.r.t. position).
// Acceptable for local dynamics; for orbital regimes where ∂g/∂x matters,
// template PointGravityField too.
template <class Scalar = Real>
class PointGravityPartT : public PartT<Scalar> {
public:
    explicit PointGravityPartT(std::string name = "point_gravity")
        : PartT<Scalar>(std::move(name)) {}

    void update() override {
        auto& gf = this->template field<fields::PointGravityField>();

        // Bridge to the (non-templated) field via Real, then back to Scalar.
        auto pos_scaled = this->template position<SceneFrame>();
        Eigen::Matrix<Real, 3, 1> pos_real;
        if constexpr (std::is_floating_point_v<Scalar>) {
            pos_real = pos_scaled.raw().template cast<Real>();
        } else {
            // Jet-like type: extract the value component (.a).
            for (int i = 0; i < 3; ++i) pos_real(i) = Real(pos_scaled.raw()(i).a);
        }
        auto g_real = gf.g_at(geom::Vec3<SceneFrame>::from_raw(pos_real));

        // Cast g back to Scalar — derivatives w.r.t. state are zero (treating
        // gravity as locally constant for the autodiff hot path).
        Eigen::Matrix<Scalar, 3, 1> g_scene = g_real.raw().template cast<Scalar>();
        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();
        auto g_part = geom::Vec3<PartFrame, Scalar>::from_raw(
            q_part_from_scene * g_scene);

        Scalar m = this->craft().root().get_mass();
        this->apply_force_at(g_part * m);
    }
};

using PointGravityPart = PointGravityPartT<Real>;

} // namespace manta::parts
