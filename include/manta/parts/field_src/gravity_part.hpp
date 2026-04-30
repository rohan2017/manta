#pragma once

#include "../../core/craft.hpp"
#include "../../fields/gravity_field.hpp"

namespace manta::parts {

// Applies gravity to the entire craft on every tick. Templated on Scalar.
//
// Note: the GravityField is not yet templated on Scalar; its `g()` returns
// `Vec3<SceneFrame, Real>` (typically float). We cast to Scalar when
// applying — fine for sim (Real == Scalar) and acceptable for the
// estimator path (the gravity vector is treated as a constant input,
// not differentiated through).
template <class Scalar = Real>
class GravityPartT : public PartT<Scalar> {
public:
    explicit GravityPartT(std::string name = "gravity")
        : PartT<Scalar>(std::move(name)) {}

    void update() override {
        auto& gf = this->template field<fields::GravityField>();

        // Rotate g from SceneFrame to PartFrame using the part's kinematic cache.
        auto q_part_from_scene = this->template orientation<SceneFrame>().raw().conjugate();

        // Cast the field's float-typed g into Scalar for downstream math.
        Eigen::Matrix<Scalar, 3, 1> g_scene = gf.g().raw().template cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> g_part_e = q_part_from_scene * g_scene;
        auto g_part = geom::Vec3<PartFrame, Scalar>::from_raw(g_part_e);

        Scalar m = this->craft().root().get_mass();
        this->apply_force_at(g_part * m);
    }
};

using GravityPart = GravityPartT<Real>;

} // namespace manta::parts
