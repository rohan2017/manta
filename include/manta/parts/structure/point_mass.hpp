#pragma once

#include "../../core/part.hpp"

namespace manta::parts {

// A leaf part that contributes only mass/MOI to the craft — no forces applied.
// Used to model structural elements, fuel tanks, ballast, etc.
//
// Templated on Scalar so the same class serves both the sim path
// (PointMassT<Real>, the standard `PointMass` alias below) and the
// estimator path (PointMassT<ceres::Jet<...>>) — autodiff-friendly when
// the EKF instantiates the est craft with Jet to extract Jacobians.
template <class Scalar = Real>
class PointMassT : public PartT<Scalar> {
public:
    explicit PointMassT(std::string name, Scalar mass = Scalar(0),
                        const geom::Vec3<PartFrame, Scalar>& com =
                            geom::Vec3<PartFrame, Scalar>::zero())
        : PartT<Scalar>(std::move(name)) {
        this->set_mass(mass);
        this->set_com(com);
    }

    void update() override {}
};

using PointMass = PointMassT<Real>;

} // namespace manta::parts
