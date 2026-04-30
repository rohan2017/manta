#pragma once

#include "../../core/part.hpp"

namespace manta::parts {

// A lump of mass with full 3×3 MOI tensor. Templated on Scalar.
template <class Scalar = Real>
class MassT : public PartT<Scalar> {
public:
    MassT(std::string name, Scalar mass,
          const geom::Mat3<PartFrame, PartFrame, Scalar>& moi)
        : PartT<Scalar>(std::move(name)) {
        this->set_mass(mass);
        this->set_moi(moi);
    }

    void update() override {}
};

using Mass = MassT<Real>;

} // namespace manta::parts
