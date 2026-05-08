// One craft definition, instantiated three times by main.cpp: sim
// (synthetic readings + truth state), est value-side (EKF's tracked
// craft), est Jet-side (Jet shadow for autodiff).

#pragma once

#include <array>
#include <string>
#include <utility>

#include "manta/core/craft.hpp"
#include "manta/parts/actuator/thruster.hpp"
#include "manta/parts/sensor/dvl.hpp"
#include "manta/parts/sensor/imu.hpp"
#include "manta/parts/structure/mass.hpp"

namespace ex10 {

template <class Scalar>
class DemoCraftT : public manta::CraftT<Scalar> {
public:
    explicit DemoCraftT(std::string name)
        : manta::CraftT<Scalar>(std::move(name)) {
        body_ = &this->root().template add<manta::parts::MassT<Scalar>>(
            "body", Scalar(1.0));
        thrust_ = &this->root().template add<manta::parts::Thruster1T<Scalar>>(
            "thrust",
            std::array<manta::geom::Vec3<manta::PartFrame, Scalar>, 1>{
                manta::geom::Vec3<manta::PartFrame, Scalar>{
                    Scalar(15), Scalar(0), Scalar(0)}},
            std::array<manta::geom::Vec3<manta::PartFrame, Scalar>, 1>{
                manta::geom::Vec3<manta::PartFrame, Scalar>::zero()});
        imu_ = &this->root().template add<manta::parts::IMUT<Scalar>>(
            "imu", /*accel_sigma=*/0.05f, /*gyro_sigma=*/0.005f);
        dvl_ = &this->root().template add<manta::parts::DVLT<Scalar>>(
            "dvl", /*velocity_sigma=*/0.02f);
        this->root().compute_params();
    }

    manta::parts::IMUT<Scalar>&       imu()    noexcept { return *imu_; }
    manta::parts::DVLT<Scalar>&       dvl()    noexcept { return *dvl_; }
    manta::parts::Thruster1T<Scalar>& thrust() noexcept { return *thrust_; }

private:
    manta::parts::MassT<Scalar>*       body_   = nullptr;
    manta::parts::Thruster1T<Scalar>*  thrust_ = nullptr;
    manta::parts::IMUT<Scalar>*        imu_    = nullptr;
    manta::parts::DVLT<Scalar>*        dvl_    = nullptr;
};

}  // namespace ex10
