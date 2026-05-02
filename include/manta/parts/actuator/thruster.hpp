#pragma once

#include <algorithm>
#include <array>
#include "../../core/part.hpp"

namespace manta::parts {

// Polynomial-in-throttle thruster — the actuator analog of `Surface`. Each
// tick the thruster's throttle command (clamped to [0, 1]) is raised to N
// powers and combined with N user-supplied force and torque coefficients in
// part frame:
//
//     F = Σ_{k=1..N} F_k · throttle^k
//     τ = Σ_{k=1..N} τ_k · throttle^k
//
// The user-facing classes are `Thruster1`..`Thruster4` (and Scalar-templated
// `Thruster1T<Scalar>`..`Thruster4T<Scalar>` for estimator use). Most real
// thrusters are well modeled by N=1 — `F_1 = direction · max_thrust`, no
// reaction torque — and the convenience constructor
// `Thruster1T(name, max_thrust, direction)` provides exactly that.
//
// `Thruster` and `ThrusterT<Scalar>` aliases at the bottom of the file
// preserve the canonical name for the common N=1 case.
namespace detail {

template <int N, class Scalar>
class ThrusterImpl : public PartT<Scalar> {
public:
    static_assert(N >= 1 && N <= 4, "ThrusterImpl<N>: N must be in [1, 4]");

    using Vec = geom::Vec3<PartFrame, Scalar>;

    ThrusterImpl(std::string name,
                 const std::array<Vec, N>& force_coefs,
                 const std::array<Vec, N>& torque_coefs)
        : PartT<Scalar>(std::move(name))
        , F_(force_coefs)
        , T_(torque_coefs) {}

    void set_throttle(Scalar t) noexcept {
        throttle_ = t < Scalar(0) ? Scalar(0) : (t > Scalar(1) ? Scalar(1) : t);
    }
    Scalar throttle() const noexcept { return throttle_; }

    const std::array<Vec, N>& force_coefs()  const noexcept { return F_; }
    const std::array<Vec, N>& torque_coefs() const noexcept { return T_; }

    // Useful for estimator parity tests / external observers — the most-recent
    // total force and torque emitted in part frame.
    const Vec& last_force()  const noexcept { return last_force_; }
    const Vec& last_torque() const noexcept { return last_torque_; }

    void update() override {
        Eigen::Matrix<Scalar, 3, 1> F = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Eigen::Matrix<Scalar, 3, 1> T = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Scalar power = throttle_;                  // throttle^1
        for (int k = 0; k < N; ++k) {
            F += F_[k].raw() * power;
            T += T_[k].raw() * power;
            if (k + 1 < N) power *= throttle_;
        }
        last_force_  = Vec::from_raw(F);
        last_torque_ = Vec::from_raw(T);

        if (throttle_ > Scalar(0)) {
            this->apply_force_at(last_force_);
            this->apply_torque(last_torque_);
        }
    }

private:
    std::array<Vec, N> F_;
    std::array<Vec, N> T_;
    Scalar             throttle_   = Scalar(0);
    Vec                last_force_;
    Vec                last_torque_;
};

} // namespace detail

template <class Scalar = Real>
class Thruster1T : public detail::ThrusterImpl<1, Scalar> {
public:
    using Base = detail::ThrusterImpl<1, Scalar>;
    using Vec  = typename Base::Vec;

    using Base::Base;

    // Convenience: linear-in-throttle thrust along `direction`, no reaction
    // torque. Equivalent to F_1 = direction · max_thrust, τ_1 = 0.
    Thruster1T(std::string name,
               Scalar max_thrust,
               Vec direction = Vec{Scalar(0), Scalar(0), Scalar(1)})
        : Base(std::move(name),
               std::array<Vec, 1>{direction * max_thrust},
               std::array<Vec, 1>{Vec::zero()}) {}
};

template <class Scalar = Real>
class Thruster2T : public detail::ThrusterImpl<2, Scalar> {
public:
    using detail::ThrusterImpl<2, Scalar>::ThrusterImpl;
};
template <class Scalar = Real>
class Thruster3T : public detail::ThrusterImpl<3, Scalar> {
public:
    using detail::ThrusterImpl<3, Scalar>::ThrusterImpl;
};
template <class Scalar = Real>
class Thruster4T : public detail::ThrusterImpl<4, Scalar> {
public:
    using detail::ThrusterImpl<4, Scalar>::ThrusterImpl;
};

using Thruster1 = Thruster1T<Real>;
using Thruster2 = Thruster2T<Real>;
using Thruster3 = Thruster3T<Real>;
using Thruster4 = Thruster4T<Real>;

// Canonical name for the common 1st-order case.
template <class Scalar = Real> using ThrusterT = Thruster1T<Scalar>;
using Thruster = Thruster1;

} // namespace manta::parts
