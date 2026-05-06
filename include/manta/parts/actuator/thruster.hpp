#pragma once

#include <algorithm>
#include <array>
#include "../../core/noise.hpp"
#include "../../core/noise_registry.hpp"
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

    // force_noise_sigma defaults to 0 (register, but contributes nothing
    // to Q until the user mutates σ at runtime). Pass < 0 to suppress
    // registration entirely. Codegen passes the descriptor's
    // force_noise_sigma here.
    ThrusterImpl(std::string name,
                 const std::array<Vec, N>& force_coefs,
                 const std::array<Vec, N>& torque_coefs,
                 float force_noise_sigma = 0.0f)
        : PartT<Scalar>(std::move(name))
        , F_(force_coefs)
        , T_(torque_coefs)
        , force_noise_(force_noise_sigma) {}

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

    // 3-axis Gaussian process noise on the emitted force, in part frame.
    // Default σ = 0 (no behavior change unless the user opts in).
    //
    // Sim path (Scalar = MFloat / double): σ samples are drawn each tick and
    // added to F before apply_force_at — gives the sim stochastic realism.
    // EKF predict path (Scalar = Jet) with the EKF's NumNoiseSlots > 0:
    // the Vec3+Noise operator's autodiff branch injects Jet inputs at the
    // registered noise slots. The EKF reads back the noise-input gain L
    // and assembles Q automatically. State-dependent σ: call set_sigma()
    // from update() before the noise is added.
    Noise<WhiteGaussian>& force_noise() noexcept { return force_noise_; }
    const Noise<WhiteGaussian>& force_noise() const noexcept { return force_noise_; }

    void register_noise(NoiseRegistry& r) override {
        // σ < 0 is the "skip registration" sentinel — codegen sets it
        // when the user didn't enable force_noise on the descriptor.
        // Hand-instantiated Thrusters use σ=0 by default, which still
        // registers (preserves existing test behaviour).
        if (force_noise_.sigma() >= 0.0f) {
            r.register_white_3d(force_noise_);
        }
    }

    void update() override {
        Eigen::Matrix<Scalar, 3, 1> F = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Eigen::Matrix<Scalar, 3, 1> T = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Scalar power = throttle_;                  // throttle^1
        for (int k = 0; k < N; ++k) {
            F += F_[k].raw() * power;
            T += T_[k].raw() * power;
            if (k + 1 < N) power *= throttle_;
        }
        // Inject force noise via the canonical Vec + Noise operator —
        // sim path samples and adds; EKF Jet path injects Jet inputs.
        // Always applied (even at zero throttle) so the noise term is
        // present in the predict Jacobian for auto-Q assembly. The
        // value-side cost of an extra zero-Wrench accumulation is
        // negligible.
        Vec F_out = Vec::from_raw(F) + force_noise_;

        last_force_  = F_out;
        last_torque_ = Vec::from_raw(T);

        this->apply_force_at(last_force_);
        this->apply_torque(last_torque_);
    }

private:
    std::array<Vec, N>   F_;
    std::array<Vec, N>   T_;
    Scalar               throttle_   = Scalar(0);
    Vec                  last_force_;
    Vec                  last_torque_;
    Noise<WhiteGaussian> force_noise_;
};

} // namespace detail

template <class Scalar = MFloat>
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

// detail helper: a Thruster of order N where only the linear term is set,
// from (max_thrust, direction). Higher-order coefficients are zero.
namespace detail {
    template <int N, class Scalar>
    inline std::array<geom::Vec3<PartFrame, Scalar>, N>
    linear_only_force_coefs(geom::Vec3<PartFrame, Scalar> direction, Scalar max_thrust) {
        std::array<geom::Vec3<PartFrame, Scalar>, N> a{};
        a[0] = direction * max_thrust;
        return a;
    }
    template <int N, class Scalar>
    inline std::array<geom::Vec3<PartFrame, Scalar>, N> zero_coefs() {
        std::array<geom::Vec3<PartFrame, Scalar>, N> a{};
        return a;
    }
} // namespace detail

template <class Scalar = MFloat>
class Thruster2T : public detail::ThrusterImpl<2, Scalar> {
public:
    using Base = detail::ThrusterImpl<2, Scalar>;
    using Vec  = typename Base::Vec;
    using Base::Base;

    Thruster2T(std::string name, Scalar max_thrust,
               Vec direction = Vec{Scalar(0), Scalar(0), Scalar(1)})
        : Base(std::move(name),
               detail::linear_only_force_coefs<2, Scalar>(direction, max_thrust),
               detail::zero_coefs<2, Scalar>()) {}
};

template <class Scalar = MFloat>
class Thruster3T : public detail::ThrusterImpl<3, Scalar> {
public:
    using Base = detail::ThrusterImpl<3, Scalar>;
    using Vec  = typename Base::Vec;
    using Base::Base;

    Thruster3T(std::string name, Scalar max_thrust,
               Vec direction = Vec{Scalar(0), Scalar(0), Scalar(1)})
        : Base(std::move(name),
               detail::linear_only_force_coefs<3, Scalar>(direction, max_thrust),
               detail::zero_coefs<3, Scalar>()) {}
};

template <class Scalar = MFloat>
class Thruster4T : public detail::ThrusterImpl<4, Scalar> {
public:
    using Base = detail::ThrusterImpl<4, Scalar>;
    using Vec  = typename Base::Vec;
    using Base::Base;

    Thruster4T(std::string name, Scalar max_thrust,
               Vec direction = Vec{Scalar(0), Scalar(0), Scalar(1)})
        : Base(std::move(name),
               detail::linear_only_force_coefs<4, Scalar>(direction, max_thrust),
               detail::zero_coefs<4, Scalar>()) {}
};

using Thruster1 = Thruster1T<MFloat>;
using Thruster2 = Thruster2T<MFloat>;
using Thruster3 = Thruster3T<MFloat>;
using Thruster4 = Thruster4T<MFloat>;

// Canonical name for the common 1st-order case.
template <class Scalar = MFloat> using ThrusterT = Thruster1T<Scalar>;
using Thruster = Thruster1;

} // namespace manta::parts
