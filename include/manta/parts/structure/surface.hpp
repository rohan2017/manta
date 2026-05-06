#pragma once

#include <array>

#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../geom/casts.hpp"

namespace manta::parts {

// Flat-plate / control-surface / drag-body abstraction. Each tick computes
// v_rel (fluid velocity − own velocity) in part frame and accumulates:
//     F = sum_{k=1..N} A_k * v_rel^(k)
//     τ = sum_{k=1..N} B_k * v_rel^(k)
// where v^(k) means component-wise k-th power.
//
// Concrete user-facing classes are `Surface1`..`Surface4` (and Scalar-
// templated `Surface1T<Scalar>`..`Surface4T<Scalar>` for estimator use).
// `Surface1` is plain linear drag/lift; `Surface2` adds a quadratic term;
// higher orders are typically overkill (capped at 4).
//
// Required fields: FluidField (not Scalar-templated; queried via Real
// bridge — same pattern as Hull / PointBuoy / GravityPart).
namespace detail {

template <int N, class Scalar>
class SurfaceImpl : public PartT<Scalar> {
public:
    static_assert(N >= 1 && N <= 4, "SurfaceImpl<N>: N must be in [1, 4]");

    using Tensor = geom::Mat3<PartFrame, PartFrame, Scalar>;

    SurfaceImpl(std::string name,
                const std::array<Tensor, N>& force_tensors,
                const std::array<Tensor, N>& torque_tensors)
        : PartT<Scalar>(std::move(name))
        , A_(force_tensors)
        , B_(torque_tensors) {}

    const std::array<Tensor, N>& force_tensors()  const noexcept { return A_; }
    const std::array<Tensor, N>& torque_tensors() const noexcept { return B_; }

    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_FLUID_FIELD,
        "Surface requires a FluidField on the world (drag/lift are "
        "ρ-scaled). Register one with World.add_field(FluidField(...)), "
        "or remove the Surface part.");

    void update() override {
        auto& fluid = this->template field<fields::FluidField>();

        // Bridge templated position to Real for the FluidField query.
        auto p_scaled = this->template position<SceneFrame>();
        auto fs       = fluid.state_at(geom::cast_to_real(p_scaled));

        Eigen::Matrix<Scalar, 3, 1> v_fluid_scene =
            fs.velocity.raw().template cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> v_self_scene =
            this->template velocity<SceneFrame>().raw();

        // Relative velocity expressed in part frame.
        auto q_part_from_scene =
            this->template orientation<SceneFrame>().raw().conjugate();
        Eigen::Matrix<Scalar, 3, 1> v_rel_part =
            q_part_from_scene * (v_fluid_scene - v_self_scene);

        // Per-order multiplier for component i is sign(v_i) · |v_i|^k —
        // i.e. v_i · |v_i|^(k-1). Magnitude grows like the k-th power but
        // the sign is preserved, so a positive A_k tensor produces force
        // pointing along v_rel (drag-like) regardless of which way the
        // fluid is flowing relative to the body. Even-order terms with
        // raw cwiseProduct would lose the sign and break this — that
        // bug used to mean Surface2's quadratic term flipped its
        // direction on a sign change.
        Eigen::Matrix<Scalar, 3, 1> F_part = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Eigen::Matrix<Scalar, 3, 1> T_part = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Eigen::Matrix<Scalar, 3, 1> v_pow  = v_rel_part;     // sign(v) · |v|^1
        Eigen::Matrix<Scalar, 3, 1> abs_v  = v_rel_part.cwiseAbs();

        for (int k = 0; k < N; ++k) {
            F_part += A_[k].raw() * v_pow;
            T_part += B_[k].raw() * v_pow;
            if (k + 1 < N) {
                v_pow = v_pow.cwiseProduct(abs_v);           // → sign(v) · |v|^(k+1)
            }
        }

        this->apply_force_at(geom::Vec3<PartFrame, Scalar>::from_raw(F_part));
        this->apply_torque  (geom::Vec3<PartFrame, Scalar>::from_raw(T_part));
    }

private:
    std::array<Tensor, N> A_;
    std::array<Tensor, N> B_;
};

} // namespace detail

// Concrete user-facing classes — one template parameter (Scalar) each.
template <class Scalar = Real> class Surface1T : public detail::SurfaceImpl<1, Scalar> {
public: using detail::SurfaceImpl<1, Scalar>::SurfaceImpl;
};
template <class Scalar = Real> class Surface2T : public detail::SurfaceImpl<2, Scalar> {
public: using detail::SurfaceImpl<2, Scalar>::SurfaceImpl;
};
template <class Scalar = Real> class Surface3T : public detail::SurfaceImpl<3, Scalar> {
public: using detail::SurfaceImpl<3, Scalar>::SurfaceImpl;
};
template <class Scalar = Real> class Surface4T : public detail::SurfaceImpl<4, Scalar> {
public: using detail::SurfaceImpl<4, Scalar>::SurfaceImpl;
};

using Surface1 = Surface1T<Real>;
using Surface2 = Surface2T<Real>;
using Surface3 = Surface3T<Real>;
using Surface4 = Surface4T<Real>;

} // namespace manta::parts
