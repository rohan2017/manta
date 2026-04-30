#pragma once

#include <array>
#include <type_traits>

#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"

namespace manta::parts {

// A flat-plate / control-surface / drag-body abstraction parameterized by N
// velocity-power terms. See header notes below for the algorithm.
//
// Templated on (N, Scalar). The user-facing alias `Surface<N>` keeps the
// existing single-parameter shape: it's `SurfaceT<N, Real>`. The Scalar
// parameter activates for templated estimator crafts (Scalar = Jet).
//
// Required fields: FluidField (not Scalar-templated; queried via Real
// bridge — same pattern as Hull / PointBuoy / GravityPart).
template <int N, class Scalar = Real>
class SurfaceT : public PartT<Scalar> {
public:
    static_assert(N >= 1 && N <= 4, "SurfaceT<N>: N must be in [1, 4]");

    using Tensor = geom::Mat3<PartFrame, PartFrame, Scalar>;

    SurfaceT(std::string name,
             const std::array<Tensor, N>& force_tensors,
             const std::array<Tensor, N>& torque_tensors)
        : PartT<Scalar>(std::move(name))
        , A_(force_tensors)
        , B_(torque_tensors) {}

    const std::array<Tensor, N>& force_tensors()  const noexcept { return A_; }
    const std::array<Tensor, N>& torque_tensors() const noexcept { return B_; }

    void update() override {
        auto& fluid = this->template field<fields::FluidField>();

        // Bridge templated position to Real for the FluidField query.
        auto p_scaled = this->template position<SceneFrame>();
        Eigen::Matrix<Real, 3, 1> p_real;
        if constexpr (std::is_floating_point_v<Scalar>) {
            p_real = p_scaled.raw().template cast<Real>();
        } else {
            for (int i = 0; i < 3; ++i)
                p_real(i) = Real(p_scaled.raw()(i).a);
        }
        auto fs = fluid.state_at(geom::Vec3<SceneFrame>::from_raw(p_real));

        Eigen::Matrix<Scalar, 3, 1> v_fluid_scene =
            fs.velocity.raw().template cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> v_self_scene =
            this->template velocity<SceneFrame>().raw();

        // Relative velocity expressed in part frame.
        auto q_part_from_scene =
            this->template orientation<SceneFrame>().raw().conjugate();
        Eigen::Matrix<Scalar, 3, 1> v_rel_part =
            q_part_from_scene * (v_fluid_scene - v_self_scene);

        Eigen::Matrix<Scalar, 3, 1> F_part = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Eigen::Matrix<Scalar, 3, 1> T_part = Eigen::Matrix<Scalar, 3, 1>::Zero();
        Eigen::Matrix<Scalar, 3, 1> v_pow  = v_rel_part;   // v^1

        for (int k = 0; k < N; ++k) {
            F_part += A_[k].raw() * v_pow;
            T_part += B_[k].raw() * v_pow;
            if (k + 1 < N) {
                v_pow = v_pow.cwiseProduct(v_rel_part);
            }
        }

        this->apply_force_at(geom::Vec3<PartFrame, Scalar>::from_raw(F_part));
        this->apply_torque  (geom::Vec3<PartFrame, Scalar>::from_raw(T_part));
    }

private:
    std::array<Tensor, N> A_;
    std::array<Tensor, N> B_;
};

// Backwards-compat alias — `Surface<N>` keeps the existing one-template-arg
// signature for non-templated user code.
template <int N>
using Surface = SurfaceT<N, Real>;

} // namespace manta::parts
