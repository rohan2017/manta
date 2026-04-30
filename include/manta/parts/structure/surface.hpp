#pragma once

#include <array>

#include "../../core/craft.hpp"
#include "../../fields/fluid_field.hpp"

namespace manta::parts {

// A flat-plate / control-surface / drag-body abstraction parameterized by N
// velocity-power terms.
//
// Each tick the part queries the local fluid velocity (currents/wind) at its
// scene-frame position and computes the relative velocity in part frame:
//
//     v_rel_part = R_part_from_scene * (v_fluid_scene - v_self_scene)
//
// For each k ∈ [1, N], the part applies:
//
//     F_part += A_k * v_rel_part^(k)        (force)
//     τ_part += B_k * v_rel_part^(k)        (torque about part origin)
//
// where v^(k) is the elementwise k-th power of v_rel_part. A_1 / B_1 are
// linear (Stokes drag, bound vortex lift). A_2 / B_2 give quadratic drag /
// dynamic-pressure lift. Higher orders are generally overkill; N <= 3
// covers the practical aerospace/marine cases.
//
// This generalizes the simple linear-drag-only "Surface" — Surface<1>
// reproduces it. The class is templated on N at compile time so each surface
// instance keeps a fixed-size tensor stack with no heap allocation.
//
// Required fields: FluidField.
template <int N>
class Surface : public Part {
public:
    static_assert(N >= 1 && N <= 4, "Surface<N>: N must be in [1, 4]");

    using Tensor = geom::Mat3<PartFrame>;

    Surface(std::string name,
            const std::array<Tensor, N>& force_tensors,
            const std::array<Tensor, N>& torque_tensors)
        : Part(std::move(name))
        , A_(force_tensors)
        , B_(torque_tensors) {}

    const std::array<Tensor, N>& force_tensors()  const noexcept { return A_; }
    const std::array<Tensor, N>& torque_tensors() const noexcept { return B_; }

    void update() override {
        auto& fluid = field<fields::FluidField>();
        auto p_scene = position<SceneFrame>();
        auto v_self  = velocity<SceneFrame>();
        auto fs      = fluid.state_at(p_scene);

        // Relative velocity of the surface w.r.t. the fluid, expressed in
        // PartFrame.
        auto q_part_from_scene = orientation<SceneFrame>().raw().conjugate();
        auto v_rel_scene_raw   = fs.velocity.raw() - v_self.raw();
        auto v_rel_part_raw    = q_part_from_scene * v_rel_scene_raw;

        // Accumulate force and torque across all N orders.
        Eigen::Matrix<Real, 3, 1> F_part = Eigen::Matrix<Real, 3, 1>::Zero();
        Eigen::Matrix<Real, 3, 1> T_part = Eigen::Matrix<Real, 3, 1>::Zero();
        Eigen::Matrix<Real, 3, 1> v_pow  = v_rel_part_raw;   // v^1

        for (int k = 0; k < N; ++k) {
            F_part += A_[k].raw() * v_pow;
            T_part += B_[k].raw() * v_pow;
            if (k + 1 < N) {
                // elementwise multiply for v^(k+1) = v^k ⊙ v
                v_pow = v_pow.cwiseProduct(v_rel_part_raw);
            }
        }

        apply_force_at (geom::Vec3<PartFrame>::from_raw(F_part));
        apply_torque   (geom::Vec3<PartFrame>::from_raw(T_part));
    }

private:
    std::array<Tensor, N> A_;   // force coefficients,  one per power 1..N
    std::array<Tensor, N> B_;   // torque coefficients, one per power 1..N
};

} // namespace manta::parts
