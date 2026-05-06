#pragma once

#include <type_traits>

#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/casts.hpp"
#include "../geom/vec3.hpp"

// Scalar-templated query helpers for the Vec3-output fields (Gravity, Mag).
//
// Fields store value-only disturbance lambdas (lifetime, in_influence,
// delta_g/delta_b). For estimator crafts that instantiate parts with
// `Scalar = ceres::Jet<double, N>` we still want the EKF/system-ID Jacobian
// to capture position-dependent field gradients (∂g/∂pos for point-mass
// gravity, ∂B/∂pos for dipole mag).
//
// The default path here is finite-difference around the value value:
//   * MFloat-Scalar: cast pos→MFloat, query f.state_at(pos), cast result→Scalar.
//   * Jet-Scalar:  evaluate at the value value plus 6 axis-aligned ±eps
//                  perturbations; build the Jet output as
//                      out_i.a = f_i(p),
//                      out_i.v = Σ_j (∂f_i/∂x_j) · pos_j.v
//                  with each ∂f_i/∂x_j approximated by central differences.
//
// 6 extra evaluations per Jet-step query — acceptable given fields are
// queried at most a few times per part-tick and Jet steps are themselves
// rarer than value steps.
//
// Future: a per-disturbance `state_at_jet` lambda would let the user
// supply an analytic gradient and bypass finite-diff. Plumbing for that
// can be added when a real consumer wants it; the helper signature here
// stays unchanged.
namespace manta::fields {

namespace detail {

// Default finite-diff step. Tuned conservatively — `MFloat = float` has
// ~7 decimal digits, so 1e-3 leaves ~4 digits of derivative accuracy,
// which is fine for EKF Jacobians at orbital scales (10^7 m positions).
template <class Scalar>
inline constexpr MFloat kFiniteDiffEps = MFloat(1e-3);

}  // namespace detail

// Query a Vec3-output field at a Scalar-templated position. Works for any
// field exposing `geom::Vec3<SceneFrame> state_at(geom::Vec3<SceneFrame>) const`.
//
// `Scalar` may be MFloat, double, or `ceres::Jet<double, N>`. For floating-point
// scalars, the result has zero derivatives; for Jet, the .v component is
// populated by chain-ruling through the field's spatial gradient.
template <class Scalar, class FieldT>
geom::Vec3<SceneFrame, Scalar>
state_at_templated(const FieldT& f,
                   const geom::Vec3<SceneFrame, Scalar>& pos) {
    using OutVec = geom::Vec3<SceneFrame, Scalar>;
    using RealVec = geom::Vec3<SceneFrame>;

    if constexpr (std::is_floating_point_v<Scalar>) {
        auto v_real = f.state_at(geom::cast_to_real(pos));
        return geom::lift_from_real<Scalar>(v_real);
    } else {
        // Jet path. Strip .a values, run 1 + 6 evaluations, compose.
        auto p_real = geom::cast_to_real(pos);
        Eigen::Matrix<MFloat, 3, 1> p0 = p_real.raw();
        auto v0 = f.state_at(p_real);

        MFloat eps = detail::kFiniteDiffEps<Scalar>;
        Eigen::Matrix<MFloat, 3, 3> J;
        for (int j = 0; j < 3; ++j) {
            auto pp = p0; pp(j) += eps;
            auto pm = p0; pm(j) -= eps;
            auto vp = f.state_at(RealVec::from_raw(pp));
            auto vm = f.state_at(RealVec::from_raw(pm));
            for (int i = 0; i < 3; ++i) {
                J(i, j) = (vp.raw()(i) - vm.raw()(i)) / (MFloat(2) * eps);
            }
        }

        Eigen::Matrix<Scalar, 3, 1> result;
        for (int i = 0; i < 3; ++i) {
            Scalar s;
            s.a = v0.raw()(i);
            s.v = J(i, 0) * pos.raw()(0).v
                + J(i, 1) * pos.raw()(1).v
                + J(i, 2) * pos.raw()(2).v;
            result(i) = s;
        }
        return OutVec::from_raw(result);
    }
}

}  // namespace manta::fields
