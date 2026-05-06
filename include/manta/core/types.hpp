#pragma once

// Numeric types for manta.
//
// `MFloat` is the project's configurable floating-point type. Single
// precision by default; opt into double via the CMake option
// `MANTA_DOUBLE_PRECISION` (which defines `MANTA_DOUBLE_PRECISION=1` for
// every TU). Use `MFloat` for sim arithmetic that should follow the
// configured precision — craft state, planet pose, field parameters,
// integrator timestep, sensor noise.
//
// Where the precision is not negotiable, use the C++ type directly:
//
//   * `float`  — explicit 32-bit storage (e.g. wire formats that must be
//                bit-stable across processes regardless of build flags).
//   * `double` — algorithms that need 64-bit conditioning (Kalman
//                covariance, Ceres autodiff Jets, long-horizon clock
//                accumulation).
//
// All physics-relevant math types in manta are templated on `Scalar`
// with a default of `MFloat`, so the same class hierarchy can be
// instantiated at multiple precisions in one binary — that's how the
// EKF stands up a `WorldT<double>` and a `WorldT<ceres::Jet<double, N>>`
// alongside a `WorldT<MFloat>` sim.

namespace manta {

#if defined(MANTA_DOUBLE_PRECISION) && MANTA_DOUBLE_PRECISION
    using MFloat = double;
#else
    using MFloat = float;
#endif

} // namespace manta
