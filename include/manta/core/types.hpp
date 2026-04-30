#pragma once

// Numeric type machinery for manta.
//
// Real is the primary scalar used throughout the framework. It is chosen at
// compile time per craft binary:
//
//   - Sim mode (default): Real = float
//   - Estimation mode:    Real = ceres::Jet<double, MANTA_STATE_DIM>
//
// To build a craft binary in estimation mode, define both
// MANTA_ESTIMATION_MODE and MANTA_STATE_DIM (the codegen-determined
// state vector size) before including any manta header.
//
// All physics-relevant math types in manta are templated on Scalar with a
// default of Real, so a single binary can also mix scalars per-craft if
// needed (e.g. one craft in float, another in Jet).

#if defined(MANTA_ESTIMATION_MODE)
    #if !defined(MANTA_STATE_DIM)
        #error "MANTA_ESTIMATION_MODE requires MANTA_STATE_DIM to be defined."
    #endif
    #if !defined(MANTA_HAVE_CERES)
        #error "MANTA_ESTIMATION_MODE requires Ceres Solver."
    #endif
    #include <ceres/jet.h>
#endif

namespace manta {

#if defined(MANTA_ESTIMATION_MODE)
    using Real = ceres::Jet<double, MANTA_STATE_DIM>;
    inline constexpr bool kEstimationMode = true;
    inline constexpr int  kStateDim       = MANTA_STATE_DIM;
#else
    using Real = float;
    inline constexpr bool kEstimationMode = false;
    inline constexpr int  kStateDim       = 0;
#endif

} // namespace manta
