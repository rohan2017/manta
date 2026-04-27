#pragma once

// Numeric type machinery for mantapilot.
//
// mFloat is the primary scalar used throughout the framework. It is chosen at
// compile time per craft binary:
//
//   - Sim mode (default): mFloat = float
//   - Estimation mode:    mFloat = ceres::Jet<double, MANTAPILOT_STATE_DIM>
//
// To build a craft binary in estimation mode, define both
// MANTAPILOT_ESTIMATION_MODE and MANTAPILOT_STATE_DIM (the codegen-determined
// state vector size) before including any mantapilot header.
//
// All physics-relevant math types in mantapilot are templated on Scalar with a
// default of mFloat, so a single binary can also mix scalars per-craft if
// needed (e.g. one craft in float, another in Jet).

#if defined(MANTAPILOT_ESTIMATION_MODE)
    #if !defined(MANTAPILOT_STATE_DIM)
        #error "MANTAPILOT_ESTIMATION_MODE requires MANTAPILOT_STATE_DIM to be defined."
    #endif
    #if !defined(MANTAPILOT_HAVE_CERES)
        #error "MANTAPILOT_ESTIMATION_MODE requires Ceres Solver."
    #endif
    #include <ceres/jet.h>
#endif

namespace manta {

#if defined(MANTAPILOT_ESTIMATION_MODE)
    using mFloat = ceres::Jet<double, MANTAPILOT_STATE_DIM>;
    inline constexpr bool kEstimationMode = true;
    inline constexpr int  kStateDim       = MANTAPILOT_STATE_DIM;
#else
    using mFloat = float;
    inline constexpr bool kEstimationMode = false;
    inline constexpr int  kStateDim       = 0;
#endif

} // namespace manta
