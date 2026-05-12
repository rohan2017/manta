#pragma once

// Feature-macro defaults + the part-side convenience macros for declaring
// field requirements / optional augmentations.
//
// The codegen emits a per-target `<world>_config.h` that `#define`s
// `MANTA_HAS_<FIELD> 1` for every field registered with the wrapped
// world. CMake force-includes that config header before any source file
// is compiled, so by the time this `features.hpp` is included by a part
// header, the codegen-set macros are already in place. The `#ifndef`s
// below fill in `0` for any feature the codegen *didn't* set, giving
// every part header a consistent, well-defined symbol it can compare
// against.
//
// Once every macro is guaranteed to be 0 or 1, parts express their
// dependencies with one of two patterns:
//
//   1. REQUIRED — the part is meaningless without the field. Fail the
//      build with an explanatory message at template-instantiation
//      time. Use `MANTA_PART_REQUIRES_FIELD` inside the part class
//      body or `update()`:
//
//          template <class Scalar = MFloat>
//          class MagnetometerT : public PartT<Scalar> {
//              MANTA_PART_REQUIRES_FIELD(MANTA_HAS_MAG_FIELD,
//                  "Magnetometer requires a MagField on the world. "
//                  "Register one with World.add_field(MagField(...)).");
//              // ... rest of class
//          };
//
//      The Python-side codegen ALSO validates `requires_fields` at
//      config time, so the user normally sees a friendly error in
//      their config.py before C++ compilation even starts. The
//      `static_assert` is defense-in-depth.
//
//   2. OPTIONAL — the part has bonus functionality if the field is
//      present (e.g. IMU subtracts gravity from the body acceleration
//      to report specific force when a GravityField exists; otherwise
//      it just reports the kinematic body accel). Use
//      `MANTA_PART_AUGMENTS_FIELD` for an `if constexpr` block:
//
//          if constexpr (MANTA_PART_AUGMENTS_FIELD(MANTA_HAS_GRAVITY_FIELD)) {
//              // gravity-aware path
//          } else {
//              // gravity-free path
//          }
//
//      No code from the inactive branch is emitted; field-registry
//      traffic for absent fields is fully compiled out.

#include <type_traits>

#include "types.hpp"   // pulls in Scalar/MFloat machinery for downstream parts

// ---- Default values for all known feature macros ----
//
// Codegen-emitted config.h overrides these to `1` for registered fields.
// New field types should add their default here so part headers can
// reference the symbol unconditionally.

#ifndef MANTA_HAS_GRAVITY_FIELD
#define MANTA_HAS_GRAVITY_FIELD 0
#endif

#ifndef MANTA_HAS_MAG_FIELD
#define MANTA_HAS_MAG_FIELD 0
#endif

#ifndef MANTA_HAS_FLUID_FIELD
#define MANTA_HAS_FLUID_FIELD 0
#endif

#ifndef MANTA_HAS_COLLISION_FIELD
#define MANTA_HAS_COLLISION_FIELD 0
#endif

// ---- Part-side declaration macros ----

namespace manta::detail {
// Dependent-false template — used to defer `static_assert` evaluation
// until the enclosing template is instantiated. Naming a dependent
// member of `T` makes the value depend on T, so the assert fires only
// when a concrete part class is instantiated, not at declaration.
template <class T> struct dependent_false : std::false_type {};
} // namespace manta::detail

// MANTA_PART_REQUIRES_FIELD(feature_macro, message)
//   Hard requirement. Fires `static_assert` at template instantiation
//   time with the given message if the feature macro is 0.
#define MANTA_PART_REQUIRES_FIELD(FEATURE, MESSAGE)                      \
    static_assert((FEATURE) || ::manta::detail::dependent_false<Scalar>::value, MESSAGE)

// MANTA_PART_AUGMENTS_FIELD(feature_macro)
//   Yields a constexpr boolean for use inside `if constexpr (...)`.
//   The `else` branch compiles to nothing when the feature is enabled,
//   and the `then` branch compiles to nothing when it isn't — no
//   runtime field-registry traffic in either case.
#define MANTA_PART_AUGMENTS_FIELD(FEATURE) ((FEATURE) != 0)
