include(FetchContent)

# ---- zenoh-c (pre-built standalone binaries; no Rust toolchain required) ----
set(ZENOH_VERSION "1.9.0")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(ZENOHC_ARCH "x86_64-unknown-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ZENOHC_ARCH "aarch64-unknown-linux-gnu")
else()
    message(FATAL_ERROR "Unsupported arch for pre-built zenoh-c: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

FetchContent_Declare(
    zenohc_binary
    URL "https://github.com/eclipse-zenoh/zenoh-c/releases/download/${ZENOH_VERSION}/zenoh-c-${ZENOH_VERSION}-${ZENOHC_ARCH}-standalone.zip")
FetchContent_GetProperties(zenohc_binary)
if(NOT zenohc_binary_POPULATED)
    FetchContent_Populate(zenohc_binary)
endif()

find_package(zenohc REQUIRED CONFIG
    PATHS "${zenohc_binary_SOURCE_DIR}/lib/cmake/zenohc"
    NO_DEFAULT_PATH)

# ---- zenoh-cpp (header-only C++ wrappers around zenoh-c) ----
set(ZENOHCXX_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(ZENOHCXX_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZENOHCXX_ZENOHPICO       OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    zenohcxx
    GIT_REPOSITORY https://github.com/eclipse-zenoh/zenoh-cpp.git
    GIT_TAG        ${ZENOH_VERSION}
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(zenohcxx)

# ---- shared example helpers ----
add_library(manta_examples_shared INTERFACE)
target_include_directories(manta_examples_shared INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/shared")
target_link_libraries(manta_examples_shared INTERFACE
    manta::manta
    zenohcxx::zenohc)

# ---- examples ----
# All four examples are codegen-driven. To regenerate any of them, run from
# the repo root (replace the `--workflow` flag and craft.py path as needed):
#
#   PYTHONPATH=python/manta_codegen/src \
#       python -m manta_codegen.cli examples/<dir>/craft.py --workflow binary
#
# The generated <name>.cmake fragment defines the target; user-written main.cpp
# files (library workflow) are added by each example's CMake block below.

# ex0 — binary workflow (codegen emits the entire main).
include(${CMAKE_CURRENT_SOURCE_DIR}/ex0_free_flight/generated/ex0/ex0.cmake)

# ex1 — binary workflow (orbit, 200x realtime).
include(${CMAKE_CURRENT_SOURCE_DIR}/ex1_orbit/generated/ex1/ex1.cmake)

# ex2 — library workflow: codegen emits Ex2Craft + telemetry; user main
# does X-config mixing on the four Thruster1 rotors.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex2_quadcopter/generated/ex2/ex2.cmake)
add_executable(ex2_quadcopter
    ex2_quadcopter/ex2.cpp
    ${manta_ex2_SOURCES})
target_link_libraries(ex2_quadcopter PRIVATE manta_examples_shared)
manta_ex2_apply(ex2_quadcopter)

# ex3 — library workflow: codegen emits Ex3Craft (Mass + yaw_motor +
# pitch_motor + Thruster1 engine); user main runs gimbal position-PD +
# rate PIDs.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex3_tvc_rocket/generated/ex3/ex3.cmake)
add_executable(ex3_tvc_rocket
    ex3_tvc_rocket/ex3.cpp
    ${manta_ex3_SOURCES})
target_link_libraries(ex3_tvc_rocket PRIVATE manta_examples_shared)
manta_ex3_apply(ex3_tvc_rocket)

# ex4 — binary workflow: reaction wheel demo (Motor articulation).
include(${CMAKE_CURRENT_SOURCE_DIR}/ex4_reaction_wheel/generated/ex4/ex4.cmake)

# ex5 — binary workflow: underwater sub with Hull + OceanAtmosField + IMU + DVL.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex5_underwater/generated/ex5/ex5.cmake)

# ex6 — library workflow: sim Craft + templated estimator Craft + CraftEKF.
# Two codegen-emitted crafts: ex6 (sim) and ex6_est (templated estimator).
include(${CMAKE_CURRENT_SOURCE_DIR}/ex6_estimator_demo/generated/ex6/ex6.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/ex6_estimator_demo/generated/ex6_est/ex6_est.cmake)
add_executable(ex6_estimator_demo
    ex6_estimator_demo/ex6.cpp
    ${manta_ex6_SOURCES}
    ${manta_ex6_est_SOURCES})
target_link_libraries(ex6_estimator_demo PRIVATE manta_examples_shared)
manta_ex6_apply(ex6_estimator_demo)
manta_ex6_est_apply(ex6_estimator_demo)

# ex7 — Pattern C: real-data-only estimator (no sim Craft). Subscribes to
# external Zenoh topics, runs CraftEKF, publishes estimate.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex7_real_data_estimator/generated/ex7_est/ex7_est.cmake)
add_executable(ex7_real_data_estimator
    ex7_real_data_estimator/ex7.cpp
    ${manta_ex7_est_SOURCES})
target_link_libraries(ex7_real_data_estimator PRIVATE manta_examples_shared)
manta_ex7_est_apply(ex7_real_data_estimator)

# ex8 — multi-craft swarm: three Ex8Craft instances tethered in a chain.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex8_swarm/generated/ex8/ex8.cmake)
add_executable(ex8_swarm
    ex8_swarm/ex8.cpp
    ${manta_ex8_SOURCES})
target_link_libraries(ex8_swarm PRIVATE manta_examples_shared)
manta_ex8_apply(ex8_swarm)

# ex9 — binary workflow demo of the new explicit Binding pub/sub API
# (single-signal + bundled-struct on both directions). Codegen emits the
# entire main, including all publishers/subscribers, from craft.bindings.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex9_bindings_demo/generated/ex9/ex9.cmake)

# ex10 — multi-craft codegen: two distinct Craft types ticked by a single
# binary. Codegen emits per-Craft .hpp/.cpp + a world-level main.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex10_multi_craft/generated/ex10/ex10.cmake)

# ex11 — codegen-driven tether between two crafts. Python Tether(...) +
# World.add_tether(...) generates the C++ Tether instance + endpoint adds.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex11_tethered_pair/generated/ex11/ex11.cmake)

# sync_smoke — Zenoh field-sync round-trip test. Spins up two Sessions in
# one process, replicates a GravityField disturbance through the codegen's
# wire format, verifies the receiver rebuilds the same Disturbance.
add_executable(sync_smoke sync_smoke/sync_smoke.cpp)
target_link_libraries(sync_smoke PRIVATE manta_examples_shared)
add_test(NAME sync_smoke COMMAND sync_smoke)

# wire_debug — interactive Zenoh subscriber that decodes field-sync
# disturbance bytes for human inspection (gravity / mag / fluid).
add_executable(wire_debug wire_debug/wire_debug.cpp)
target_link_libraries(wire_debug PRIVATE manta_examples_shared)
