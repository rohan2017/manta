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
# Codegen examples regenerate from the repo root (replace --workflow and
# the craft.py path):
#
#   PYTHONPATH=python/manta_codegen/src \
#       python -m manta_codegen.cli examples/<dir>/craft.py --workflow binary
#
# The generated <name>.cmake fragment defines the target; user-written main.cpp
# files (library workflow) are added by each example's CMake block below.

# ex0 — pure C++, no codegen, no Zenoh. Drag-comparison freefall.
add_executable(ex0_free_flight ex0_free_flight/ex0.cpp)
target_link_libraries(ex0_free_flight PRIVATE manta::manta)

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

# ex5 — Pattern A: sim Craft + templated estimator Craft + CraftEKF in one
# binary. The codegen-emitted main wires sim sensor outputs into the EKF
# craft's set_measurement hooks each tick via cross-world connect(), runs
# both worlds in lockstep, and publishes truth + estimate to Zenoh.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex5_estimator_demo/generated/ex5/ex5.cmake)
target_link_libraries(ex5 PRIVATE manta_examples_shared)

# ex6 — Pattern C: real-data-only estimator (no sim Craft). Codegen-driven
# EKF target: include() defines the `ex6_est` binary directly; the
# generated _main.cpp wraps `manta::estimation::CraftEKF<Ex6EstCraftT, 3>`,
# subscribes to external Zenoh sensor topics, publishes the estimate.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex6_real_data_estimator/generated/ex6_est/ex6_est.cmake)
target_link_libraries(ex6_est PRIVATE manta_examples_shared)

# ex7 — codegen-driven tether between two crafts. Python Tether(...) +
# World.add_tether(...) generates the C++ Tether instance + endpoint adds.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex7_tethered_pair/generated/ex7/ex7.cmake)

# connect_demo — in-process signal-to-signal binding demo. One craft with
# two thrusters; follower mirrors leader via w.connect() (no Zenoh).
include(${CMAKE_CURRENT_SOURCE_DIR}/connect_demo/generated/connect_demo/connect_demo.cmake)

# ukf_smoke — minimal UKF codegen smoke target. Mass + IMU + DVL on a
# plain (non-templated) craft, wrapped in CraftUKFOf<...>. Catches
# regressions in the UKF emit path.
include(${CMAKE_CURRENT_SOURCE_DIR}/ukf_smoke/generated/ukf_smoke/ukf_smoke.cmake)
target_link_libraries(ukf_smoke PRIVATE manta_examples_shared)

# ex8 — submarine: AUV with full sensor suite (IMU + DVL + Magnetometer)
# + buoyancy + drag + 2× thruster, sim+EKF in one binary.
include(${CMAKE_CURRENT_SOURCE_DIR}/ex8_submarine/generated/ex8/ex8.cmake)
target_link_libraries(ex8 PRIVATE manta_examples_shared)

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
