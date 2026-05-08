include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(doctest)

# ---- Codegen-driven signal sync test ----
# Regenerated at configure time from each part descriptor's `signals` table.
# Compile failure here = a Python signal expression no longer matches the C++
# part API.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_signal_sync_test_out
    "${CMAKE_CURRENT_BINARY_DIR}/test_codegen_signals.gen.cpp")
set(_signal_sync_script
    "${CMAKE_SOURCE_DIR}/python/manta_codegen/scripts/emit_signal_sync_test.py")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${CMAKE_SOURCE_DIR}/python/manta_codegen/src"
        "${Python3_EXECUTABLE}" "${_signal_sync_script}" "${_signal_sync_test_out}"
    RESULT_VARIABLE _sync_rc)
if(NOT _sync_rc EQUAL 0)
    message(FATAL_ERROR "emit_signal_sync_test.py failed (exit ${_sync_rc})")
endif()

add_executable(manta_tests
    test_main.cpp
    test_vec3.cpp
    test_ori.cpp
    test_mat3.cpp
    test_wrench.cpp
    test_static_link.cpp
    test_kinematic_link.cpp
    test_craft.cpp
    test_craft_state.cpp
    test_noise_registry.cpp
    test_random_walk_bias.cpp
    test_noise.cpp
    test_parts.cpp
    test_world.cpp
    test_orbit.cpp
    test_tether.cpp
    test_fluid.cpp
    test_field.cpp
    test_field_templated.cpp
    test_field_sync.cpp
    test_stock_parts.cpp
    test_articulation.cpp
    test_ekf.cpp
    test_ekf_against_sim.cpp
    test_manifold.cpp
    test_generic_ekf.cpp
    test_generic_ukf.cpp
    test_craft_view.cpp
    test_measurement_e2e.cpp
    test_ukf.cpp
    test_planet.cpp
    test_system_id.cpp
    test_parameter_fit.cpp
    test_rate_gate.cpp
    "${_signal_sync_test_out}")

target_link_libraries(manta_tests PRIVATE
    manta::manta
    doctest::doctest)

# The hand-written tests exercise every part — Mass under gravity,
# Magnetometer in a MagField, PointBuoy in a FluidField, etc. — so they
# need every feature macro that codegen would normally emit into a
# target-specific config.h. Define them all up front; the same TU mixes
# tests for crafts with and without each field, so the runtime
# `world().field_ptr(...)` registry already gates the inactive paths.
target_compile_definitions(manta_tests PRIVATE
    MANTA_HAS_GRAVITY_FIELD=1
    MANTA_HAS_MAG_FIELD=1
    MANTA_HAS_FLUID_FIELD=1)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(manta_tests)
