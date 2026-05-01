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
    test_noise.cpp
    test_parts.cpp
    test_world.cpp
    test_orbit.cpp
    test_tether.cpp
    test_fluid.cpp
    test_stock_parts.cpp
    test_articulation.cpp
    test_ekf.cpp
    test_ekf_against_sim.cpp
    test_ukf.cpp
    test_planet.cpp
    "${_signal_sync_test_out}")

target_link_libraries(manta_tests PRIVATE
    manta::manta
    doctest::doctest)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(manta_tests)
