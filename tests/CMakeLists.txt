include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(doctest)

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
    test_planet.cpp)

target_link_libraries(manta_tests PRIVATE
    manta::manta
    doctest::doctest)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(manta_tests)
