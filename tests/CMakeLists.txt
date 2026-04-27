include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(mantapilot_tests
    test_main.cpp
    test_vec3.cpp
    test_ori.cpp
    test_mat3.cpp
    test_wrench.cpp
    test_static_link.cpp
    test_kinematic_link.cpp)

target_link_libraries(mantapilot_tests PRIVATE
    mantapilot::mantapilot
    doctest::doctest)

target_compile_definitions(mantapilot_tests PRIVATE MANTAPILOT_DEBUG_FRAMES=1)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(mantapilot_tests)
