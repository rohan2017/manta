cmake_minimum_required(VERSION 3.20)
project(mantapilot
    VERSION 0.0.1
    DESCRIPTION "Lightweight C++ framework for large-scale robotics swarm simulation"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MANTAPILOT_BUILD_TESTS    "Build mantapilot tests"    ON)
option(MANTAPILOT_BUILD_EXAMPLES "Build mantapilot examples" ON)
option(MANTAPILOT_DEBUG_FRAMES   "Enable runtime FrameId checks" ON)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# Ceres is optional at this stage — only required for estimation-mode crafts.
find_package(Ceres QUIET)

add_library(mantapilot INTERFACE)
add_library(mantapilot::mantapilot ALIAS mantapilot)

target_include_directories(mantapilot INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(mantapilot INTERFACE Eigen3::Eigen)

if(Ceres_FOUND)
    target_link_libraries(mantapilot INTERFACE Ceres::ceres)
    target_compile_definitions(mantapilot INTERFACE MANTAPILOT_HAVE_CERES=1)
endif()

if(MANTAPILOT_DEBUG_FRAMES)
    target_compile_definitions(mantapilot INTERFACE
        $<$<CONFIG:Debug>:MANTAPILOT_DEBUG_FRAMES=1>)
endif()

if(MANTAPILOT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MANTAPILOT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
