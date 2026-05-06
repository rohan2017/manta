cmake_minimum_required(VERSION 3.20)
project(manta
    VERSION 0.0.1
    DESCRIPTION "Lightweight C++ framework for large-scale robotics swarm simulation"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MANTA_BUILD_TESTS    "Build manta tests"    ON)
option(MANTA_BUILD_EXAMPLES "Build manta examples" ON)
option(MANTA_DEBUG_FRAMES   "Enable runtime FrameId checks" OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# Ceres is optional at this stage — only required for estimation-mode crafts.
find_package(Ceres QUIET)

add_library(manta STATIC
    src/core/frame.cpp
    src/core/noise.cpp
    src/parts/articulation/motor.cpp)
add_library(manta::manta ALIAS manta)

target_include_directories(manta PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(manta PUBLIC Eigen3::Eigen)

if(Ceres_FOUND)
    target_link_libraries(manta PUBLIC Ceres::ceres)
    target_compile_definitions(manta PUBLIC MANTA_HAVE_CERES=1)
endif()

if(MANTA_DEBUG_FRAMES)
    target_compile_definitions(manta PUBLIC MANTA_DEBUG_FRAMES=1)
endif()

if(MANTA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MANTA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
