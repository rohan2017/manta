#pragma once

#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/geom/kinematic_link.hpp"
#include "../include/manta/geom/mat3.hpp"
#include "../include/manta/geom/ori.hpp"
#include "../include/manta/geom/static_link.hpp"
#include "../include/manta/geom/vec3.hpp"

namespace manta::test {

constexpr float kTol      = 1e-5f;
constexpr float kLooseTol = 1e-3f;

template <typename Frame, typename Scalar>
inline bool approx_equal(const geom::Vec3<Frame, Scalar>& a,
                         const geom::Vec3<Frame, Scalar>& b,
                         Scalar tol = Scalar(kTol)) {
    return (a.raw() - b.raw()).norm() < tol;
}

template <typename Frame, typename Scalar>
inline bool approx_equal(const geom::Ori<Frame, Scalar>& a,
                         const geom::Ori<Frame, Scalar>& b,
                         Scalar tol = Scalar(kTol)) {
    // Compare as the two rotations being equivalent (q and -q represent the
    // same rotation).
    auto qa = a.raw().normalized();
    auto qb = b.raw().normalized();
    Scalar d  = std::abs(qa.dot(qb));
    return std::abs(Scalar(1) - d) < tol;
}

template <typename From, typename To, typename Scalar>
inline bool approx_equal(const geom::Mat3<From, To, Scalar>& a,
                         const geom::Mat3<From, To, Scalar>& b,
                         Scalar tol = Scalar(kTol)) {
    return (a.raw() - b.raw()).norm() < tol;
}

} // namespace manta::test
