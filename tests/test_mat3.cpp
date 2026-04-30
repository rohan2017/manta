#include <doctest/doctest.h>

#include "../include/manta/geom/mat3.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using M = Mat3<WorldFrame, WorldFrame, float>;
using V = Vec3<WorldFrame, float>;

TEST_CASE("Mat3: default and zero are equivalent") {
    M a;
    M z = M::zero();
    CHECK(test::approx_equal(a, z));
    CHECK(a.raw()(0,0) == 0.0f);
    CHECK(a.raw()(2,2) == 0.0f);
}

TEST_CASE("Mat3: identity acts as identity on vectors") {
    M id = M::identity();
    V v{3, -1, 7};
    CHECK(test::approx_equal(id * v, v));
}

TEST_CASE("Mat3: from_diagonal scales each axis") {
    M d = M::from_diagonal(2, 3, 5);
    V v{1, 1, 1};
    CHECK(test::approx_equal(d * v, V{2, 3, 5}));
}

TEST_CASE("Mat3: matrix-matrix product is associative on identity") {
    M id = M::identity();
    M a{(Eigen::Matrix3f() << 1,2,3, 4,5,6, 7,8,10).finished()};
    CHECK(test::approx_equal(id * a, a));
    CHECK(test::approx_equal(a * id, a));
}

TEST_CASE("Mat3: transpose round-trips") {
    M a{(Eigen::Matrix3f() << 1,2,3, 4,5,6, 7,8,10).finished()};
    auto at = a.transpose();
    auto att = at.transpose();
    CHECK(test::approx_equal(a, att));
}

TEST_CASE("Mat3: scalar multiply") {
    M id = M::identity();
    M two = id * 2.0f;
    V v{1, 1, 1};
    CHECK(test::approx_equal(two * v, V{2, 2, 2}));
}

TEST_CASE("Mat3: addition and subtraction") {
    M id = M::identity();
    M two = id + id;
    V v{1, 2, 3};
    CHECK(test::approx_equal(two * v, V{2, 4, 6}));
    M zero = id - id;
    CHECK(test::approx_equal(zero * v, V::zero()));
}
