#include <doctest/doctest.h>

#include "../include/manta/geom/vec3.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using V = Vec3<WorldFrame, float>;

TEST_CASE("Vec3: default construction is zero") {
    V v;
    CHECK(v.x() == 0.0f);
    CHECK(v.y() == 0.0f);
    CHECK(v.z() == 0.0f);
    CHECK(v.norm() == 0.0f);
}

TEST_CASE("Vec3: zero() factory and componentwise constructor") {
    V z = V::zero();
    CHECK(test::approx_equal(z, V{0,0,0}));

    V v{1, 2, 3};
    CHECK(v.x() == 1.0f);
    CHECK(v.y() == 2.0f);
    CHECK(v.z() == 3.0f);
}

TEST_CASE("Vec3: addition and subtraction") {
    V a{1, 2, 3};
    V b{4, 5, 6};
    CHECK(test::approx_equal(a + b, V{5, 7, 9}));
    CHECK(test::approx_equal(b - a, V{3, 3, 3}));
    CHECK(test::approx_equal(-a,    V{-1, -2, -3}));
}

TEST_CASE("Vec3: compound assignment") {
    V a{1, 2, 3};
    a += V{1, 1, 1};
    CHECK(test::approx_equal(a, V{2, 3, 4}));
    a -= V{0, 1, 2};
    CHECK(test::approx_equal(a, V{2, 2, 2}));
}

TEST_CASE("Vec3: scalar multiply and divide") {
    V a{1, 2, 3};
    CHECK(test::approx_equal(a * 2.0f,    V{2, 4, 6}));
    CHECK(test::approx_equal(2.0f * a,    V{2, 4, 6}));
    CHECK(test::approx_equal(a / 2.0f,    V{0.5f, 1.0f, 1.5f}));
}

TEST_CASE("Vec3: dot, cross, norm") {
    V x{1, 0, 0};
    V y{0, 1, 0};
    V z{0, 0, 1};
    CHECK(x.dot(y) == 0.0f);
    CHECK(x.dot(x) == 1.0f);
    CHECK(test::approx_equal(x.cross(y),  z));
    CHECK(test::approx_equal(y.cross(z),  x));
    CHECK(test::approx_equal(z.cross(x),  y));
    CHECK(test::approx_equal(y.cross(x), -z));

    V v{3, 4, 0};
    CHECK(v.norm() == doctest::Approx(5.0f));
    CHECK(v.squared_norm() == doctest::Approx(25.0f));
}

TEST_CASE("Vec3: normalized") {
    V v{3, 4, 0};
    auto n = v.normalized();
    CHECK(n.norm() == doctest::Approx(1.0f));
    CHECK(test::approx_equal(n, V{0.6f, 0.8f, 0.0f}));
}

TEST_CASE("Vec3: cross product anticommutativity and self-cross is zero") {
    V a{1.5f, -2.0f, 3.7f};
    V b{0.4f,  5.1f, -1.2f};
    CHECK(test::approx_equal(a.cross(b), -b.cross(a)));
    CHECK(test::approx_equal(a.cross(a), V::zero()));
}

TEST_CASE("Vec3: from_raw and raw round-trip") {
    Eigen::Vector3f e{7, -8, 9};
    V v = V::from_raw(e);
    CHECK(v.raw() == e);
    CHECK(v.x() == 7.0f);
    CHECK(v.y() == -8.0f);
    CHECK(v.z() == 9.0f);
}

TEST_CASE("Vec3: frame identity wildcard passes") {
    // Default-constructed (id=0) is wildcard; mixing with a tagged vector
    // should not fire the debug assert.
    V tagged = V::from_raw({1, 2, 3}, FrameId{42});
    V wild;  // id=0
    CHECK(test::approx_equal(tagged + wild, V{1, 2, 3}));
}
