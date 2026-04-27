#include <doctest/doctest.h>

#include "../include/mantapilot/geom/ori.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using O = Ori<WorldFrame, float>;
using V = Vec3<WorldFrame, float>;

constexpr float kPi = 3.14159265358979323846f;

TEST_CASE("Ori: default and identity are equivalent") {
    O a;
    O id = O::identity();
    CHECK(test::approx_equal(a, id));
    CHECK(a.raw().w() == doctest::Approx(1.0f));
    CHECK(a.raw().x() == doctest::Approx(0.0f));
    CHECK(a.raw().y() == doctest::Approx(0.0f));
    CHECK(a.raw().z() == doctest::Approx(0.0f));
}

TEST_CASE("Ori: from_axis_angle around z by 90deg") {
    O r = O::from_axis_angle(V{0, 0, 1}, kPi / 2.0f);
    // Quaternion for 90deg about z: (cos(45), 0, 0, sin(45)).
    CHECK(r.raw().w() == doctest::Approx(std::cos(kPi / 4.0f)));
    CHECK(r.raw().x() == doctest::Approx(0.0f));
    CHECK(r.raw().y() == doctest::Approx(0.0f));
    CHECK(r.raw().z() == doctest::Approx(std::sin(kPi / 4.0f)));
}

TEST_CASE("Ori: identity composed with anything is identity") {
    O r  = O::from_axis_angle(V{1, 1, 1}, 0.7f);
    O id = O::identity();
    CHECK(test::approx_equal(id * r, r));
    CHECK(test::approx_equal(r * id, r));
}

TEST_CASE("Ori: r * r.inverse() is identity") {
    O r = O::from_axis_angle(V{0.3f, -0.5f, 0.8f}, 1.234f);
    CHECK(test::approx_equal(r * r.inverse(), O::identity()));
    CHECK(test::approx_equal(r.inverse() * r, O::identity()));
}

TEST_CASE("Ori: composition of two rotations about same axis adds angles") {
    O a = O::from_axis_angle(V{0, 0, 1}, 0.7f);
    O b = O::from_axis_angle(V{0, 0, 1}, 0.4f);
    O c = O::from_axis_angle(V{0, 0, 1}, 1.1f);
    CHECK(test::approx_equal(a * b, c));
    CHECK(test::approx_equal(b * a, c));  // commutes for same-axis rotations
}

TEST_CASE("Ori: small rotations do not commute when axes differ") {
    O ax = O::from_axis_angle(V{1, 0, 0}, 1.2f);
    O ay = O::from_axis_angle(V{0, 1, 0}, 0.9f);
    // Different orderings should produce different orientations.
    CHECK_FALSE(test::approx_equal(ax * ay, ay * ax));
}

TEST_CASE("Ori: normalize produces unit quaternion") {
    Eigen::Quaternionf q{2.0f, 0.0f, 0.0f, 0.0f};
    O r{q};
    CHECK(r.raw().norm() == doctest::Approx(2.0f));
    auto rn = r.normalized();
    CHECK(rn.raw().norm() == doctest::Approx(1.0f));
}
