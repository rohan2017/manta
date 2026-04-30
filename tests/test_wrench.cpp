#include <doctest/doctest.h>

#include "../include/manta/core/wrench.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;
using W = Wrench<WorldFrame, float>;
using V = Vec3<WorldFrame, float>;

TEST_CASE("Wrench: default and zero are equivalent") {
    W a;
    W z = W::zero();
    CHECK(test::approx_equal(a.force(),  z.force()));
    CHECK(test::approx_equal(a.torque(), z.torque()));
}

TEST_CASE("Wrench: addition is componentwise") {
    W a{V{1,2,3}, V{4,5,6}};
    W b{V{6,5,4}, V{3,2,1}};
    auto c = a + b;
    CHECK(test::approx_equal(c.force(),  V{7, 7, 7}));
    CHECK(test::approx_equal(c.torque(), V{7, 7, 7}));
}

TEST_CASE("Wrench: from_force_at gives torque = r x F") {
    V force{0, 0, 10};        // upward force
    V point{1, 0, 0};         // applied 1m to the +x of the origin
    W w = W::from_force_at(force, point);
    CHECK(test::approx_equal(w.force(), force));
    // r x F = (1,0,0) x (0,0,10) = (0, -10, 0)
    CHECK(test::approx_equal(w.torque(), V{0, -10, 0}));
}

TEST_CASE("Wrench: from_force_at at origin produces no torque") {
    V force{1, 2, 3};
    W w = W::from_force_at(force, V::zero());
    CHECK(test::approx_equal(w.force(),  force));
    CHECK(test::approx_equal(w.torque(), V::zero()));
}

TEST_CASE("Wrench: from_torque has zero force") {
    V torque{0, 0, 7};
    W w = W::from_torque(torque);
    CHECK(test::approx_equal(w.force(),  V::zero()));
    CHECK(test::approx_equal(w.torque(), torque));
}

TEST_CASE("Wrench: negation flips both components") {
    W a{V{1, 2, 3}, V{4, 5, 6}};
    auto n = -a;
    CHECK(test::approx_equal(n.force(),  V{-1, -2, -3}));
    CHECK(test::approx_equal(n.torque(), V{-4, -5, -6}));
}

TEST_CASE("Wrench: compound addition") {
    W a{V{1, 0, 0}, V{0, 1, 0}};
    W b{V{0, 1, 0}, V{0, 0, 1}};
    a += b;
    CHECK(test::approx_equal(a.force(),  V{1, 1, 0}));
    CHECK(test::approx_equal(a.torque(), V{0, 1, 1}));
}
