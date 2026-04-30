#include <doctest/doctest.h>

#include "../include/manta/core/frame.hpp"
#include "../include/manta/geom/static_link.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;

constexpr float kPi = 3.14159265358979323846f;

// Two distinct part-frame tags by reusing the FrameKind machinery; for
// transform composition tests we just need three distinct tag types.
struct A { static constexpr FrameKind kind = FrameKind::World; };
struct B { static constexpr FrameKind kind = FrameKind::Scene; };
struct C { static constexpr FrameKind kind = FrameKind::Craft; };

using SAB = StaticLink<A, B, float>;
using SBC = StaticLink<B, C, float>;
using SBA = StaticLink<B, A, float>;

using VA = Vec3<A, float>;
using VB = Vec3<B, float>;

TEST_CASE("StaticLink: identity has zero position and identity orientation") {
    auto id = SAB::identity();
    CHECK(test::approx_equal(id.position(), VA::zero()));
    CHECK(test::approx_equal(id.orientation(), Ori<A, float>::identity()));
}

TEST_CASE("StaticLink: identity transforms a position to itself (after frame change)") {
    auto id = SAB::identity();
    VB p_b{1, 2, 3};
    auto p_a = id.apply_position(p_b);
    CHECK(test::approx_equal(p_a, VA{1, 2, 3}));
}

TEST_CASE("StaticLink: pure translation moves the point but does not rotate it") {
    SAB tf{VA{10, 0, 0}, Ori<A, float>::identity()};
    VB p_b{1, 2, 3};
    auto p_a = tf.apply_position(p_b);
    CHECK(test::approx_equal(p_a, VA{11, 2, 3}));

    // Pure rotation (no offset) under a translation-only transform
    // is just the rotation portion (identity), so the vector is unchanged.
    auto v_a = tf.rotate(p_b);
    CHECK(test::approx_equal(v_a, VA{1, 2, 3}));
}

TEST_CASE("StaticLink: pure rotation about z by 90deg") {
    auto rot_z = Ori<A, float>::from_axis_angle(VA{0, 0, 1}, kPi / 2.0f);
    SAB tf{VA::zero(), rot_z};
    VB x_b{1, 0, 0};
    auto x_a = tf.apply_position(x_b);
    CHECK(test::approx_equal(x_a, VA{0, 1, 0}));

    auto y_a = tf.apply_position(VB{0, 1, 0});
    CHECK(test::approx_equal(y_a, VA{-1, 0, 0}));
}

TEST_CASE("StaticLink: rotate vs apply_position for translated frame") {
    auto rot_z = Ori<A, float>::from_axis_angle(VA{0, 0, 1}, kPi / 2.0f);
    SAB tf{VA{5, 5, 5}, rot_z};
    VB v_b{1, 0, 0};
    // apply_position = R*v + p
    auto p_a = tf.apply_position(v_b);
    CHECK(test::approx_equal(p_a, VA{5, 6, 5}));
    // rotate = R*v, no translation
    auto v_a = tf.rotate(v_b);
    CHECK(test::approx_equal(v_a, VA{0, 1, 0}));
}

TEST_CASE("StaticLink: forward then inverse is identity") {
    auto rot = Ori<A, float>::from_axis_angle(VA{0.3f, 0.5f, 0.8f}, 0.7f);
    SAB tf{VA{2, -3, 1}, rot};
    auto inv = tf.inverse();
    auto round_trip = tf * inv;
    CHECK(test::approx_equal(round_trip.position(), VA::zero()));
    CHECK(test::approx_equal(round_trip.orientation(), Ori<A, float>::identity()));

    auto round_trip_2 = inv * tf;
    CHECK(test::approx_equal(round_trip_2.position(), VB::zero()));
    CHECK(test::approx_equal(round_trip_2.orientation(), Ori<B, float>::identity()));
}

TEST_CASE("StaticLink: rotate_inverse undoes rotate") {
    auto rot = Ori<A, float>::from_axis_angle(VA{0, 0, 1}, kPi / 3.0f);
    SAB tf{VA::zero(), rot};
    VB v_b{1.5f, 2.5f, -0.7f};
    auto v_a   = tf.rotate(v_b);
    auto v_b2  = tf.rotate_inverse(v_a);
    CHECK(test::approx_equal(v_b2, v_b));
}

TEST_CASE("StaticLink: composition of translations adds positions") {
    SAB ab{VA{1, 0, 0}, Ori<A, float>::identity()};
    SBC bc{VB{0, 2, 0}, Ori<B, float>::identity()};
    auto ac = ab * bc;
    CHECK(test::approx_equal(ac.position(), VA{1, 2, 0}));
    CHECK(test::approx_equal(ac.orientation(), Ori<A, float>::identity()));
}

TEST_CASE("StaticLink: composition rotates the right offset before adding") {
    // ab rotates 90deg about z; bc is a pure translation of (1,0,0) in B.
    auto rot_z = Ori<A, float>::from_axis_angle(VA{0, 0, 1}, kPi / 2.0f);
    SAB ab{VA::zero(), rot_z};
    SBC bc{VB{1, 0, 0}, Ori<B, float>::identity()};
    auto ac = ab * bc;
    // The bc offset of (1,0,0) in B becomes (0,1,0) in A after rotation.
    CHECK(test::approx_equal(ac.position(), VA{0, 1, 0}));
}

TEST_CASE("StaticLink: associativity of composition") {
    auto rA = Ori<A, float>::from_axis_angle(VA{1, 0, 0}, 0.3f);
    auto rB = Ori<B, float>::from_axis_angle(VB{0, 1, 0}, 0.5f);
    SAB ab{VA{1, 2, 3}, rA};
    SBC bc{VB{4, 5, 6}, rB};
    // Apply a vector through both compositions.
    VB v_b{1, 0, 0};
    auto via_compose = (ab * bc).apply_position(Vec3<C, float>{0, 0, 0});
    auto via_chain   = ab.apply_position(bc.apply_position(Vec3<C, float>{0, 0, 0}));
    CHECK(test::approx_equal(via_compose, via_chain));
}
