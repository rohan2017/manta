#include <cmath>
#include <doctest/doctest.h>

#include "../include/mantapilot/core/frame.hpp"
#include "../include/mantapilot/geom/kinematic_link.hpp"
#include "../include/mantapilot/geom/static_link.hpp"
#include "test_helpers.hpp"

using namespace manta;
using namespace manta::geom;

constexpr float kPi = 3.14159265358979323846f;

struct A { static constexpr FrameKind kind = FrameKind::World; };
struct B { static constexpr FrameKind kind = FrameKind::Scene; };
struct C { static constexpr FrameKind kind = FrameKind::Craft; };

using KAB = KinematicLink<A, B, float>;
using KBC = KinematicLink<B, C, float>;
using SAB = StaticLink<A, B, float>;
using VA = Vec3<A, float>;
using VB = Vec3<B, float>;
using VC = Vec3<C, float>;

// ---------- identity / static cases ----------

TEST_CASE("KinematicLink: identity has zero everything and identity rotation") {
    auto k = KAB::identity();
    CHECK(test::approx_equal(k.position(),    VA::zero()));
    CHECK(test::approx_equal(k.orientation(), Ori<A, float>::identity()));
    CHECK(test::approx_equal(k.vel_linear(),  VA::zero()));
    CHECK(test::approx_equal(k.vel_angular(), VB::zero()));
    CHECK(test::approx_equal(k.acc_linear(),  VA::zero()));
    CHECK(test::approx_equal(k.acc_angular(), VB::zero()));
}

TEST_CASE("KinematicLink: from_static promotes a static link to a kinematic one") {
    auto rot = Ori<A, float>::from_axis_angle(VA{0,0,1}, kPi / 4);
    SAB s{VA{1, 2, 3}, rot};
    auto k = KAB::from_static(s);
    CHECK(test::approx_equal(k.position(),    s.position()));
    CHECK(test::approx_equal(k.orientation(), s.orientation()));
    CHECK(test::approx_equal(k.vel_linear(),  VA::zero()));
    CHECK(test::approx_equal(k.vel_angular(), VB::zero()));
}

TEST_CASE("KinematicLink: apply_position acts like the static transform") {
    auto rot_z = Ori<A, float>::from_axis_angle(VA{0,0,1}, kPi / 2.0f);
    KAB k{VA{1, 0, 0}, rot_z, VA::zero(), VB::zero()};
    auto p_a = k.apply_position(VB{1, 0, 0});
    // (1,0,0) rotated 90deg about z -> (0,1,0); plus origin offset (1,0,0).
    CHECK(test::approx_equal(p_a, VA{1, 1, 0}));
}

// ---------- velocity of a static point under rotation ----------

TEST_CASE("KinematicLink: stationary parent, parent-fixed point has zero velocity in parent") {
    KAB k = KAB::identity();  // zero vels
    auto v_a = k.apply_velocity_of_static_point(VB{1, 0, 0});
    CHECK(test::approx_equal(v_a, VA::zero()));
}

TEST_CASE("KinematicLink: rotating frame imparts omega x r velocity to a B-fixed point") {
    // B is at (0,0,0) in A, oriented identical to A, rotating about z with omega=2.
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA::zero(), VB{0, 0, 2.0f}};
    // Point fixed at (1,0,0) in B. omega x r = (0,0,2) x (1,0,0) = (0,2,0) in A.
    auto v_a = k.apply_velocity_of_static_point(VB{1, 0, 0});
    CHECK(test::approx_equal(v_a, VA{0, 2, 0}));
}

// ---------- inverse ----------

TEST_CASE("KinematicLink: static-pose inverse round-trips position") {
    auto rot = Ori<A, float>::from_axis_angle(VA{0.3f, 0.5f, 0.8f}, 0.6f);
    KAB k{VA{2, -1, 3}, rot, VA::zero(), VB::zero()};
    auto inv = k.inverse();
    // Inverse-of-inverse should restore position and orientation (vels remain
    // zero since both are zero here).
    auto k2 = inv.inverse();
    CHECK(test::approx_equal(k2.position(),    k.position()));
    CHECK(test::approx_equal(k2.orientation(), k.orientation()));
}

TEST_CASE("KinematicLink: inverse of pure translation negates position and velocity") {
    KAB k{VA{5, 0, 0}, Ori<A, float>::identity(),
          VA{1, 0, 0}, VB::zero()};
    auto inv = k.inverse();
    CHECK(test::approx_equal(inv.position(),   VB{-5, 0, 0}));
    CHECK(test::approx_equal(inv.vel_linear(), VB{-1, 0, 0}));
    CHECK(test::approx_equal(inv.vel_angular(), VA::zero()));
}

TEST_CASE("KinematicLink: inverse of rotating-but-not-translating link") {
    // B sits at A's origin, rotating about z with omega=omega_z (in B-frame).
    // The A-origin (fixed in A) has zero velocity in A. In B-coordinates, the
    // A-origin is at (0,0,0) (coincident with B-origin), so its velocity in B
    // is also zero.
    float w = 2.5f;
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA::zero(), VB{0, 0, w}};
    auto inv = k.inverse();
    CHECK(test::approx_equal(inv.position(),    VB::zero()));
    CHECK(test::approx_equal(inv.vel_linear(),  VB::zero()));
    // Angular velocity of A relative to B, in A frame, should be (0,0,-w).
    CHECK(test::approx_equal(inv.vel_angular(), VA{0, 0, -w}));
}

TEST_CASE("KinematicLink: inverse of offset rotating link sees A-origin moving in B") {
    // B is at (1,0,0) in A, B is rotating about its own z with omega=1.
    // The A-origin is at (-1,0,0) in B, and as seen in B it should appear to
    // move at (0,1,0) — see the derivation in the design doc / code comment.
    float w = 1.0f;
    KAB k{VA{1, 0, 0}, Ori<A, float>::identity(),
          VA::zero(), VB{0, 0, w}};
    auto inv = k.inverse();
    CHECK(test::approx_equal(inv.position(),   VB{-1, 0, 0}));
    CHECK(test::approx_equal(inv.vel_linear(), VB{0, w, 0}));
}

// ---------- composition ----------

TEST_CASE("KinematicLink: identity composition is a no-op") {
    auto rot = Ori<A, float>::from_axis_angle(VA{0,0,1}, 0.3f);
    KAB k{VA{1,2,3}, rot, VA{0.1f, 0.2f, 0.3f}, VB{0.0f, 0.0f, 0.5f}};
    auto id_BC = KBC::identity();
    auto kc = k * id_BC;
    CHECK(test::approx_equal(kc.position(),    k.position()));
    CHECK(test::approx_equal(kc.orientation(), k.orientation()));
    CHECK(test::approx_equal(kc.vel_linear(),  k.vel_linear()));
    // Note: vel_angular of k is in B; of kc is in C; with identity bc they coincide.
    CHECK(test::approx_equal(Vec3<C, float>::from_raw(k.vel_angular().raw()),
                             kc.vel_angular()));
}

TEST_CASE("KinematicLink: composing static-pose links matches StaticLink composition") {
    auto rA = Ori<A, float>::from_axis_angle(VA{0,0,1}, kPi / 3.0f);
    auto rB = Ori<B, float>::from_axis_angle(VB{1,0,0}, 0.4f);
    KAB k1{VA{1, 0, 0}, rA, VA::zero(), VB::zero()};
    KBC k2{VB{0, 2, 0}, rB, VB::zero(), VC::zero()};
    auto kc = k1 * k2;

    StaticLink<A, B, float> s1{VA{1, 0, 0}, rA};
    StaticLink<B, C, float> s2{VB{0, 2, 0}, rB};
    auto sc = s1 * s2;

    CHECK(test::approx_equal(kc.position(),    sc.position()));
    CHECK(test::approx_equal(kc.orientation(), sc.orientation()));
    CHECK(test::approx_equal(kc.vel_linear(),  VA::zero()));
}

TEST_CASE("KinematicLink: a B-fixed point inherits B's rotational velocity in A") {
    // A fixed; B at origin rotating about z with omega=ω in B.
    // C fixed in B at (1,0,0). The C-origin in A should have linear velocity
    // omega x (1,0,0) = (0,ω,0).
    float w = 1.5f;
    KAB k1{VA::zero(), Ori<A, float>::identity(), VA::zero(), VB{0, 0, w}};
    KBC k2{VB{1, 0, 0}, Ori<B, float>::identity(), VB::zero(), VC::zero()};
    auto kc = k1 * k2;
    CHECK(test::approx_equal(kc.position(),    VA{1, 0, 0}));
    CHECK(test::approx_equal(kc.vel_linear(),  VA{0, w, 0}));
    // Angular velocity of C in A, in C frame: same as omega of B (omega_BC = 0).
    CHECK(test::approx_equal(kc.vel_angular(), VC{0, 0, w}));
}

TEST_CASE("KinematicLink: angular velocities add when frames spin about same axis") {
    // B spinning at ω1 about z relative to A; C spinning at ω2 about z
    // relative to B. Then C spins at ω1+ω2 about z relative to A.
    float w1 = 0.7f, w2 = 0.3f;
    KAB k1{VA::zero(), Ori<A, float>::identity(), VA::zero(), VB{0, 0, w1}};
    KBC k2{VB::zero(), Ori<B, float>::identity(), VB::zero(), VC{0, 0, w2}};
    auto kc = k1 * k2;
    CHECK(test::approx_equal(kc.vel_angular(), VC{0, 0, w1 + w2}));
}

// ---------- RK4 / update ----------

TEST_CASE("KinematicLink::update: zero-velocity link does not change") {
    auto rot = Ori<A, float>::from_axis_angle(VA{0,0,1}, 0.5f);
    KAB k{VA{1, 2, 3}, rot, VA::zero(), VB::zero()};
    auto before = k;
    k.update(0.1f);
    CHECK(test::approx_equal(k.position(),    before.position()));
    CHECK(test::approx_equal(k.orientation(), before.orientation()));
}

TEST_CASE("KinematicLink::update: constant linear velocity gives p = p0 + v*dt") {
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA{1.0f, 2.0f, -3.0f}, VB::zero()};
    k.update(0.5f);
    CHECK(test::approx_equal(k.position(), VA{0.5f, 1.0f, -1.5f}));
    CHECK(test::approx_equal(k.vel_linear(), VA{1.0f, 2.0f, -3.0f}));
}

TEST_CASE("KinematicLink::update: constant linear acceleration matches kinematic eqn") {
    // p(dt) = p0 + v0*dt + 0.5*a*dt^2; v(dt) = v0 + a*dt.
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA{0.0f, 0.0f, 0.0f}, VB::zero(),
          VA{0.0f, 0.0f, -9.81f}, VB::zero()};
    float dt = 1.0f;
    k.update(dt);
    CHECK(test::approx_equal(k.position(),
                             VA{0.0f, 0.0f, -0.5f * 9.81f * dt * dt}));
    CHECK(test::approx_equal(k.vel_linear(),
                             VA{0.0f, 0.0f, -9.81f}));
}

TEST_CASE("KinematicLink::update: free-fall over multiple steps tracks parabola") {
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA{1.0f, 0.0f, 0.0f}, VB::zero(),
          VA{0.0f, 0.0f, -9.81f}, VB::zero()};
    float dt = 0.01f;
    int   n  = 100;
    for (int i = 0; i < n; ++i) k.update(dt);
    float t = dt * n;
    // p_x = v0_x * t = 1.0; p_z = -0.5 g t^2.
    CHECK(k.position().x() == doctest::Approx(1.0f).epsilon(1e-3));
    CHECK(k.position().z() == doctest::Approx(-0.5f * 9.81f * t * t).epsilon(1e-3));
    CHECK(k.vel_linear().z() == doctest::Approx(-9.81f * t).epsilon(1e-3));
}

TEST_CASE("KinematicLink::update: constant angular velocity rotates by omega*dt") {
    // Spin about z with omega=pi rad/s for dt=0.5s -> 90deg rotation.
    float w = kPi;
    float dt = 0.5f;
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA::zero(), VB{0, 0, w}};
    k.update(dt);
    auto expected = Ori<A, float>::from_axis_angle(VA{0,0,1}, w * dt);
    CHECK(test::approx_equal(k.orientation(), expected));
    CHECK(test::approx_equal(k.vel_angular(), VB{0, 0, w}));
}

TEST_CASE("KinematicLink::update: integrating a full revolution returns to start") {
    // Spin about z with omega=2pi rad/s for dt=1s -> 360deg, back to start.
    float w  = 2.0f * kPi;
    float dt = 0.001f;
    int   n  = 1000;
    KAB k{VA::zero(), Ori<A, float>::identity(),
          VA::zero(), VB{0, 0, w}};
    for (int i = 0; i < n; ++i) k.update(dt);
    CHECK(test::approx_equal(k.orientation(), Ori<A, float>::identity(),
                             test::kLooseTol));
}
