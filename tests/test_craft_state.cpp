// Manifold primitives on CraftState — boxplus retraction and boxminus.
// These are the building blocks the ESKF uses to keep its 13-DOF reference
// state on-manifold while running its 12-DOF tangent covariance.

#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/scene.hpp"

using namespace manta;
using namespace manta::geom;

namespace {

CraftState make_state(MFloat px, MFloat py, MFloat pz,
                      MFloat qw, MFloat qx, MFloat qy, MFloat qz,
                      MFloat vx = 0, MFloat vy = 0, MFloat vz = 0,
                      MFloat wx = 0, MFloat wy = 0, MFloat wz = 0) {
    CraftState s;
    s.position    = Vec3<SceneFrame>{px, py, pz};
    s.orientation = Ori<SceneFrame>{Eigen::Quaternion<MFloat>{qw, qx, qy, qz}.normalized()};
    s.vel_linear  = Vec3<SceneFrame>{vx, vy, vz};
    s.vel_angular = Vec3<CraftFrame>{wx, wy, wz};
    return s;
}

CraftTangent make_tangent(MFloat dpx, MFloat dpy, MFloat dpz,
                          MFloat dtx, MFloat dty, MFloat dtz,
                          MFloat dvx, MFloat dvy, MFloat dvz,
                          MFloat dwx, MFloat dwy, MFloat dwz) {
    CraftTangent t;
    t.position    = Vec3<SceneFrame>{dpx, dpy, dpz};
    t.orientation = Vec3<SceneFrame>{dtx, dty, dtz};
    t.vel_linear  = Vec3<SceneFrame>{dvx, dvy, dvz};
    t.vel_angular = Vec3<CraftFrame>{dwx, dwy, dwz};
    return t;
}

bool quat_close(const Eigen::Quaternion<MFloat>& a, const Eigen::Quaternion<MFloat>& b,
                MFloat tol = MFloat(1e-5)) {
    // q and -q are the same rotation; |q · b| ≈ 1 either way.
    return std::abs(std::abs(a.dot(b)) - MFloat(1)) < tol;
}

bool state_close(const CraftState& a, const CraftState& b, MFloat tol = MFloat(1e-5)) {
    if ((a.position.raw() - b.position.raw()).norm() > tol) return false;
    if ((a.vel_linear.raw() - b.vel_linear.raw()).norm() > tol) return false;
    if ((a.vel_angular.raw() - b.vel_angular.raw()).norm() > tol) return false;
    return quat_close(a.orientation.raw(), b.orientation.raw(), tol);
}

bool tangent_close(const CraftTangent& a, const CraftTangent& b, MFloat tol = MFloat(1e-5)) {
    return (a.position.raw()    - b.position.raw()).norm()    < tol
        && (a.orientation.raw() - b.orientation.raw()).norm() < tol
        && (a.vel_linear.raw()  - b.vel_linear.raw()).norm()  < tol
        && (a.vel_angular.raw() - b.vel_angular.raw()).norm() < tol;
}

} // namespace

TEST_CASE("CraftTangent: zero tangent leaves state unchanged") {
    auto x = make_state(1, 2, 3, 0.5f, 0.5f, 0.5f, 0.5f, 4, 5, 6, 7, 8, 9);
    auto out = boxplus(x, CraftTangent::zero());
    CHECK(state_close(x, out));
}

TEST_CASE("CraftTangent: pure-translation tangent acts on Euclidean parts") {
    auto x   = make_state(0, 0, 0, 1, 0, 0, 0);
    auto d   = make_tangent(2, 3, 5,  0, 0, 0,  -1, 1, 2,   0, 0, 0);
    auto out = boxplus(x, d);
    CHECK(out.position.x()   == doctest::Approx(2));
    CHECK(out.position.y()   == doctest::Approx(3));
    CHECK(out.position.z()   == doctest::Approx(5));
    CHECK(out.vel_linear.x() == doctest::Approx(-1));
    CHECK(out.vel_linear.y() == doctest::Approx(1));
    CHECK(out.vel_linear.z() == doctest::Approx(2));
    // Orientation untouched (identity in, identity out).
    CHECK(quat_close(out.orientation.raw(), Eigen::Quaternion<MFloat>::Identity()));
}

TEST_CASE("CraftTangent: small δθ retracts via the so(3) exp map") {
    // q_ref = identity, δθ = (0.1, 0, 0) → q_full ≈ rotation(0.1 rad about x).
    auto x   = make_state(0, 0, 0, 1, 0, 0, 0);
    auto d   = make_tangent(0, 0, 0,  MFloat(0.1), 0, 0,  0, 0, 0,  0, 0, 0);
    auto out = boxplus(x, d);

    // exp(δθ/2) = (cos(0.05), sin(0.05), 0, 0).
    Eigen::Quaternion<MFloat> expected(
        std::cos(MFloat(0.05)), std::sin(MFloat(0.05)), MFloat(0), MFloat(0));
    CHECK(quat_close(out.orientation.raw(), expected));
}

TEST_CASE("CraftTangent: large δθ still produces a unit quaternion") {
    // 2π/3 about (1, 1, 1)/√3 — well outside the small-angle regime.
    const MFloat angle = MFloat(2) * MFloat(M_PI) / MFloat(3);
    Eigen::Matrix<MFloat, 3, 1> axis =
        Eigen::Matrix<MFloat, 3, 1>{MFloat(1), MFloat(1), MFloat(1)}.normalized();
    Eigen::Matrix<MFloat, 3, 1> delta_theta = angle * axis;

    auto x   = make_state(0, 0, 0, 1, 0, 0, 0);
    CraftTangent d;
    d.orientation = Vec3<SceneFrame>{delta_theta(0), delta_theta(1), delta_theta(2)};
    auto out = boxplus(x, d);

    CHECK(out.orientation.raw().norm() == doctest::Approx(1.0).epsilon(1e-5));
    Eigen::Quaternion<MFloat> expected(Eigen::AngleAxis<MFloat>(angle, axis));
    CHECK(quat_close(out.orientation.raw(), expected));
}

TEST_CASE("CraftTangent: boxplus / boxminus round-trip recovers the tangent") {
    auto base = make_state(MFloat(1), MFloat(-2), MFloat(0.5),
                           MFloat(0.7071068), MFloat(0), MFloat(0.7071068), MFloat(0),
                           MFloat(1), MFloat(2), MFloat(3),
                           MFloat(0.1), MFloat(-0.2), MFloat(0.05));
    auto d_in = make_tangent(MFloat(0.01), MFloat(-0.02), MFloat(0.03),
                             MFloat(-0.04), MFloat(0.05), MFloat(0.06),
                             MFloat(0.1), MFloat(-0.2), MFloat(0.3),
                             MFloat(-0.05), MFloat(0.07), MFloat(-0.02));

    auto x_full = boxplus(base, d_in);
    auto d_out  = boxminus(x_full, base);

    CHECK(tangent_close(d_in, d_out, MFloat(1e-5)));
}

TEST_CASE("CraftTangent: boxminus(x, x) is zero") {
    auto x = make_state(MFloat(1), MFloat(2), MFloat(3),
                        MFloat(0.6), MFloat(0), MFloat(0.8), MFloat(0));
    auto d = boxminus(x, x);
    CHECK(d.to_vec().norm() == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("CraftTangent: 12-vec round-trip via to_vec / from_vec") {
    auto d_in = make_tangent(1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12);
    auto v    = d_in.to_vec();
    REQUIRE(v.rows() == 12);
    auto d_out = CraftTangent::from_vec(v);
    CHECK(tangent_close(d_in, d_out));
}
