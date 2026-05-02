// Verifies that `manta::fields::state_at_templated<Scalar>` correctly
// propagates derivatives through the field query when Scalar is a
// ceres::Jet. The Jet output's .v component should match the analytical
// spatial gradient of the field, contracted with the input Jet's .v.

#include <cmath>
#include <ceres/jet.h>
#include <doctest/doctest.h>

#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/fields/mag_field.hpp"
#include "../include/manta/fields/templated_query.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::fields;

TEST_CASE("state_at_templated<double>: matches state_at exactly") {
    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(
               Vec3<SceneFrame>::zero(), Real(4.0e14f)),
           PERSISTENT);

    Vec3<SceneFrame, double> p_d{1.0e7, 0.0, 0.0};
    auto g = state_at_templated<double>(gf, p_d);
    // Reference: |g| = mu/r² = 4 m/s² along -x.
    CHECK(g.x() == doctest::Approx(-4.0).epsilon(1e-4));
    CHECK(g.y() == doctest::Approx( 0.0).epsilon(1e-4));
    CHECK(g.z() == doctest::Approx( 0.0).epsilon(1e-4));
}

TEST_CASE("state_at_templated<Jet>: uniform field has zero spatial gradient") {
    GravityField gf{Vec3<SceneFrame>{0, 0, -9.81f}};   // uniform
    using Jet = ceres::Jet<double, 3>;

    // Build a position with .v = identity (so we can read gradients off
    // the output Jet's .v rows directly).
    Vec3<SceneFrame, Jet> p_jet;
    p_jet.raw()(0) = Jet(1.0); p_jet.raw()(0).v = Eigen::Matrix<double, 3, 1>::Unit(0);
    p_jet.raw()(1) = Jet(2.0); p_jet.raw()(1).v = Eigen::Matrix<double, 3, 1>::Unit(1);
    p_jet.raw()(2) = Jet(3.0); p_jet.raw()(2).v = Eigen::Matrix<double, 3, 1>::Unit(2);

    auto g_jet = state_at_templated<Jet>(gf, p_jet);

    // Uniform g everywhere → output values are constants, derivatives all zero.
    CHECK(g_jet.z().a == doctest::Approx(-9.81));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(g_jet.raw()(i).v(j) == doctest::Approx(0.0).epsilon(1e-3));
        }
    }
}

TEST_CASE("state_at_templated<Jet>: point-mass gravity gradient matches analytical") {
    // Point mass μ at origin. g(r) = -μ r / |r|³.
    // Spatial Jacobian J(i,j) = ∂g_i/∂r_j = -μ/|r|³ · (δ_ij - 3 r_i r_j / |r|²).
    constexpr double MU = 4.0e14;
    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(
               Vec3<SceneFrame>::zero(), Real(MU)),
           PERSISTENT);

    using Jet = ceres::Jet<double, 3>;

    // Sample at (1e7, 0, 0). |r| = 1e7. Analytical Jacobian:
    //   r_i r_j / |r|² has only (0,0) = 1 nonzero, so:
    //     J(0,0) = -μ/|r|³ · (1 - 3) = 2μ/|r|³ ≈ +8e-7
    //     J(1,1) = -μ/|r|³ · (1 - 0) = -μ/|r|³ ≈ -4e-7
    //     J(2,2) = -μ/|r|³ · (1 - 0) = -μ/|r|³ ≈ -4e-7
    //     off-diagonals = 0.
    constexpr double R = 1.0e7;
    Vec3<SceneFrame, Jet> p_jet;
    p_jet.raw()(0) = Jet(R);   p_jet.raw()(0).v = Eigen::Matrix<double, 3, 1>::Unit(0);
    p_jet.raw()(1) = Jet(0.0); p_jet.raw()(1).v = Eigen::Matrix<double, 3, 1>::Unit(1);
    p_jet.raw()(2) = Jet(0.0); p_jet.raw()(2).v = Eigen::Matrix<double, 3, 1>::Unit(2);

    auto g_jet = state_at_templated<Jet>(gf, p_jet);

    double inv_r3 = 1.0 / (R * R * R);
    double J00 = +2.0 * MU * inv_r3;
    double J11 = -1.0 * MU * inv_r3;
    double J22 = -1.0 * MU * inv_r3;

    CHECK(g_jet.x().v(0) == doctest::Approx(J00).epsilon(1e-2));
    CHECK(g_jet.y().v(1) == doctest::Approx(J11).epsilon(1e-2));
    CHECK(g_jet.z().v(2) == doctest::Approx(J22).epsilon(1e-2));

    // Off-diagonals should be ~0 (down to finite-diff noise).
    CHECK(std::abs(g_jet.x().v(1)) < 1e-9);
    CHECK(std::abs(g_jet.x().v(2)) < 1e-9);
    CHECK(std::abs(g_jet.y().v(0)) < 1e-9);
    CHECK(std::abs(g_jet.z().v(0)) < 1e-9);
}

TEST_CASE("state_at_templated<Jet>: chains through a Jet input with non-trivial .v") {
    // Same point-mass setup. Now the input position itself depends on a
    // single parameter (e.g. a craft's scalar state coordinate), so each
    // pos_i.v has length 1 with a single entry.
    constexpr double MU = 4.0e14;
    GravityField gf;
    gf.add(GravityField::Disturbance::point_mass(
               Vec3<SceneFrame>::zero(), Real(MU)),
           PERSISTENT);

    using Jet = ceres::Jet<double, 1>;
    constexpr double R = 1.0e7;

    // pos = (R + s, 0, 0) where s is the parameter. ∂pos/∂s = (1, 0, 0).
    Vec3<SceneFrame, Jet> p_jet;
    p_jet.raw()(0) = Jet(R);   p_jet.raw()(0).v(0) = 1.0;
    p_jet.raw()(1) = Jet(0.0); p_jet.raw()(1).v(0) = 0.0;
    p_jet.raw()(2) = Jet(0.0); p_jet.raw()(2).v(0) = 0.0;

    auto g_jet = state_at_templated<Jet>(gf, p_jet);

    // Output ∂g_x/∂s = ∂g_x/∂x · ∂x/∂s = J(0,0) · 1 = +2μ/R³.
    double expected = +2.0 * MU / (R * R * R);
    CHECK(g_jet.x().v(0) == doctest::Approx(expected).epsilon(1e-2));
    CHECK(std::abs(g_jet.y().v(0)) < 1e-9);
    CHECK(std::abs(g_jet.z().v(0)) < 1e-9);
}
