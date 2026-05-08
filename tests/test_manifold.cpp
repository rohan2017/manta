// Manifold primitives + StateSpec tests.
// =====================================
//
// Covers:
//   * Euclidean<N>: boxplus is just addition; boxminus is subtraction.
//   * SO3: boxplus(q, 0) = q; boxminus(boxplus(q, dθ), q) = dθ within
//     the small-angle range.
//   * Compose: layout offsets agree with the slice list.
//   * Jet path: boxplus on a Jet δ produces a derivative matching the
//     analytical Lie generator (½·e_i for the orientation part).
//   * StateSpec/StateBuilder: track(craft) yields a RigidBody slice;
//     track(noise_rw3) yields a BiasRandomWalk<3> slice; pull_ambient
//     round-trips a craft's rigid state.

#include <doctest/doctest.h>
#include <ceres/jet.h>

#include "../include/manta/estimation/manifold.hpp"
#include "../include/manta/estimation/state_spec.hpp"
#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/noise.hpp"

using namespace manta;
using namespace manta::manifold;
using namespace manta::estimation;

TEST_CASE("Manifold: Euclidean<3> boxplus is addition") {
    double x_ref[3] = {1.0, 2.0, 3.0};
    double delta[3] = {0.1, -0.2, 0.5};
    double x_post[3];

    Euclidean<3>::boxplus<double>(x_ref, delta, x_post);

    CHECK(x_post[0] == doctest::Approx(1.1));
    CHECK(x_post[1] == doctest::Approx(1.8));
    CHECK(x_post[2] == doctest::Approx(3.5));
}

TEST_CASE("Manifold: Euclidean<3> boxminus is subtraction") {
    double a[3] = {1.5, 2.5, 3.5};
    double b[3] = {1.0, 2.0, 3.0};
    double delta[3];

    Euclidean<3>::boxminus<double>(a, b, delta);

    CHECK(delta[0] == doctest::Approx(0.5));
    CHECK(delta[1] == doctest::Approx(0.5));
    CHECK(delta[2] == doctest::Approx(0.5));
}

TEST_CASE("Manifold: SO3 boxplus(q, 0) returns q unchanged") {
    // q = small rotation around z, ~30°.
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.524, Eigen::Vector3d::UnitZ()));
    double x_ref[4] = {q.w(), q.x(), q.y(), q.z()};
    double delta[3] = {0.0, 0.0, 0.0};
    double x_post[4];

    SO3::boxplus<double>(x_ref, delta, x_post);

    CHECK(x_post[0] == doctest::Approx(q.w()));
    CHECK(x_post[1] == doctest::Approx(q.x()));
    CHECK(x_post[2] == doctest::Approx(q.y()));
    CHECK(x_post[3] == doctest::Approx(q.z()));
}

TEST_CASE("Manifold: SO3 boxminus(boxplus(q, dθ), q) ≈ dθ for small dθ") {
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
    double x_ref[4] = {q.w(), q.x(), q.y(), q.z()};
    double dtheta[3] = {0.01, -0.005, 0.02};
    double x_post[4];
    double dtheta_recovered[3];

    SO3::boxplus<double>(x_ref, dtheta, x_post);
    SO3::boxminus<double>(x_post, x_ref, dtheta_recovered);

    CHECK(dtheta_recovered[0] == doctest::Approx(dtheta[0]).epsilon(1e-5));
    CHECK(dtheta_recovered[1] == doctest::Approx(dtheta[1]).epsilon(1e-5));
    CHECK(dtheta_recovered[2] == doctest::Approx(dtheta[2]).epsilon(1e-5));
}

TEST_CASE("Manifold: SO3 boxplus Jacobian on a Jet input matches Lie generator at zero") {
    using Jet = ceres::Jet<double, 3>;

    // q_ref = identity. Jet δ = identity-Jet (∂/∂δθ at 0).
    double x_ref[4] = {1.0, 0.0, 0.0, 0.0};
    Jet dtheta[3] = { Jet(0.0, 0), Jet(0.0, 1), Jet(0.0, 2) };
    Jet x_post[4];

    SO3::boxplus<Jet>(x_ref, dtheta, x_post);

    // Value channel: q_ref ⊗ exp(0) = (1, 0, 0, 0).
    CHECK(x_post[0].a == doctest::Approx(1.0));
    CHECK(x_post[1].a == doctest::Approx(0.0));
    CHECK(x_post[2].a == doctest::Approx(0.0));
    CHECK(x_post[3].a == doctest::Approx(0.0));

    // Derivative channel: ∂q/∂δθ_i at 0 = (0, ½·e_i).
    // x_post[0] = qw, has ∂qw/∂δθ_i = 0 for all i.
    for (int i = 0; i < 3; ++i) CHECK(x_post[0].v[i] == doctest::Approx(0.0));
    // x_post[1..3] = (qx, qy, qz). ∂q_imag/∂δθ_i = 0.5 · e_i.
    CHECK(x_post[1].v[0] == doctest::Approx(0.5));
    CHECK(x_post[1].v[1] == doctest::Approx(0.0));
    CHECK(x_post[1].v[2] == doctest::Approx(0.0));
    CHECK(x_post[2].v[0] == doctest::Approx(0.0));
    CHECK(x_post[2].v[1] == doctest::Approx(0.5));
    CHECK(x_post[2].v[2] == doctest::Approx(0.0));
    CHECK(x_post[3].v[0] == doctest::Approx(0.0));
    CHECK(x_post[3].v[1] == doctest::Approx(0.0));
    CHECK(x_post[3].v[2] == doctest::Approx(0.5));
}

TEST_CASE("Manifold: Compose layout sums child slice dims") {
    using S = Compose<Euclidean<3>, SO3, Euclidean<3>, Euclidean<3>>;
    static_assert(S::ambient == 13, "3+4+3+3");
    static_assert(S::tangent == 12, "3+3+3+3");
}

TEST_CASE("Manifold: RigidBody alias matches the legacy 13/12 layout") {
    static_assert(RigidBody::ambient == 13);
    static_assert(RigidBody::tangent == 12);
}

TEST_CASE("StateSpec: track(craft) yields a RigidBody slice") {
    CraftT<double> c("test_craft");
    auto state = make_state().track(c).build();
    using Spec = decltype(state);
    static_assert(Spec::num_slices  == 1);
    static_assert(Spec::ambient_dim == 13);
    static_assert(Spec::tangent_dim == 12);
}

TEST_CASE("StateSpec: track(craft) + track(rw bias) sums dims") {
    CraftT<double> c("test_craft");
    Noise<RandomWalk<3>> bias{0.01f};
    auto state = make_state().track(c).track(bias).build();
    using Spec = decltype(state);
    static_assert(Spec::num_slices  == 2);
    static_assert(Spec::ambient_dim == 13 + 3);
    static_assert(Spec::tangent_dim == 12 + 3);
    static_assert(Spec::template ambient_offset<1> == 13);
    static_assert(Spec::template tangent_offset<1> == 12);
}

TEST_CASE("StateSpec: pull_ambient round-trips a craft's rigid state") {
    CraftT<double> c("rt");
    typename CraftT<double>::RigidState rs;
    rs << 1.0, 2.0, 3.0,                    // p
          1.0, 0.0, 0.0, 0.0,                // q (identity)
          0.5, -0.5, 0.25,                   // v
          0.1, 0.2, 0.3;                     // ω
    c.set_rigid_state(rs);

    auto state = make_state().track(c).build();
    typename decltype(state)::AmbientVec x;
    state.pull_ambient(x);

    for (int i = 0; i < 13; ++i) {
        CHECK(x(i) == doctest::Approx(rs(i)));
    }
}

TEST_CASE("StateSpec: pull_ambient round-trips a RW bias") {
    Noise<RandomWalk<3>> bias{0.01f};
    Eigen::Matrix<float, 3, 1> v;
    v << 0.1f, -0.2f, 0.3f;
    bias.set_state(v);

    CraftT<double> dummy_craft("dummy");   // need a craft so the spec isn't empty
    auto state = make_state().track(dummy_craft).track(bias).build();
    typename decltype(state)::AmbientVec x;
    state.pull_ambient(x);

    // Bias values land at ambient_offset<1> = 13.
    constexpr int off = decltype(state)::template ambient_offset<1>;
    CHECK(x(off + 0) == doctest::Approx(0.1));
    CHECK(x(off + 1) == doctest::Approx(-0.2));
    CHECK(x(off + 2) == doctest::Approx(0.3));
}

TEST_CASE("StateSpec: push_ambient writes back into the craft") {
    CraftT<double> c("rt");
    auto state = make_state().track(c).build();
    typename decltype(state)::AmbientVec x = decltype(state)::AmbientVec::Zero();
    x(0) = 7.0; x(1) = 8.0; x(2) = 9.0;
    x(3) = 1.0;   // identity quaternion
    state.push_ambient(x);

    auto rs = c.get_rigid_state();
    CHECK(rs(0) == doctest::Approx(7.0));
    CHECK(rs(1) == doctest::Approx(8.0));
    CHECK(rs(2) == doctest::Approx(9.0));
    CHECK(rs(3) == doctest::Approx(1.0));
}
