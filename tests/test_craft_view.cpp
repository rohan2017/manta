// CraftView tests — confirms the per-craft adapter reads from the right
// slice of an EKFGeneric/UKFGeneric.

#include <doctest/doctest.h>

#include "../include/manta/estimation/craft_view.hpp"
#include "../include/manta/estimation/generic_ekf.hpp"
#include "../include/manta/estimation/state_spec.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::estimation;

namespace {

template <class Scalar>
class TwoDronesCraftV : public CraftT<Scalar> {
public:
    explicit TwoDronesCraftV(std::string name)
        : CraftT<Scalar>(std::move(name)) {
        this->root().template add<parts::MassT<Scalar>>("body", Scalar(1.0));
        this->root().compute_params();
    }
};

}

TEST_CASE("CraftView: position() reads the right slice for each craft") {
    CraftT<double> drone0("drone0");
    CraftT<double> drone1("drone1");
    drone0.root().template add<parts::MassT<double>>("body", 1.0);
    drone1.root().template add<parts::MassT<double>>("body", 1.0);
    drone0.root().compute_params();
    drone1.root().compute_params();

    auto state = make_state().track(drone0).track(drone1).build();
    using Spec = decltype(state);

    // Build an EKFGeneric we won't actually run — just need a filter
    // whose state vector we can read.
    EKFGeneric<Spec, /*MeasDim=*/0, /*NoiseSlots=*/0> ekf{state};

    // Manually set a state vector with distinct values per craft slot
    // so the views can demonstrate slice-isolation.
    Spec::AmbientVec x = Spec::AmbientVec::Zero();
    // Craft 0 occupies [0, 13).
    x(0)  = 1.0;  x(1)  = 2.0;  x(2)  = 3.0;        // pos
    x(3)  = 1.0;                                     // q.w (identity)
    x(7)  = 0.5;  x(8)  = 0.6;  x(9)  = 0.7;        // vel_linear
    x(10) = 0.1;  x(11) = 0.2;  x(12) = 0.3;        // vel_angular
    // Craft 1 occupies [13, 26).
    x(13) = 10.0; x(14) = 20.0; x(15) = 30.0;       // pos
    x(16) = 1.0;                                     // q.w
    x(20) = 5.0;  x(21) = 6.0;  x(22) = 7.0;        // vel_linear
    x(23) = 1.0;  x(24) = 2.0;  x(25) = 3.0;        // vel_angular

    ekf.set_state(x);

    CraftView<decltype(ekf), /*SliceIdx=*/0> view0{ekf};
    CraftView<decltype(ekf), /*SliceIdx=*/1> view1{ekf};

    auto p0 = view0.position();
    CHECK(p0(0) == doctest::Approx(1.0));
    CHECK(p0(1) == doctest::Approx(2.0));
    CHECK(p0(2) == doctest::Approx(3.0));

    auto p1 = view1.position();
    CHECK(p1(0) == doctest::Approx(10.0));
    CHECK(p1(1) == doctest::Approx(20.0));
    CHECK(p1(2) == doctest::Approx(30.0));

    auto v0 = view0.vel_linear();
    CHECK(v0(0) == doctest::Approx(0.5));
    auto v1 = view1.vel_linear();
    CHECK(v1(0) == doctest::Approx(5.0));

    auto w0 = view0.vel_angular();
    CHECK(w0(2) == doctest::Approx(0.3));
    auto w1 = view1.vel_angular();
    CHECK(w1(2) == doctest::Approx(3.0));
}

TEST_CASE("CraftView: stddev reads the right tangent slice") {
    CraftT<double> drone0("drone0");
    drone0.root().template add<parts::MassT<double>>("body", 1.0);
    drone0.root().compute_params();

    auto state = make_state().track(drone0).build();
    using Spec = decltype(state);
    EKFGeneric<Spec, /*MeasDim=*/0, /*NoiseSlots=*/0> ekf{state};

    // P with distinguishable diagonal: position variance 4.0, attitude 1.0,
    // vel 0.25, angvel 0.16.
    typename EKFGeneric<Spec, 0, 0>::StateCov P =
        EKFGeneric<Spec, 0, 0>::StateCov::Identity();
    for (int i = 0; i < 3;  ++i) P(i,   i)   = 4.0;
    for (int i = 3; i < 6;  ++i) P(i,   i)   = 1.0;
    for (int i = 6; i < 9;  ++i) P(i,   i)   = 0.25;
    for (int i = 9; i < 12; ++i) P(i,   i)   = 0.16;
    ekf.set_covariance(P);

    CraftView<decltype(ekf), 0> view{ekf};
    auto p_sd  = view.position_stddev();
    auto a_sd  = view.orientation_stddev();
    auto v_sd  = view.vel_linear_stddev();
    auto w_sd  = view.vel_angular_stddev();

    for (int i = 0; i < 3; ++i) CHECK(p_sd(i) == doctest::Approx(2.0));
    for (int i = 0; i < 3; ++i) CHECK(a_sd(i) == doctest::Approx(1.0));
    for (int i = 0; i < 3; ++i) CHECK(v_sd(i) == doctest::Approx(0.5));
    for (int i = 0; i < 3; ++i) CHECK(w_sd(i) == doctest::Approx(0.4));
}
