// Phase 5e: bias-as-tracked-slice in EKFGeneric.
//
// When the user does:
//
//   auto state = make_state()
//       .track(craft)
//       .track(imu.accel_bias())
//       .build();
//
// the StateSpec gains a BiasRandomWalk<3> slice, EKFGeneric::bind
// resolves the Jet-side counterpart of the noise (by part-name +
// random-walk name lookup), and predict's L matrix gains σ_rw·√dt
// at the bias's tangent rows. Auto-Q then grows the Q diagonal by
// σ_rw²·dt on the bias state.
//
// Tests:
//   1. Slice dim/offset arithmetic — state grows by 3 ambient + 3 tangent.
//   2. After predict, P diagonal at the bias slot grew by ≥ σ²·dt.
//   3. Bias state propagation: Euclidean boxplus, so without input the
//      bias stays put across many predicts (its mean doesn't drift on
//      its own — random-walk drift comes from auto-Q's Q growth, not
//      from x_ref).

#include <doctest/doctest.h>

#include "../include/manta/estimation/craft_view.hpp"
#include "../include/manta/estimation/generic_ekf.hpp"
#include "../include/manta/estimation/state_spec.hpp"
#include "../include/manta/parts/sensor/imu.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/fields/gravity_field.hpp"

#include <memory>
#include <type_traits>

using namespace manta;
using namespace manta::estimation;

namespace {

template <class Scalar>
class BiasTestCraft : public CraftT<Scalar> {
public:
    BiasTestCraft() : CraftT<Scalar>("test") {
        this->root().template add<parts::MassT<Scalar>>("body", Scalar(1.0));
        imu_ = &this->root().template add<parts::IMUT<Scalar>>(
            "imu",
            /*accel_sigma=*/0.05f,
            /*gyro_sigma=*/0.005f,
            /*accel_bias_sigma=*/0.01f,    // RW driver — non-zero ⇒ active
            /*gyro_bias_sigma=*/-1.0f);     // gyro bias not tracked
        this->root().compute_params();
    }

    parts::IMUT<Scalar>& imu() { return *imu_; }

private:
    parts::IMUT<Scalar>* imu_ = nullptr;
};

}

TEST_CASE("Phase 5e: track(imu.accel_bias()) grows the StateSpec") {
    BiasTestCraft<double> craft;
    auto state = make_state()
        .track(craft)
        .track(craft.imu().accel_bias())
        .build();
    using Spec = decltype(state);
    static_assert(Spec::num_slices  == 2);
    static_assert(Spec::ambient_dim == 13 + 3);    // craft + bias
    static_assert(Spec::tangent_dim == 12 + 3);
    static_assert(Spec::template ambient_offset<1> == 13);
    static_assert(Spec::template tangent_offset<1> == 12);
}

TEST_CASE("Phase 5e: predict grows P diagonal at bias slots by σ²·dt") {
    constexpr double DT = 0.001;
    constexpr double accel_bias_sigma = 0.01;

    BiasTestCraft<double> craft;
    auto state = make_state()
        .track(craft)
        .track(craft.imu().accel_bias())
        .build();
    using Spec = decltype(state);

    // 6 noise driver slots from IMU's white channels (accel + gyro)
    // + 3 RW driver slots from the tracked accel_bias.
    constexpr int kNoiseSlots = 9;
    using Ekf = EKFGeneric<Spec, /*MeasDim=*/0, kNoiseSlots>;
    using Jet = typename Ekf::Jet;

    Ekf ekf{state};

    // The value-side craft needs a host world for field mirroring + the
    // EKF's RW-bias resolution to work — point it at a minimal scaffolding.
    WorldT<double> est_world;
    est_world.clock().set_dt(DT);
    fields::GravityField grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    est_world.register_field(grav);
    est_world.create_scene().add_craft(craft);

    CraftView<Ekf, 0> view(ekf, [](auto& w) {
        using S = typename std::remove_reference_t<decltype(w)>::Scalar;
        auto c = std::make_unique<BiasTestCraft<S>>();
        w.create_scene().add_craft(*c);
        return c;
    });

    typename Ekf::StateVec x0 = Ekf::StateVec::Zero();
    x0(3) = 1.0;   // identity quat
    typename Ekf::StateCov P0 = Ekf::StateCov::Zero();
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    typename Ekf::StateCov Q = Ekf::StateCov::Zero();
    ekf.predict(DT, Q);
    const auto& P = ekf.covariance();

    // Bias slot lives at tangent_offset<1> = 12. Three components.
    constexpr int bias_off = Spec::template tangent_offset<1>;
    const double expected_var = accel_bias_sigma * accel_bias_sigma * DT;
    for (int i = 0; i < 3; ++i) {
        CAPTURE(i);
        CHECK(P(bias_off + i, bias_off + i) ==
              doctest::Approx(expected_var).epsilon(0.05));
    }

    // Ten predicts should accumulate to ~10·σ²·dt.
    for (int s = 0; s < 9; ++s) ekf.predict(DT, Q);
    const double expected_after_10 = 10.0 * expected_var;
    for (int i = 0; i < 3; ++i) {
        CAPTURE(i);
        CHECK(P(bias_off + i, bias_off + i) ==
              doctest::Approx(expected_after_10).epsilon(0.10));
    }
}
