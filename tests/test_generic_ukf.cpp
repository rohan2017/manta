// Numerical-equivalence smoke test for `UKFGeneric<StateSpec, ...>`
// against the legacy `UKF<NumCrafts, ...>` on a single-craft scenario.
//
// Setup: one craft with a Mass under gravity, free-fall sim. Both UKFs
// run their predict on the same dynamics; we check that the state and
// covariance stay equal across many steps.

#include <doctest/doctest.h>

#include "../include/manta/estimation/generic_ukf.hpp"
#include "../include/manta/estimation/state_spec.hpp"
#include "../include/manta/estimation/ukf.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"

using namespace manta;
using namespace manta::estimation;

namespace {

template <class Scalar>
class FreeFallCraftU : public CraftT<Scalar> {
public:
    FreeFallCraftU() : CraftT<Scalar>("ff") {
        this->root().template add<parts::MassT<Scalar>>("body", Scalar(1.0));
        this->root().compute_params();
    }
};

} // namespace

TEST_CASE("UKFGeneric: predict-only matches legacy UKF on free-fall sim") {
    constexpr double DT      = 0.01;
    constexpr int    N_STEPS = 50;

    // ---- Legacy UKF ----
    using LegacyUkf = UKF<1, /*MeasDim=*/0, /*BiasDim=*/0>;

    WorldT<double> world_legacy;
    world_legacy.clock().set_dt(DT);
    auto& s_legacy = world_legacy.create_scene();
    fields::GravityField grav_legacy{
        geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    world_legacy.register_field(grav_legacy);
    FreeFallCraftU<double> craft_legacy;
    s_legacy.add_craft(craft_legacy);

    LegacyUkf ukf_legacy{1e-3, 2.0, 0.0};
    LegacyUkf::StateVec x0 = LegacyUkf::StateVec::Zero();
    x0(3) = 1.0;   // identity quat
    LegacyUkf::StateCov P0 = LegacyUkf::StateCov::Identity() * 0.01;
    ukf_legacy.set_state(x0);
    ukf_legacy.set_covariance(P0);
    ukf_legacy.bind(world_legacy,
                    std::array<CraftT<double>*, 1>{&craft_legacy});

    // ---- Generic UKF (separate world to avoid stomping the legacy one) ----
    WorldT<double> world_generic;
    world_generic.clock().set_dt(DT);
    auto& s_generic = world_generic.create_scene();
    fields::GravityField grav_generic{
        geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    world_generic.register_field(grav_generic);
    FreeFallCraftU<double> craft_generic;
    s_generic.add_craft(craft_generic);

    auto state = make_state().track(craft_generic).build();
    UKFGeneric<decltype(state), /*MeasDim=*/0> ukf_generic{state, 1e-3, 2.0, 0.0};
    ukf_generic.set_state(x0);
    ukf_generic.set_covariance(P0);
    ukf_generic.bind(world_generic);

    // ---- Run both, compare ----
    using StateCov = LegacyUkf::StateCov;
    StateCov Q = StateCov::Zero();

    for (int step = 0; step < N_STEPS; ++step) {
        ukf_legacy.predict(DT, Q);
        ukf_generic.predict(DT, Q);

        const auto& x_legacy  = ukf_legacy.state();
        const auto& x_generic = ukf_generic.state();
        for (int i = 0; i < 13; ++i) {
            CAPTURE(step);
            CAPTURE(i);
            CHECK(x_legacy(i) == doctest::Approx(x_generic(i)).epsilon(1e-6));
        }

        const auto& P_legacy  = ukf_legacy.covariance();
        const auto& P_generic = ukf_generic.covariance();
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                CAPTURE(step);
                CAPTURE(i);
                CAPTURE(j);
                CHECK(P_legacy(i, j) ==
                      doctest::Approx(P_generic(i, j)).epsilon(1e-6));
            }
        }
    }
}
