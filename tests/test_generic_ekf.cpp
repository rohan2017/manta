// Numerical-equivalence smoke test for `EKFGeneric<StateSpec, ...>`
// against the legacy `EKF<NumCrafts, ...>` on a single-craft scenario.
//
// Setup: one craft with a Mass under gravity. Both filters run the same
// predict over the same Jet world; we check that x_ref and P_ stay
// equal to within float epsilon after each step.

#include <doctest/doctest.h>

#include "../include/manta/estimation/ekf.hpp"
#include "../include/manta/estimation/generic_ekf.hpp"
#include "../include/manta/estimation/state_spec.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"

using namespace manta;
using namespace manta::estimation;

namespace {

template <class Scalar>
class FreeFallCraft : public CraftT<Scalar> {
public:
    FreeFallCraft() : CraftT<Scalar>("ff") {
        this->root().template add<parts::MassT<Scalar>>("body", Scalar(1.0));
        this->root().compute_params();
    }
};

// Thin "no measurements" predict-only test. No measurement functor
// needed since both filters go through pure prediction.

} // namespace

TEST_CASE("EKFGeneric: predict-only matches legacy EKF on free-fall sim") {
    constexpr double DT = 0.01;
    constexpr int    N_STEPS = 50;

    // ---- Legacy EKF ----
    using LegacyEkf = EKF<1, /*MeasDim=*/0, /*NoiseSlots=*/0, /*BiasDim=*/0>;
    using LegacyJet = LegacyEkf::Jet;

    WorldT<LegacyJet> w_jet_legacy;
    w_jet_legacy.clock().set_dt(DT);
    auto& s_legacy = w_jet_legacy.create_scene();
    fields::GravityField grav_legacy{
        geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    w_jet_legacy.register_field(grav_legacy);

    FreeFallCraft<double>    craft_real_legacy;
    FreeFallCraft<LegacyJet> craft_jet_legacy;
    s_legacy.add_craft(craft_jet_legacy);

    LegacyEkf ekf_legacy;
    LegacyEkf::StateVec x0 = LegacyEkf::StateVec::Zero();
    x0(3) = 1.0;   // identity quat
    LegacyEkf::StateCov P0 = LegacyEkf::StateCov::Identity() * 0.01;
    ekf_legacy.set_state(x0);
    ekf_legacy.set_covariance(P0);
    ekf_legacy.bind(w_jet_legacy,
                    std::array<CraftT<double>*,    1>{&craft_real_legacy},
                    std::array<CraftT<LegacyJet>*, 1>{&craft_jet_legacy});

    // ---- Generic EKF ----
    auto state = make_state().track(craft_real_legacy).build();
    using Spec = decltype(state);
    using GenericEkf = EKFGeneric<Spec, /*MeasDim=*/0, /*NoiseSlots=*/0>;
    using GenericJet = GenericEkf::Jet;

    static_assert(std::is_same_v<LegacyJet, GenericJet>,
                  "Jet widths must match for a fair comparison.");

    WorldT<GenericJet> w_jet_generic;
    w_jet_generic.clock().set_dt(DT);
    auto& s_generic = w_jet_generic.create_scene();
    fields::GravityField grav_generic{
        geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    w_jet_generic.register_field(grav_generic);

    FreeFallCraft<GenericJet> craft_jet_generic;
    s_generic.add_craft(craft_jet_generic);

    GenericEkf ekf_generic{state};
    ekf_generic.set_state(x0);   // matches legacy
    ekf_generic.set_covariance(P0);
    ekf_generic.bind(w_jet_generic, { static_cast<void*>(&craft_jet_generic) });

    // ---- Run both, compare ----
    LegacyEkf::StateCov Q = LegacyEkf::StateCov::Zero();

    for (int step = 0; step < N_STEPS; ++step) {
        ekf_legacy.predict(DT, Q);
        ekf_generic.predict(DT, Q);

        const auto& x_legacy  = ekf_legacy.state();
        const auto& x_generic = ekf_generic.state();
        for (int i = 0; i < 13; ++i) {
            CAPTURE(step);
            CAPTURE(i);
            CHECK(x_legacy(i) == doctest::Approx(x_generic(i)).epsilon(1e-9));
        }

        const auto& P_legacy  = ekf_legacy.covariance();
        const auto& P_generic = ekf_generic.covariance();
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                CAPTURE(step);
                CAPTURE(i);
                CAPTURE(j);
                CHECK(P_legacy(i, j) ==
                      doctest::Approx(P_generic(i, j)).epsilon(1e-9));
            }
        }
    }
}
