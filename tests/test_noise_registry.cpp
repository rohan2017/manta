// Verifies the NoiseRegistry + auto-Q pipeline end-to-end.
//
//  1. A Thruster with force_noise is registered with the EKF at bind time.
//     The registry slots are promoted to global Jet-column indices.
//  2. EKF::predict assembles Q automatically from L · diag(σᵢ²) · Lᵀ
//     plus any user-supplied Q.
//  3. State-dependent σ: mutating the noise's sigma between predicts
//     changes the next tick's auto-Q contribution.
//
// manta's integrator uses a cached acc (force from tick k is reflected
// in state at tick k+1), so a single predict produces a small L. The
// tests below predict over multiple ticks so the noise has propagated
// through the integrator chain.

#include <cmath>
#include <doctest/doctest.h>
#include <ceres/jet.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/estimation/ekf.hpp"
#include "../include/manta/parts/actuator/thruster.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::geom;

namespace {

template <class Scalar>
class NoisyThrusterCraft : public manta::CraftT<Scalar> {
public:
    NoisyThrusterCraft() : manta::CraftT<Scalar>("noisy") {
        auto& thr =
            this->root().template add<manta::parts::Thruster1T<Scalar>>(
                "thr", Scalar(1.0), Vec3<PartFrame, Scalar>{Scalar(0), Scalar(0), Scalar(1)});
        this->root().template add<manta::parts::MassT<Scalar>>("body", Scalar(1.0));
        thr.set_throttle(Scalar(0));
        this->root().compute_params();
    }

    auto& thruster() { return *static_cast<manta::parts::Thruster1T<Scalar>*>(
        this->root().children()->at(0).get()); }
};

} // namespace

TEST_CASE("NoiseRegistry: Thruster's force_noise is registered at bind time") {
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*NumNoiseSlots=*/3>;
    using Jet = Ekf::Jet;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    NoisyThrusterCraft<double> craft_real;
    s_real.add_craft(craft_real);

    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    NoisyThrusterCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    // Slot starts at kNoiseColStart (= kTangentDim = 12 for one craft).
    const int slot = craft_jet.thruster().force_noise().slot();
    CHECK(slot == Ekf::kNoiseColStart);
}

TEST_CASE("NoiseRegistry: predict auto-assembles Q from force_noise") {
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*NumNoiseSlots=*/3>;
    using Jet = Ekf::Jet;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    NoisyThrusterCraft<double> craft_real;
    s_real.add_craft(craft_real);

    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    NoisyThrusterCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    craft_jet.thruster().force_noise().set_sigma(0.5f);

    Ekf::StateVec x0 = Ekf::StateVec::Zero();
    x0(3) = 1.0;        // identity quaternion w
    ekf.set_state(x0);
    ekf.set_covariance(Ekf::StateCov::Identity() * 1e-12);

    Ekf::StateCov Q_user = Ekf::StateCov::Zero();
    constexpr double dt = 0.01;

    // Run several predicts so the cached-acc → state propagation has had
    // time to integrate the noise into v and then into p.
    for (int i = 0; i < 10; ++i) ekf.predict(dt, Q_user);

    const auto& P = ekf.covariance();
    const double vel_var = P(6, 6) + P(7, 7) + P(8, 8);
    const double pos_var = P(0, 0) + P(1, 1) + P(2, 2);
    INFO("vel_var=", vel_var, " pos_var=", pos_var);
    CHECK(vel_var > 1e-8);   // grew from 1e-12 init
    CHECK(pos_var > 1e-12);  // p picks up noise via v over time
}

TEST_CASE("NoiseRegistry: state-dependent sigma propagates into Q") {
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*NumNoiseSlots=*/3>;
    using Jet = Ekf::Jet;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    NoisyThrusterCraft<double> craft_real;
    s_real.add_craft(craft_real);

    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    NoisyThrusterCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    Ekf::StateVec x0 = Ekf::StateVec::Zero();
    x0(3) = 1.0;
    Ekf::StateCov P_init = Ekf::StateCov::Identity() * 1e-12;
    Ekf::StateCov Q_user = Ekf::StateCov::Zero();
    constexpr double dt = 0.01;

    // Pass 1: σ = 0.1 over 10 ticks.
    ekf.set_state(x0);
    ekf.set_covariance(P_init);
    craft_jet.thruster().force_noise().set_sigma(0.1f);
    for (int i = 0; i < 10; ++i) ekf.predict(dt, Q_user);
    const double vel_var_low = ekf.covariance()(6, 6);

    // Pass 2: σ = 1.0 over 10 ticks. 100× σ² → ~100× larger Q
    // contribution per tick, accumulated over the same horizon.
    ekf.set_state(x0);
    ekf.set_covariance(P_init);
    craft_jet.thruster().force_noise().set_sigma(1.0f);
    for (int i = 0; i < 10; ++i) ekf.predict(dt, Q_user);
    const double vel_var_high = ekf.covariance()(6, 6);

    INFO("vel_var_low=", vel_var_low, " vel_var_high=", vel_var_high);
    // Linear-noise scaling: covariance is quadratic in σ (Σ = σ² → factor
    // of 100). Generous band to absorb float rounding and any cross-terms.
    CHECK(vel_var_high > 50.0 * vel_var_low);
    CHECK(vel_var_high < 200.0 * vel_var_low);
}
