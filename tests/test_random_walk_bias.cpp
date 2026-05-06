// Phase E: Noise<RandomWalk> as an EKF-estimated bias state.
//
// The setup: a custom test part has a Noise<RandomWalk> bias that adds
// to its reported "measurement" output. The EKF augments its tangent
// state by the bias DOFs, propagates `bias_post = bias_pre + driver·
// σ_rw·√dt`, and corrects the bias from a measurement that depends on
// it.
//
// Three checks:
//   1. Slot bookkeeping is consistent at bind time.
//   2. predict() grows the bias-state covariance by ~σ²·dt per tick.
//   3. update_n() pulls the bias estimate toward truth from a direct
//      bias measurement.

#include <cmath>
#include <doctest/doctest.h>
#include <ceres/jet.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/noise.hpp"
#include "../include/manta/core/noise_registry.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/estimation/ekf.hpp"
#include "../include/manta/estimation/ukf.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::geom;

namespace {

// A "measurement" part that exposes a 3-axis reading equal to a
// constant true value plus a random-walk bias. No physics — exists
// purely to expose the bias as a filter-observable quantity.
template <class Scalar>
class BiasObserverPart : public PartT<Scalar> {
public:
    explicit BiasObserverPart(std::string name)
        : PartT<Scalar>(std::move(name)) {}

    void update() override {
        // Read the bias on the Jet path so its state-tangent slot
        // partials propagate. We don't apply any wrench — this part is
        // a pure observation channel.
        auto truth = Vec3<PartFrame, Scalar>{Scalar(0), Scalar(0), Scalar(0)};
        last_value_ = truth + bias_;
    }

    void register_noise(NoiseRegistry& r) override {
        r.register_random_walk_3d(bias_);
    }

    Noise<RandomWalk>& bias() noexcept { return bias_; }
    const Noise<RandomWalk>& bias() const noexcept { return bias_; }

    const Vec3<PartFrame, Scalar>& last_value() const noexcept { return last_value_; }

private:
    Noise<RandomWalk>           bias_;
    Vec3<PartFrame, Scalar>     last_value_;
};

template <class Scalar>
class BiasObserverCraft : public manta::CraftT<Scalar> {
public:
    BiasObserverCraft() : manta::CraftT<Scalar>("bias_obs") {
        this->root().template add<manta::parts::MassT<Scalar>>("body", Scalar(1.0));
        this->root().template add<BiasObserverPart<Scalar>>("obs");
        this->root().compute_params();
    }

    auto& observer() {
        return *static_cast<BiasObserverPart<Scalar>*>(
            this->root().children()->at(1).get());
    }
};

} // namespace

TEST_CASE("RW bias: registry assigns state and driver slots") {
    // 3-axis RW bias → 3 state slots + 3 driver slots.
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*NumNoiseSlots=*/3, /*BiasDim=*/3>;
    using Jet = Ekf::Jet;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    BiasObserverCraft<double> craft_real;
    s_real.add_craft(craft_real);

    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    BiasObserverCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    auto& bias = craft_jet.observer().bias();
    // Bias state lives at the start of the augmented tangent's bias
    // range = kBiasColStart = kRigidTangentDim = 12.
    CHECK(bias.state_slot()  == Ekf::kBiasColStart);
    // Driver slot lives in the noise-input range.
    CHECK(bias.driver_slot() == Ekf::kNoiseColStart);
}

TEST_CASE("RW bias: predict grows bias-state covariance by σ²·dt per tick") {
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*NumNoiseSlots=*/3, /*BiasDim=*/3>;
    using Jet = Ekf::Jet;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    BiasObserverCraft<double> craft_real;
    s_real.add_craft(craft_real);

    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    BiasObserverCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    const float sigma_rw = 0.05f;
    craft_jet.observer().bias().set_sigma(sigma_rw);

    Ekf::StateVec x0 = Ekf::StateVec::Zero();
    x0(3) = 1.0;        // identity quaternion
    ekf.set_state(x0);

    // Tiny initial covariance so noise growth dominates.
    Ekf::StateCov P0 = Ekf::StateCov::Zero();
    ekf.set_covariance(P0);

    Ekf::StateCov Q_user = Ekf::StateCov::Zero();
    constexpr double dt = 0.01;
    constexpr int    N  = 100;     // 1 s of predicts

    for (int i = 0; i < N; ++i) ekf.predict(dt, Q_user);

    // After N ticks of σ²·dt growth, bias variance ≈ N·σ²·dt.
    const auto& P = ekf.covariance();
    const int bias_row = Ekf::kBiasColStart;
    const double bias_var_x = P(bias_row + 0, bias_row + 0);
    const double expected   = N * sigma_rw * sigma_rw * dt;

    INFO("bias_var=", bias_var_x, " expected≈", expected);
    CHECK(bias_var_x > 0.5 * expected);
    CHECK(bias_var_x < 2.0 * expected);
}

TEST_CASE("RW bias: update_n pulls the bias estimate toward measured truth") {
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*NumNoiseSlots=*/3, /*BiasDim=*/3>;
    using Jet = Ekf::Jet;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    BiasObserverCraft<double> craft_real;
    s_real.add_craft(craft_real);

    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    BiasObserverCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    craft_jet.observer().bias().set_sigma(0.05f);

    Ekf::StateVec x0 = Ekf::StateVec::Zero();
    x0(3) = 1.0;
    ekf.set_state(x0);
    Ekf::StateCov P0 = Ekf::StateCov::Identity() * 0.1;
    ekf.set_covariance(P0);

    // Start with a wrong bias estimate.
    craft_jet.observer().bias().set_state3(
        Eigen::Matrix<float, 3, 1>{0.f, 0.f, 0.f});

    // Measurement functor: reads the observer's last_value, which equals
    // the bias.
    struct BiasMeas {
        Eigen::Matrix<Jet, 3, 1> operator()(
            const Eigen::Matrix<Jet, Ekf::kStateDim, 1>& /*x_jet*/) const
        {
            // The Jet world's kinematic_and_aggregate populated last_value,
            // but we don't have direct access here — this functor signature
            // is the legacy update_n path. Use add_update via the fused
            // path instead.
            return Eigen::Matrix<Jet, 3, 1>::Zero();
        }
    };

    // Use the fused step path so we can read the observer's last_value
    // from the Jet craft.
    Ekf::StateCov Q = Ekf::StateCov::Zero();
    ekf.begin_step(0.01, Q);

    auto h = [](const Ekf& ekf_in) {
        const auto& obs = const_cast<Ekf&>(ekf_in).craft_jet(0);
        // Walk to the BiasObserverPart.
        auto* bo = static_cast<BiasObserverPart<Jet>*>(
            obs.root().children()->at(1).get());
        const auto& v = bo->last_value();
        return v.raw();   // Eigen::Matrix<Jet, 3, 1>
    };

    // Measured truth: bias is actually (0.2, -0.1, 0.05).
    Eigen::Matrix<double, 3, 1> z;
    z << 0.2, -0.1, 0.05;
    Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity() * 1e-4;
    ekf.add_update<3>(h, z, R);
    ekf.end_step();

    // After the update, the Jet-side bias should have moved toward z.
    const auto bias_post = craft_jet.observer().bias().state3();
    INFO("bias_post=(", bias_post(0), ",", bias_post(1), ",", bias_post(2), ")");
    CHECK(bias_post(0) > 0.05f);   // moved a lot toward 0.2
    CHECK(bias_post(1) < -0.02f);  // moved toward -0.1
    CHECK(bias_post(2) > 0.01f);   // moved toward 0.05
}

// ---- UKF: same bias-state machinery but no Jet shadow ----

TEST_CASE("UKF: RW bias predict grows bias-state covariance by σ²·dt") {
    using Ukf = manta::estimation::UKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*BiasDim=*/3>;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    BiasObserverCraft<double> craft_real;
    s_real.add_craft(craft_real);

    Ukf ukf;
    ukf.bind(w_real, {&craft_real});

    const float sigma_rw = 0.05f;
    craft_real.observer().bias().set_sigma(sigma_rw);

    Ukf::StateVec x0 = Ukf::StateVec::Zero();
    x0(3) = 1.0;
    ukf.set_state(x0);

    // Tiny non-zero initial P — UKFKernel's LLT cholesky needs the
    // covariance to be strictly positive definite. The 1e-10 floor is
    // small relative to σ²·dt over a single tick (2.5e-5) so doesn't
    // perturb the variance growth measurement.
    Ukf::StateCov P0 = Ukf::StateCov::Identity() * 1e-10;
    ukf.set_covariance(P0);

    Ukf::StateCov Q_user = Ukf::StateCov::Zero();
    constexpr double dt = 0.01;
    constexpr int    N  = 100;

    for (int i = 0; i < N; ++i) ukf.predict(dt, Q_user);

    const auto& P = ukf.covariance();
    const int bias_row = Ukf::kBiasColStart;
    const double bias_var_x = P(bias_row + 0, bias_row + 0);
    const double expected   = N * sigma_rw * sigma_rw * dt;

    INFO("UKF bias_var=", bias_var_x, " expected≈", expected);
    CHECK(bias_var_x > 0.5 * expected);
    CHECK(bias_var_x < 2.0 * expected);
}

TEST_CASE("UKF: RW bias measurement pulls estimate toward truth") {
    using Ukf = manta::estimation::UKF</*NumCrafts=*/1, /*MeasDim=*/3,
                                       /*BiasDim=*/3>;

    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    BiasObserverCraft<double> craft_real;
    s_real.add_craft(craft_real);

    Ukf ukf;
    ukf.bind(w_real, {&craft_real});

    craft_real.observer().bias().set_sigma(0.05f);

    Ukf::StateVec x0 = Ukf::StateVec::Zero();
    x0(3) = 1.0;
    ukf.set_state(x0);
    ukf.set_covariance(Ukf::StateCov::Identity() * 0.1);

    craft_real.observer().bias().set_state3(
        Eigen::Matrix<float, 3, 1>{0.f, 0.f, 0.f});

    // Measurement functor reads the BiasObserverPart's last_value via
    // the bound craft. The functor takes a 13·N x_full vector but
    // doesn't need to look at it — it walks straight to the part.
    auto h = [&craft_real](const Ukf::StateVec& /*x_full*/) {
        // The wrapper has just run kinematic_and_aggregate at this
        // sigma's lift, so observer().last_value() is current.
        const auto& v = craft_real.observer().last_value();
        return v.raw();
    };

    Eigen::Matrix<double, 3, 1> z;
    z << 0.2, -0.1, 0.05;
    Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity() * 1e-4;
    ukf.update_n<3>(h, z, R);

    const auto bias_post = craft_real.observer().bias().state3();
    INFO("UKF bias_post=(", bias_post(0), ",", bias_post(1), ",", bias_post(2), ")");
    CHECK(bias_post(0) > 0.05f);
    CHECK(bias_post(1) < -0.02f);
    CHECK(bias_post(2) > 0.01f);
}
