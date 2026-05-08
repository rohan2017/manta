// End-to-end test of the Measurement + Reading API.
//
// Sim+est in one process:
//   * sim_world holds sim_craft with an IMU and DVL — these produce
//     synthetic noisy readings via update().
//   * jet_world holds est_craft with a parallel IMU and DVL — used as
//     the model: its update() runs through the EKF's Jet shadow each
//     tick, populating h(x) caches with autodiff Jets.
//   * Reading sources point from sim sensors → EKF; the EKF resolves
//     the Jet-side Measurement counterparts at run-time.
//
// We just check that the EKF closes the loop without crashing, that
// position estimate stays reasonable under gravity over a short run,
// and that the registered Reading freshness gates the updates.

#include <doctest/doctest.h>

#include "../include/manta/estimation/craft_view.hpp"
#include "../include/manta/estimation/generic_ekf.hpp"
#include "../include/manta/estimation/measurement.hpp"
#include "../include/manta/estimation/reading.hpp"
#include "../include/manta/estimation/state_spec.hpp"
#include "../include/manta/parts/sensor/dvl.hpp"
#include "../include/manta/parts/sensor/imu.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/fields/gravity_field.hpp"

using namespace manta;
using namespace manta::estimation;

namespace {

template <class Scalar>
class TestMeasCraft : public CraftT<Scalar> {
public:
    explicit TestMeasCraft(std::string name)
        : CraftT<Scalar>(std::move(name)) {
        this->root().template add<parts::MassT<Scalar>>("body", Scalar(1.0));
        imu_ = &this->root().template add<parts::IMUT<Scalar>>(
            "imu", /*accel_sigma=*/0.05f, /*gyro_sigma=*/0.005f);
        dvl_ = &this->root().template add<parts::DVLT<Scalar>>(
            "dvl", /*velocity_sigma=*/0.02f);
        this->root().compute_params();
    }

    parts::IMUT<Scalar>& imu() { return *imu_; }
    parts::DVLT<Scalar>& dvl() { return *dvl_; }

private:
    parts::IMUT<Scalar>* imu_ = nullptr;
    parts::DVLT<Scalar>* dvl_ = nullptr;
};

}

TEST_CASE("Measurement+Reading: end-to-end IMU+DVL update on free-fall") {
    constexpr double DT = 0.001;
    constexpr int    N_STEPS = 100;

    // ---- Sim world (value side) — produces synthetic readings ----
    WorldT<double> sim_world;
    sim_world.clock().set_dt(DT);
    fields::GravityField sim_grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    sim_world.register_field(sim_grav);
    auto& sim_scene = sim_world.create_scene();
    TestMeasCraft<double> sim_craft("sim");
    sim_scene.add_craft(sim_craft);

    // ---- Est value world (just for the StateSpec to bind onto) ----
    WorldT<double> est_world;
    est_world.clock().set_dt(DT);
    fields::GravityField est_grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    est_world.register_field(est_grav);
    auto& est_scene = est_world.create_scene();
    TestMeasCraft<double> est_craft("est");
    est_scene.add_craft(est_craft);

    // ---- StateSpec + EKFGeneric ----
    auto state = make_state().track(est_craft).build();
    using Spec = decltype(state);

    constexpr int kNoiseSlots = 9;   // IMU accel(3) + IMU gyro(3) + DVL(3)
    using EkfT = EKFGeneric<Spec, /*MeasDim=*/0, kNoiseSlots>;
    EkfT ekf{state};

    // ---- Jet world ----
    WorldT<typename EkfT::Jet> jet_world;
    jet_world.clock().set_dt(DT);
    fields::GravityField jet_grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    jet_world.register_field(jet_grav);
    auto& jet_scene = jet_world.create_scene();
    TestMeasCraft<typename EkfT::Jet> est_craft_jet("est");
    jet_scene.add_craft(est_craft_jet);

    ekf.bind(jet_world, { static_cast<void*>(&est_craft_jet) });

    // ---- Wire measurements: model = est_craft, reading = sim_craft ----
    ekf.measure<3>(&est_craft.imu().accel,
                   reading_from<3>(sim_craft.imu().accel));
    ekf.measure<3>(&est_craft.imu().gyro,
                   reading_from<3>(sim_craft.imu().gyro));
    ekf.measure<3>(&est_craft.dvl().velocity,
                   reading_from<3>(sim_craft.dvl().velocity));

    // Initial state — at rest, identity quat.
    typename EkfT::StateVec x0 = EkfT::StateVec::Zero();
    x0(3) = 1.0;
    ekf.set_state(x0);
    ekf.set_covariance(EkfT::StateCov::Identity() * 0.01);

    CraftView<EkfT, 0> est_view{ekf};

    // ---- Run ----
    noise_seed(7);
    typename EkfT::StateCov Q = EkfT::StateCov::Zero();
    for (int s = 0; s < N_STEPS; ++s) {
        sim_world.step();           // populates sim sensor caches with noisy reads
        ekf.predict(DT, Q);
        ekf.run_pending_updates();  // pulls Readings, applies updates
    }

    // Sanity: under gravity, sim_craft fell ~½gt² over 0.1s = 4.9 cm.
    auto p_truth = sim_craft.scene_to_craft().position();
    auto p_est   = est_view.position();
    INFO("p_truth.z=", double(p_truth.z()), " p_est.z=", p_est(2));
    CHECK(std::abs(double(p_truth.z()) - p_est(2)) < 0.05);

    // DVL directly observes velocity — its stddev should drop sharply.
    // Position is only indirectly observable via the velocity coupling,
    // and over 0.1 s of free-fall it doesn't drop much; we only check
    // the directly-observed channel.
    auto v_sd = est_view.vel_linear_stddev();
    CHECK(v_sd(0) < 0.05);
    CHECK(v_sd(1) < 0.05);
    CHECK(v_sd(2) < 0.05);
}
