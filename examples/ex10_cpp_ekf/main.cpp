// ex10_cpp_ekf — hand-written reference implementation of the
// StateSpec + EKFGeneric + Measurement + Reading API.
//
// Pure C++, no codegen, no Zenoh. One binary running:
//
//   * a sim world with a thrust-driven craft (IMU + DVL),
//   * a parallel est world bound to an EKFGeneric tracking the est craft,
//   * Reading sources wired sim → est measurement-by-measurement.
//
// Logs the truth-vs-estimate trajectory to stdout each second and
// exits cleanly after a fixed run.
//
// This is the reference shape for the codegen migration: the codegen
// emits code shaped like this main, with the boilerplate (world setup,
// craft construction, ekf.measure() calls, tick body) generated from
// the user's Python config.

#include <cstdio>

#include "craft.hpp"
#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/estimation/craft_view.hpp"
#include "manta/estimation/generic_ekf.hpp"
#include "manta/estimation/measurement.hpp"
#include "manta/estimation/reading.hpp"
#include "manta/estimation/state_spec.hpp"
#include "manta/fields/gravity_field.hpp"

using namespace manta;
using namespace manta::estimation;
using ex10::DemoCraftT;

int main() {
    constexpr double DT       = 0.001;     // 1 kHz
    constexpr int    N_STEPS  = 5000;      // 5 s of sim
    constexpr int    LOG_EVERY = 1000;     // 1 Hz log

    // ---- Sim world (the truth): produces noisy synthetic readings. ----
    WorldT<double> sim_world;
    sim_world.clock().set_dt(DT);
    fields::GravityField sim_grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    sim_world.register_field(sim_grav);
    auto& sim_scene = sim_world.create_scene();
    DemoCraftT<double> sim_craft("sim");
    sim_scene.add_craft(sim_craft);
    sim_craft.thrust().set_throttle(1.0f);   // full thrust along +x

    // ---- Est value world (EKF's tracked craft). ----
    WorldT<double> est_world;
    est_world.clock().set_dt(DT);
    fields::GravityField est_grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    est_world.register_field(est_grav);
    auto& est_scene = est_world.create_scene();
    DemoCraftT<double> est_craft("est");
    est_scene.add_craft(est_craft);
    // The est craft mirrors the sim's actuator state every tick so the
    // process model integrates the same input the sim is applying.
    est_craft.thrust().set_throttle(1.0f);

    // ---- StateSpec + EKFGeneric ----
    auto state = make_state().track(est_craft).build();
    using Spec = decltype(state);
    constexpr int kNoiseSlots = 12;  // IMU accel(3) + IMU gyro(3) + DVL(3) + Thruster force(3)
    using EkfT = EKFGeneric<Spec, /*MeasDim=*/0, kNoiseSlots>;
    EkfT ekf{state};

    // ---- Jet shadow world (autodiff). ----
    using Jet = EkfT::Jet;
    WorldT<Jet> jet_world;
    jet_world.clock().set_dt(DT);
    fields::GravityField jet_grav{geom::Vec3<SceneFrame>{0.0f, 0.0f, -9.81f}};
    jet_world.register_field(jet_grav);
    auto& jet_scene = jet_world.create_scene();
    DemoCraftT<Jet> est_craft_jet("est");
    jet_scene.add_craft(est_craft_jet);

    ekf.bind(jet_world, { static_cast<void*>(&est_craft_jet) });
    est_craft_jet.thrust().set_throttle(Jet(1.0));

    // ---- Wire the measurements. ----
    // model = est_craft's IMU/DVL (h(x) + R)
    // reading = sim_craft's IMU/DVL (z source — synthetic noisy readings)
    ekf.measure<3>(&est_craft.imu().accel,
                   reading_from<3>(sim_craft.imu().accel));
    ekf.measure<3>(&est_craft.imu().gyro,
                   reading_from<3>(sim_craft.imu().gyro));
    ekf.measure<3>(&est_craft.dvl().velocity,
                   reading_from<3>(sim_craft.dvl().velocity));

    // ---- Initial belief: at rest at origin, moderate covariance. ----
    CraftView<EkfT, 0> est_view{ekf};
    est_view.reset_to_rest();
    est_view.set_state_covariance(/*pos_var=*/1e-4,
                                  /*attitude_var=*/1e-4,
                                  /*vel_var=*/1e-2,
                                  /*angvel_var=*/1e-4);

    // ---- Tick loop. ----
    noise_seed(7);
    EkfT::StateCov Q = EkfT::StateCov::Zero();   // Q comes from auto-Q + L·Σ·L^T

    std::printf("step    truth_x    truth_z    est_x      est_z      "
                "vel_sd_x\n");
    for (int s = 0; s < N_STEPS; ++s) {
        sim_world.step();
        ekf.predict(DT, Q);
        ekf.run_pending_updates();

        if (s % LOG_EVERY == 0) {
            auto p_truth = sim_craft.scene_to_craft().position();
            auto p_est   = est_view.position();
            auto v_sd    = est_view.vel_linear_stddev();
            std::printf("%5d  %9.4f  %9.4f  %9.4f  %9.4f  %8.5f\n",
                        s,
                        double(p_truth.x()), double(p_truth.z()),
                        p_est(0), p_est(2),
                        v_sd(0));
        }
    }

    auto p_truth = sim_craft.scene_to_craft().position();
    auto p_est   = est_view.position();
    std::printf("\nfinal:  truth=(%6.3f, %6.3f, %6.3f)  est=(%6.3f, %6.3f, %6.3f)\n",
                double(p_truth.x()), double(p_truth.y()), double(p_truth.z()),
                p_est(0), p_est(1), p_est(2));
    return 0;
}
