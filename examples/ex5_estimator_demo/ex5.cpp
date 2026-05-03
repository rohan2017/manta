// ex5 — sim + EKF side-by-side, both publishing to Zenoh.
//
// This is the canonical "specify a craft once, get the EKF for free" demo.
// No hand-written process functor; CraftEKF wraps the templated estimator
// craft and extracts Jacobians via Ceres Jets automatically.
//
// Two craft definitions, both feeding through codegen:
//   * craft.py     → Ex5Craft       (sim, runs in a Scene)
//   * est_craft.py → Ex5EstCraftT<S> (estimator, wrapped in CraftEKF)
//
// They have matching part names (Pattern A) — the sim's noisy IMU + DVL
// outputs flow into the est craft's `set_measurement(...)` hooks each
// tick, and the est-side thruster mirrors the sim-side throttle so the
// predict step uses the same commanded force the sim is applying.
//
// Topics:
//   manta/ex6/state            — truth pose (top-level kinematic state)
//   manta/ex6/estimate         — EKF estimate (13-D state + diag covariance)
//   manta/ex6/thrust/cmd       — float[1] throttle [0,1]

#include <atomic>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.hxx>

#include "manta/core/scene.hpp"
#include "manta/core/world.hpp"
#include "manta/estimation/craft_ekf.hpp"
#include "manta/fields/gravity_field.hpp"

#include "ex5.hpp"           // Ex5Craft (sim)
#include "ex5_telemetry.hpp"
#include "ex5_est.hpp"       // Ex5EstCraftT<Scalar> (estimator)
#include "sim_loop.hpp"
#include "state_codec.hpp"

namespace {
std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }
}

// Measurement model h(x): the DVL reads scene-frame velocity (the craft has
// identity orientation in this demo, so body == scene frame). State indices
// 7..9 are linear velocity in CraftT::RigidState layout:
//   [px py pz | qw qx qy qz | vx vy vz | wx wy wz]
struct DvlVelocityMeas {
    template <class S, int N>
    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, N, 1>& x) const {
        return x.template segment<3>(7);
    }
};

// Encode the EKF estimate as JSON for the viewer. The 13-D rigid state is
// reduced to position + velocity for the wire format (the rest is internal).
static std::string encode_estimate(double t_sec,
                                   const Eigen::Matrix<double, 13, 1>& x,
                                   const Eigen::Matrix<double, 13, 13>& P) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"t\":%.6f,"
        "\"p\":[%.6f,%.6f,%.6f],"
        "\"v\":[%.6f,%.6f,%.6f],"
        "\"P\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]}",
        t_sec,
        x(0), x(1), x(2),
        x(7), x(8), x(9),
        P(0,0), P(1,1), P(2,2), P(7,7), P(8,8), P(9,9));
    return std::string(buf);
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr double DT             = 0.001;     // 1 kHz sim
    constexpr int    DVL_HZ_DECIM   = 20;        // 50 Hz DVL update
    constexpr int    PUB_DECIM      = 20;        // 50 Hz publish to Zenoh

    manta::World w;
    w.clock().set_dt(static_cast<float>(DT));
    auto& scene = w.create_scene();

    manta::fields::GravityField grav;
    w.register_field(grav);

    // Pattern A: sim Craft drives the physics; estimator Craft is a separate
    // instance (NOT in the Scene) wrapped by CraftEKF. The two crafts have
    // matching part names so sensor + command piping is mechanical.
    Ex5Craft sim;
    scene.add_craft(sim, manta::InitialState{});

    // CraftEKF<EstCraftTpl, MeasDim>. State dim is fixed at the rigid 13-DOF
    // layout from CraftT::RigidState. The wrapper internally holds two craft
    // instances (double + Jet) and runs predict/update with autodiff.
    manta::estimation::CraftEKF<Ex5EstCraftT, 3> ekf;

    // The est-side Mass parts auto-apply gravity from a registered
    // GravityField, so the field needs to be visible to both internal
    // crafts (value step + Jacobian step).
    ekf.register_field(grav);

    // Initial state — at origin, identity quaternion, at rest.
    Ex5EstCraftT<double>::RigidState x0;
    x0.setZero();
    x0(3) = 1.0;     // quaternion w
    Eigen::Matrix<double, 13, 13> P0 = Eigen::Matrix<double, 13, 13>::Identity();
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    constexpr double DVL_SIGMA = 0.02;
    Eigen::Matrix<double, 13, 13> Q = Eigen::Matrix<double, 13, 13>::Identity() * 1e-6;
    Eigen::Matrix<double, 3, 3>   R = Eigen::Matrix<double, 3, 3>::Identity()
                                    * (DVL_SIGMA * DVL_SIGMA);

    // ---- Zenoh ----
    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    std::mutex thrust_cmd_mtx;
    std::vector<float> thrust_cmd;
    auto thrust_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex5/thrust/cmd"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (manta::examples::parse_float_array(payload, v, /*expected=*/1)) {
                std::lock_guard<std::mutex> lk(thrust_cmd_mtx);
                thrust_cmd = std::move(v);
            }
        }, zenoh::closures::none);

    auto state_pub    = session.declare_publisher(zenoh::KeyExpr("manta/ex5/state"));
    auto estimate_pub = session.declare_publisher(zenoh::KeyExpr("manta/ex5/estimate"));

    std::printf("ex5: ready. truth on 'manta/ex6/state', estimate on "
                "'manta/ex6/estimate'.\n");

    manta::examples::RealTimePacer pacer(DT);
    int pub_decim = 0;
    int step = 0;

    while (g_run.load()) {
        // Apply latest thrust command to BOTH crafts. Sim drives physics;
        // est mirrors so its predict() step models the same input.
        {
            std::lock_guard<std::mutex> lk(thrust_cmd_mtx);
            if (!thrust_cmd.empty()) {
                sim.thrust().set_throttle(thrust_cmd[0]);
                ekf.craft().thrust().set_throttle(thrust_cmd[0]);
            }
        }

        w.update();

        // Pattern A glue: pipe sim sensor outputs into est sensor measurement
        // inputs. Sim is float-typed (Real); est-side double for filter
        // conditioning. Cast at the boundary.
        // (This block is what a future codegen extension would emit.)
        using V3d = manta::geom::Vec3<manta::PartFrame, double>;
        ekf.craft().imu().set_measurement(
            V3d::from_raw(sim.imu().last_accel().raw().cast<double>()),
            V3d::from_raw(sim.imu().last_gyro().raw().cast<double>()));
        ekf.craft().dvl().set_measurement(
            V3d::from_raw(sim.dvl().last_velocity().raw().cast<double>()));

        // EKF predict — internally calls est.evaluate(x, dt) with both
        // double and Jet to extract value + Jacobian. No hand-written process
        // functor.
        ekf.predict(DT, Q);

        // EKF update at 50 Hz from sim DVL.
        if (step % DVL_HZ_DECIM == DVL_HZ_DECIM - 1) {
            Eigen::Matrix<double, 3, 1> z;
            auto v = sim.dvl().last_velocity();
            z(0) = double(v.x()); z(1) = double(v.y()); z(2) = double(v.z());
            ekf.update(DvlVelocityMeas{}, z, R);
        }

        // Publish at 50 Hz.
        if (++pub_decim >= PUB_DECIM) {
            pub_decim = 0;
            double t = double(w.clock().time());
            state_pub.put(zenoh::Bytes(
                manta::examples::encode_craft_state(t, sim)));
            estimate_pub.put(zenoh::Bytes(
                encode_estimate(t, ekf.state(), ekf.covariance())));
        }

        ++step;
        pacer.wait_for_next_tick();
    }

    std::printf("ex5: shutting down.\n");
    return 0;
}
