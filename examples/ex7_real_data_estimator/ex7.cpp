// ex7 — real-data-only estimator (Pattern C from estimation_workflow.md).
//
// No sim Craft in this binary. The estimator runs against external sensor
// data fed via Zenoh — typical robot deployment shape. Topics:
//
//     manta/ex7/imu       ← IMU payload  [ax, ay, az, gx, gy, gz] (float[6])
//     manta/ex7/dvl       ← DVL payload  [vx, vy, vz]              (float[3])
//     manta/ex7/estimate  → EKF output   {t, p[3], v[3], P_diag[6]}
//
// The codegen-emitted Ex7EstCraftT<Scalar> is wrapped in CraftEKF<...>,
// which extracts predict-step Jacobians via Ceres Jets automatically. No
// hand-written process functor.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.hxx>

#include "manta/estimation/craft_ekf.hpp"

#include "ex7_est.hpp"          // Ex7EstCraftT<Scalar>
#include "sim_loop.hpp"
#include "state_codec.hpp"

namespace {
std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }
}

// DVL measurement model: read scene-frame velocity from the rigid state
// vector (state indices 7..9). The craft has identity orientation in this
// demo, so body velocity == scene velocity.
struct DvlVelocityMeas {
    template <class S, int N>
    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, N, 1>& x) const {
        return x.template segment<3>(7);
    }
};

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

    constexpr double DT             = 0.001;
    constexpr int    PUB_DECIM      = 20;        // 50 Hz publish

    manta::estimation::CraftEKF<Ex7EstCraftT, 3> ekf;

    Ex7EstCraftT<double>::RigidState x0;
    x0.setZero();
    x0(3) = 1.0;     // identity quaternion w
    Eigen::Matrix<double, 13, 13> P0 = Eigen::Matrix<double, 13, 13>::Identity();
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    constexpr double DVL_SIGMA = 0.02;
    Eigen::Matrix<double, 13, 13> Q = Eigen::Matrix<double, 13, 13>::Identity() * 1e-6;
    Eigen::Matrix<double, 3, 3>   R = Eigen::Matrix<double, 3, 3>::Identity()
                                    * (DVL_SIGMA * DVL_SIGMA);

    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    // ---- IMU subscriber: feed est.imu().set_measurement(...) ----
    std::mutex meas_mtx;
    std::atomic<bool> have_dvl{false};
    Eigen::Matrix<double, 3, 1> latest_dvl{0, 0, 0};

    auto imu_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex7/imu"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (!manta::examples::parse_float_array(payload, v, /*expected=*/6)) return;
            std::lock_guard<std::mutex> lk(meas_mtx);
            ekf.craft().imu().set_measurement(
                manta::geom::Vec3<manta::PartFrame, double>{
                    double(v[0]), double(v[1]), double(v[2])},
                manta::geom::Vec3<manta::PartFrame, double>{
                    double(v[3]), double(v[4]), double(v[5])});
        }, zenoh::closures::none);

    // ---- DVL subscriber: stash latest reading for next predict tick ----
    auto dvl_sub = session.declare_subscriber(
        zenoh::KeyExpr("manta/ex7/dvl"),
        [&](const zenoh::Sample& s) {
            std::vector<float> v;
            std::string payload(s.get_payload().as_string());
            if (!manta::examples::parse_float_array(payload, v, /*expected=*/3)) return;
            std::lock_guard<std::mutex> lk(meas_mtx);
            latest_dvl(0) = double(v[0]);
            latest_dvl(1) = double(v[1]);
            latest_dvl(2) = double(v[2]);
            ekf.craft().dvl().set_measurement(
                manta::geom::Vec3<manta::PartFrame, double>{
                    latest_dvl(0), latest_dvl(1), latest_dvl(2)});
            have_dvl.store(true);
        }, zenoh::closures::none);

    auto estimate_pub = session.declare_publisher(zenoh::KeyExpr("manta/ex7/estimate"));

    std::printf("ex7: subscribing to manta/ex7/imu and manta/ex7/dvl, "
                "publishing manta/ex7/estimate.\n");

    manta::examples::RealTimePacer pacer(DT);
    int pub_decim = 0;
    int step = 0;
    double t = 0.0;

    while (g_run.load()) {
        // Predict every tick using whatever the latest IMU reading was.
        // (set_measurement is called from the Zenoh callback; predict reads
        // the cached value.)
        {
            std::lock_guard<std::mutex> lk(meas_mtx);
            ekf.predict(DT, Q);
        }

        // Update opportunistically when a fresh DVL reading is available.
        if (have_dvl.exchange(false)) {
            std::lock_guard<std::mutex> lk(meas_mtx);
            ekf.update(DvlVelocityMeas{}, latest_dvl, R);
        }

        if (++pub_decim >= PUB_DECIM) {
            pub_decim = 0;
            estimate_pub.put(zenoh::Bytes(
                encode_estimate(t, ekf.state(), ekf.covariance())));
        }

        t += DT;
        ++step;
        pacer.wait_for_next_tick();
    }

    std::printf("ex7: shutting down.\n");
    return 0;
}
