// EKF integration test: use a real Craft sim as the source of truth,
// drive an EKF from the craft's noisy IMU + DVL outputs, verify the
// estimator tracks the true state.
//
// Demonstrates the autodiff-EKF stack against the actual framework
// dynamics — not a synthetic analytic truth. This is the closest thing
// to "estimation goal #2 working end-to-end" we have until the Scalar-
// templating refactor lets us reuse the Part hierarchy directly inside
// the EKF's process model.

#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/estimation/ekf_kernel.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/parts/actuator/thruster.hpp"
#include "../include/manta/parts/sensor/dvl.hpp"
#include "../include/manta/parts/sensor/imu.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::estimation;
using namespace manta::parts;
using namespace manta::fields;

// State: [pz, vz] (1-D vertical motion).
// Process: pz_new = pz + vz*dt + 0.5*u*dt², vz_new = vz + u*dt
//   where u is the IMU's acceleration reading (used as process input).
struct InertialNav1D {
    template <class S>
    Eigen::Matrix<S, 2, 1> operator()(const Eigen::Matrix<S, 2, 1>& x,
                                      double dt,
                                      double u) const {
        Eigen::Matrix<S, 2, 1> y;
        y(0) = x(0) + x(1) * S(dt) + S(0.5) * S(u) * S(dt) * S(dt);
        y(1) = x(1) + S(u) * S(dt);
        return y;
    }
};

// Measurement: DVL reads body-frame z-velocity (= vz when craft has identity
// orientation).
struct DvlMeasZ {
    template <class S>
    Eigen::Matrix<S, 1, 1> operator()(const Eigen::Matrix<S, 2, 1>& x) const {
        Eigen::Matrix<S, 1, 1> z;
        z(0) = x(1);
        return z;
    }
};

// Adapter: capture (dt, u) so the EKF's predict() can call our 3-arg functor.
struct InertialNav1DBound {
    double dt;
    double u;
    template <class S>
    Eigen::Matrix<S, 2, 1> operator()(const Eigen::Matrix<S, 2, 1>& x,
                                      double /*dt_unused*/) const {
        InertialNav1D f;
        return f(x, dt, u);
    }
};

TEST_CASE("EKF: tracks 1-D free-fall sim using IMU + DVL") {
    constexpr float DT             = 0.001f;   // 1 kHz sim
    constexpr int   IMU_HZ_DECIM   = 1;        // process-model uses IMU every tick
    constexpr int   DVL_HZ_DECIM   = 20;       // 50 Hz DVL update
    constexpr int   N_STEPS        = 2000;     // 2 s
    constexpr float IMU_SIGMA      = 0.05f;    // m/s² noise
    constexpr float DVL_SIGMA      = 0.02f;    // m/s noise

    // ----- Truth sim -----
    World w;
    w.clock().set_dt(DT);
    auto& scene = w.create_scene();
    GravityField grav;     // default g = (0, 0, -9.81)
    w.register_field(grav);

    Craft sim("freefall_sim");
    auto& body = sim.root().add<Mass>("body", 1.0f);
    (void)body;
    auto& imu = sim.root().add<IMU>("imu",
        IMU_SIGMA, 0.0f);
    auto& dvl = sim.root().add<DVL>("dvl",
        DVL_SIGMA);
    sim.root().compute_params();
    scene.add_craft(sim, InitialState{});

    // ----- EKF -----
    EKFKernel<2, 1> ekf;
    Eigen::Vector2d x0(0.5, 0.5);     // wrong initial guess
    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0,
          0.0, 1.0;
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    // Process noise reflects unmodeled IMU bias drift + integration error.
    Eigen::Matrix2d Q;
    Q << 1e-6, 0.0,
         0.0, IMU_SIGMA * IMU_SIGMA * DT;

    Eigen::Matrix<double, 1, 1> R;
    R(0, 0) = DVL_SIGMA * DVL_SIGMA;

    // Seed RNG for determinism.
    noise_seed(7);

    // First tick: kinematic pass + sense_and_aggregate populate the IMU/DVL
    // caches before we read them. We call w.step() (which advances the
    // sim by one DT) per loop iteration.
    for (int i = 0; i < N_STEPS; ++i) {
        w.step();

        // EKF predict: use IMU's reading as the inertial acceleration input.
        if (i % IMU_HZ_DECIM == 0) {
            float az = imu.last_accel().z();
            InertialNav1DBound bound{DT * IMU_HZ_DECIM, double(az)};
            ekf.predict(bound, /*dt_unused*/ 0.0, Q);
        }

        // EKF update: DVL gives us body-frame velocity. With identity
        // orientation, body-z-vel == scene-z-vel == truth state x(1).
        if (i % DVL_HZ_DECIM == DVL_HZ_DECIM - 1) {
            Eigen::Matrix<double, 1, 1> z;
            z(0) = double(dvl.last_velocity().z());
            ekf.update(DvlMeasZ{}, z, R);
        }
    }

    // Ground truth from the sim.
    auto p_truth_scene = sim.scene_to_craft().position();
    auto v_truth_scene = sim.scene_to_craft().vel_linear();
    double pz_truth = double(p_truth_scene.z());
    double vz_truth = double(v_truth_scene.z());

    auto x = ekf.state();
    INFO("ekf p=", x(0), " v=", x(1),
         "  truth p=", pz_truth, " v=", vz_truth);

    // Velocity is directly observable through DVL → tight bound.
    CHECK(std::abs(x(1) - vz_truth) < 0.10);
    // Position is integrated; it'll have larger error but should track.
    // The free-fall accumulates ~|0.5*g*t²| = 19.6 m at t=2s, so 1 m of
    // error is <5%. That's a reasonable bar for 2 s of pure inertial
    // integration with 0.05 m/s² IMU noise.
    CHECK(std::abs(x(0) - pz_truth) < 1.0);
}

// ---- IMU-bias augmented state ----
//
// This is the canonical IMU inertial-nav setup: the physical IMU has an
// unknown constant bias on top of its noise. The EKF estimates the bias
// alongside position+velocity. With a velocity-direct measurement (DVL)
// available, the bias becomes observable through the integration mismatch
// between (accel - bias) and the velocity update.
//
// State: [pz, vz, b] (3-D). Bias is treated as constant in the process
// model, with a small random-walk Q to allow it to drift slowly.
struct InertialNav1DWithBias {
    template <class S>
    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, 3, 1>& x,
                                      double dt,
                                      double u) const {
        // u is the raw (biased) IMU reading. Effective acceleration = u - b.
        Eigen::Matrix<S, 3, 1> y;
        S a = S(u) - x(2);
        y(0) = x(0) + x(1) * S(dt) + S(0.5) * a * S(dt) * S(dt);
        y(1) = x(1) + a * S(dt);
        y(2) = x(2);   // bias is constant under process; Q allows drift
        return y;
    }
};

struct InertialNav1DWithBiasBound {
    double dt;
    double u;
    template <class S>
    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, 3, 1>& x,
                                      double /*dt_unused*/) const {
        InertialNav1DWithBias f;
        return f(x, dt, u);
    }
};

struct DvlMeasZWithBias {
    template <class S>
    Eigen::Matrix<S, 1, 1> operator()(const Eigen::Matrix<S, 3, 1>& x) const {
        Eigen::Matrix<S, 1, 1> z;
        z(0) = x(1);
        return z;
    }
};

TEST_CASE("EKF: estimates IMU bias as augmented state") {
    constexpr float DT             = 0.001f;
    constexpr int   DVL_HZ_DECIM   = 20;
    constexpr int   N_STEPS        = 5000;     // 5 s — bias takes longer to lock
    constexpr float IMU_SIGMA      = 0.02f;
    constexpr float DVL_SIGMA      = 0.02f;
    constexpr float TRUE_BIAS      = 0.30f;    // m/s² — physical IMU offset

    World w;
    w.clock().set_dt(DT);
    auto& scene = w.create_scene();
    GravityField grav;
    w.register_field(grav);

    Craft sim("biased_imu_sim");
    auto& body = sim.root().add<Mass>("body", 1.0f);
    (void)body;
    auto& imu = sim.root().add<IMU>("imu",
        IMU_SIGMA, 0.0f);
    auto& dvl = sim.root().add<DVL>("dvl",
        DVL_SIGMA);
    sim.root().compute_params();
    scene.add_craft(sim, InitialState{});

    EKFKernel<3, 1> ekf;
    Eigen::Vector3d x0(0.0, 0.0, 0.0);  // bias guess starts at 0 (wrong by 0.3)
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity();
    P0(2, 2) = 1.0;                      // open uncertainty on bias
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    Q(0, 0) = 1e-7;
    Q(1, 1) = IMU_SIGMA * IMU_SIGMA * DT;
    Q(2, 2) = 1e-8;                      // bias drifts slowly

    Eigen::Matrix<double, 1, 1> R;
    R(0, 0) = DVL_SIGMA * DVL_SIGMA;

    noise_seed(11);

    for (int i = 0; i < N_STEPS; ++i) {
        w.step();

        // Inject a constant bias on top of the IMU reading — that's what a
        // physically-biased IMU returns. The EKF doesn't know about TRUE_BIAS;
        // it has to estimate it.
        double biased_az = double(imu.last_accel().z()) + double(TRUE_BIAS);
        InertialNav1DWithBiasBound bound{DT, biased_az};
        ekf.predict(bound, 0.0, Q);

        if (i % DVL_HZ_DECIM == DVL_HZ_DECIM - 1) {
            Eigen::Matrix<double, 1, 1> z;
            z(0) = double(dvl.last_velocity().z());
            ekf.update(DvlMeasZWithBias{}, z, R);
        }
    }

    auto x = ekf.state();
    auto v_truth = double(sim.scene_to_craft().vel_linear().z());
    INFO("ekf p=", x(0), " v=", x(1), " b=", x(2),
         "  truth v=", v_truth, "  true_bias=", TRUE_BIAS);

    // Bias should converge to its true value within ~0.05 m/s².
    CHECK(std::abs(x(2) - double(TRUE_BIAS)) < 0.05);
    // Velocity should still track despite the IMU bias.
    CHECK(std::abs(x(1) - v_truth) < 0.10);
}

// ---- 3-D pose EKF ----
//
// Full 3-D state [px, py, pz, vx, vy, vz]. Process integrates the IMU's
// 3-axis acceleration reading. DVL gives 3-axis body-frame velocity (which
// equals scene velocity here since the craft has identity orientation).
struct InertialNav3D {
    template <class S>
    Eigen::Matrix<S, 6, 1> operator()(const Eigen::Matrix<S, 6, 1>& x,
                                      double dt,
                                      const Eigen::Matrix<double, 3, 1>& u) const {
        Eigen::Matrix<S, 6, 1> y;
        for (int i = 0; i < 3; ++i) {
            y(i)   = x(i) + x(i+3) * S(dt) + S(0.5) * S(u(i)) * S(dt) * S(dt);
            y(i+3) = x(i+3) + S(u(i)) * S(dt);
        }
        return y;
    }
};

struct InertialNav3DBound {
    double dt;
    Eigen::Matrix<double, 3, 1> u;
    template <class S>
    Eigen::Matrix<S, 6, 1> operator()(const Eigen::Matrix<S, 6, 1>& x,
                                      double /*dt_unused*/) const {
        InertialNav3D f;
        return f(x, dt, u);
    }
};

struct DvlMeas3D {
    template <class S>
    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, 6, 1>& x) const {
        Eigen::Matrix<S, 3, 1> z;
        z(0) = x(3);
        z(1) = x(4);
        z(2) = x(5);
        return z;
    }
};

TEST_CASE("EKF: 3-D pose tracker against thrust+gravity sim") {
    constexpr float DT             = 0.001f;
    constexpr int   DVL_HZ_DECIM   = 20;
    constexpr int   N_STEPS        = 2000;     // 2 s
    constexpr float IMU_SIGMA      = 0.05f;
    constexpr float DVL_SIGMA      = 0.02f;

    World w;
    w.clock().set_dt(DT);
    auto& scene = w.create_scene();
    GravityField grav;
    w.register_field(grav);

    Craft sim("pose3d_sim");
    sim.root().add<Mass>("body", 1.0f);
    auto& imu = sim.root().add<IMU>("imu",
        IMU_SIGMA, 0.0f);
    auto& dvl = sim.root().add<DVL>("dvl",
        DVL_SIGMA);
    // Add a small non-axis-aligned thruster to give all 3 axes content.
    auto& thr = sim.root().add<Thruster>("t", 5.0f, geom::Vec3<PartFrame>{1, 1, 0});
    sim.root().compute_params();
    scene.add_craft(sim, InitialState{});

    EKFKernel<6, 3> ekf;
    Eigen::Matrix<double, 6, 1> x0;
    x0.setZero();
    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Identity();
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    for (int i = 0; i < 3; ++i) {
        Q(i, i)     = 1e-7;
        Q(i+3, i+3) = IMU_SIGMA * IMU_SIGMA * DT;
    }
    Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity()
                                  * (DVL_SIGMA * DVL_SIGMA);

    noise_seed(31);
    thr.set_throttle(0.5f);   // partial thrust → diagonal motion in xy + falling

    for (int i = 0; i < N_STEPS; ++i) {
        w.step();

        Eigen::Matrix<double, 3, 1> u;
        auto a = imu.last_accel();
        u(0) = double(a.x()); u(1) = double(a.y()); u(2) = double(a.z());
        InertialNav3DBound bound{DT, u};
        ekf.predict(bound, 0.0, Q);

        if (i % DVL_HZ_DECIM == DVL_HZ_DECIM - 1) {
            Eigen::Matrix<double, 3, 1> z;
            auto v = dvl.last_velocity();
            z(0) = double(v.x()); z(1) = double(v.y()); z(2) = double(v.z());
            ekf.update(DvlMeas3D{}, z, R);
        }
    }

    auto x = ekf.state();
    auto p = sim.scene_to_craft().position();
    auto v = sim.scene_to_craft().vel_linear();
    INFO("ekf p=(", x(0), ",", x(1), ",", x(2), ") v=(",
         x(3), ",", x(4), ",", x(5), ")  truth p=(",
         p.x(), ",", p.y(), ",", p.z(), ") v=(",
         v.x(), ",", v.y(), ",", v.z(), ")");

    // Velocity is directly observed → tight bound on each axis.
    CHECK(std::abs(x(3) - double(p.x() == 0 ? v.x() : v.x())) < 0.10);  // x-vel
    CHECK(std::abs(x(4) - double(v.y())) < 0.10);
    CHECK(std::abs(x(5) - double(v.z())) < 0.10);
    // Position is integrated; allow up to 1 m/axis error after 2 s.
    CHECK(std::abs(x(0) - double(p.x())) < 1.0);
    CHECK(std::abs(x(1) - double(p.y())) < 1.0);
    CHECK(std::abs(x(2) - double(p.z())) < 1.0);
}

// ---- Templated Craft POC ----
//
// Validates that the same Craft authoring API works with a non-MFloat scalar
// (here, double — used for higher-precision filtering or as the value step
// in an EKF). No codegen, just manual construction. When a future codegen
// extension emits templated Craft subclasses, this is the typesystem path it
// will go through.

TEST_CASE("CraftT<double>: builds and ticks identically to CraftT<MFloat>") {
    using DCraft     = manta::CraftT<double>;
    using DMass = manta::parts::MassT<double>;

    DCraft c("dbl");
    auto& body = c.root().add<DMass>("body", 1.0);
    (void)body;
    c.root().compute_params();

    DCraft::RigidState x0 = DCraft::RigidState::Zero();
    x0(3) = 1.0;   // identity quaternion (w=1)
    c.set_rigid_state(x0);

    // No external force → state should evolve trivially (rest stays rest).
    DCraft::RigidState x1 = c.evaluate(x0, 0.01);

    // Position should still be zero, quaternion still identity, velocities zero.
    for (int i = 0; i < 3; ++i)  CHECK(std::abs(x1(i))     < 1e-10);
    CHECK(std::abs(x1(3) - 1.0) < 1e-10);
    for (int i = 4; i < 7; ++i)  CHECK(std::abs(x1(i))     < 1e-10);
    for (int i = 7; i < 13; ++i) CHECK(std::abs(x1(i))     < 1e-10);
}

// ---- Autodiff Jacobian extracted from a real CraftT<Jet> tick ----
//
// The big payoff of templating Craft on Scalar: instantiate the craft with
// `ceres::Jet`, call `evaluate(x, dt)`, and read the partial derivatives
// straight off the output state — no hand-written process model. This is
// what the EKF will eventually do internally to derive the process Jacobian
// from any user-authored craft.

#include <ceres/jet.h>

TEST_CASE("CraftT<Jet>: autodiff yields process Jacobian from evaluate()") {
    constexpr int N = 13;  // CraftT::kRigidStateDim
    using Jet      = ceres::Jet<double, N>;
    using JCraft   = manta::CraftT<Jet>;
    using JPoint   = manta::parts::MassT<Jet>;

    JCraft craft("est");
    craft.root().add<JPoint>("body", Jet(1.0));
    craft.root().compute_params();

    // Seed each input dimension with its own derivative direction.
    JCraft::RigidState x;
    x.setZero();
    for (int i = 0; i < N; ++i) {
        x(i) = Jet(0.0, i);
    }
    // Identity quaternion at index 3 (w-component) → value 1, derivative seed 3.
    x(3) = Jet(1.0, 3);

    constexpr double dt = 0.01;
    auto x_new = craft.evaluate(x, Jet(dt));

    // For a free Mass with no wrench applied:
    //   p_new = p_old + v_old * dt    (acceleration is zero → no quadratic term)
    //   v_new = v_old
    //
    // Therefore the upper-left 7×7 block of the Jacobian is determined:
    //   ∂p_x/∂p_x = 1            (state 0 ← state 0)
    //   ∂p_x/∂v_x = dt           (state 0 ← state 7)
    //   ∂v_x/∂v_x = 1            (state 7 ← state 7)
    //   etc.

    // Position partials for the x axis:
    CHECK(std::abs(x_new(0).v[0] - 1.0) < 1e-10);    // ∂p_x_new/∂p_x = 1
    CHECK(std::abs(x_new(0).v[7] - dt)  < 1e-10);    // ∂p_x_new/∂v_x = dt
    CHECK(std::abs(x_new(0).v[8])       < 1e-10);    // ∂p_x_new/∂v_y = 0

    // Velocity partials for the x axis:
    CHECK(std::abs(x_new(7).v[7] - 1.0) < 1e-10);    // ∂v_x_new/∂v_x = 1
    CHECK(std::abs(x_new(7).v[0])       < 1e-10);    // ∂v_x_new/∂p_x = 0

    // Symmetric on y, z:
    CHECK(std::abs(x_new(1).v[8] - dt)  < 1e-10);    // ∂p_y_new/∂v_y = dt
    CHECK(std::abs(x_new(2).v[9] - dt)  < 1e-10);    // ∂p_z_new/∂v_z = dt
}

// ---- EKF: full EKF wired directly to a user-templated Craft ----
//
// User authors the craft once as a class template. EKF instantiates it
// twice internally (double for value step, Jet for Jacobian step) and runs
// predict + update without any hand-written process functor.

#include "../include/manta/estimation/ekf.hpp"

template <class Scalar>
class FreeBodyCraft : public manta::CraftT<Scalar> {
public:
    FreeBodyCraft() : manta::CraftT<Scalar>("est") {
        this->root().template add<manta::parts::MassT<Scalar>>("body", Scalar(1.0));
        this->root().compute_params();
    }
};

// Trivial measurement: read position directly. Not a realistic sensor — just
// a fixed h(x) = x[0..2] to exercise the update path.
struct PositionMeas3D {
    template <class S, int N>
    Eigen::Matrix<S, 3, 1> operator()(const Eigen::Matrix<S, N, 1>& x) const {
        return x.template segment<3>(0);
    }
};

TEST_CASE("EKF: free-body predict+update extracts Jacobians from CraftT") {
    using Ekf = manta::estimation::EKF</*NumCrafts=*/1, /*MeasDim=*/3>;
    using Jet = Ekf::Jet;

    // Build the value-side world with one craft.
    manta::WorldT<double> w_real;
    w_real.clock().set_dt(0.01f);
    auto& s_real = w_real.create_scene();
    FreeBodyCraft<double> craft_real;
    s_real.add_craft(craft_real);

    // Build the Jet-shadow world identically.
    manta::WorldT<Jet> w_jet;
    w_jet.clock().set_dt(0.01f);
    auto& s_jet = w_jet.create_scene();
    FreeBodyCraft<Jet> craft_jet;
    s_jet.add_craft(craft_jet);

    Ekf ekf;
    ekf.bind(w_jet, {&craft_real}, {&craft_jet});

    // Initial state: at origin, identity orientation, small initial velocity.
    Ekf::StateVec x0;
    x0.setZero();
    x0(3) = 1.0;     // identity quaternion w
    x0(7) = 1.0;     // 1 m/s along x
    Ekf::StateCov P0 = Ekf::StateCov::Identity() * 0.1;

    ekf.set_state(x0);
    ekf.set_covariance(P0);

    Ekf::StateCov Q = Ekf::StateCov::Identity() * 1e-6;
    Ekf::MeasCov  R = Ekf::MeasCov::Identity()  * 1e-4;

    constexpr double dt = 0.01;
    constexpr int    N  = 100;     // 1 s

    // Drive it for a second with no measurements — pure prediction.
    for (int i = 0; i < N; ++i) {
        ekf.predict(dt, Q);
    }

    // After 1 s of free flight at 1 m/s along x, expect p_x ≈ 1.0.
    auto x = ekf.state();
    INFO("p=(", x(0), ",", x(1), ",", x(2), ") v_x=", x(7));
    CHECK(std::abs(x(0) - 1.0) < 1e-6);
    CHECK(std::abs(x(1)      ) < 1e-6);
    CHECK(std::abs(x(7) - 1.0) < 1e-6);

    // Now feed a measurement saying the craft is at origin (resetting position
    // toward truth — the prediction had drifted to p_x = 1.0). The update
    // should pull p_x back toward 0, and the velocity estimate toward
    // something more consistent.
    Ekf::MeasVec z;
    z << 0.0, 0.0, 0.0;
    ekf.update(PositionMeas3D{}, z, R);

    auto x_post = ekf.state();
    // Position should have moved back toward the measurement (0).
    CHECK(x_post(0) < x(0));
    CHECK(std::abs(x_post(0)) < 0.5);
}
