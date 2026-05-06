#include <doctest/doctest.h>

#include "../include/manta/estimation/ekf.hpp"
#include <cmath>
#include <random>

using namespace manta::estimation;

// 1-D constant-velocity state: x = [position, velocity].
// Process: position += velocity * dt, velocity unchanged.
struct ConstVelProcess {
    template <class S>
    Eigen::Matrix<S, 2, 1> operator()(const Eigen::Matrix<S, 2, 1>& x,
                                      double dt) const {
        Eigen::Matrix<S, 2, 1> y;
        y(0) = x(0) + x(1) * S(dt);
        y(1) = x(1);
        return y;
    }
};

// Position-only measurement: z = position.
struct PositionMeas {
    template <class S>
    Eigen::Matrix<S, 1, 1> operator()(const Eigen::Matrix<S, 2, 1>& x) const {
        Eigen::Matrix<S, 1, 1> z;
        z(0) = x(0);
        return z;
    }
};

TEST_CASE("EKF: 1-D constant-velocity tracker converges to truth") {
    constexpr double dt        = 0.01;     // 100 Hz
    constexpr int    N_steps   = 1000;     // 10 seconds of sim
    constexpr int    meas_every= 10;       // 10 Hz measurement
    constexpr double v_true    = 1.5;      // m/s
    constexpr double meas_sigma= 0.05;     // 5 cm position noise
    constexpr double proc_sigma= 0.001;    // tiny process noise on velocity

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, meas_sigma);

    EKFKernel<2, 1> ekf;
    // Initialize with a wrong guess: position 0, velocity 0 (truth: 0, 1.5).
    Eigen::Vector2d x0(0.0, 0.0);
    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0,
          0.0, 4.0;   // high initial velocity uncertainty
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    Eigen::Matrix2d Q;
    Q << 0.0, 0.0,
         0.0, proc_sigma * proc_sigma;

    Eigen::Matrix<double, 1, 1> R;
    R(0, 0) = meas_sigma * meas_sigma;

    double truth_pos = 0.0;
    double truth_vel = v_true;

    for (int i = 0; i < N_steps; ++i) {
        truth_pos += truth_vel * dt;
        ekf.predict(ConstVelProcess{}, dt, Q);
        if (i % meas_every == 0) {
            Eigen::Matrix<double, 1, 1> z;
            z(0) = truth_pos + noise(rng);
            ekf.update(PositionMeas{}, z, R);
        }
    }

    auto x = ekf.state();
    INFO("estimate p=", x(0), " v=", x(1), "  truth p=", truth_pos, " v=", truth_vel);
    // Should converge to within ~5 cm of truth in position and ~5 cm/s in velocity.
    CHECK(std::abs(x(0) - truth_pos) < 0.10);
    CHECK(std::abs(x(1) - truth_vel) < 0.10);
}

// Nonlinear measurement: 1-D pendulum-style. State = [theta, theta_dot].
// Process: linear damped oscillator with gravity restoring torque.
//   theta_dot += (-g/L * sin(theta) - b * theta_dot) * dt
//   theta     += theta_dot * dt
// Measurement: angle observation z = sin(theta) (e.g. an encoder mounted on
// a yaw-arm whose horizontal projection is what's read).
struct PendulumProcess {
    template <class S>
    Eigen::Matrix<S, 2, 1> operator()(const Eigen::Matrix<S, 2, 1>& x,
                                      double dt) const {
        const double g_over_L = 9.81 / 0.5;
        const double damp     = 0.2;
        Eigen::Matrix<S, 2, 1> y;
        S theta_dd = -S(g_over_L) * sin(x(0)) - S(damp) * x(1);
        y(1) = x(1) + theta_dd * S(dt);
        y(0) = x(0) + y(1) * S(dt);   // semi-implicit (rate updates first)
        return y;
    }
};

struct SinAngleMeas {
    template <class S>
    Eigen::Matrix<S, 1, 1> operator()(const Eigen::Matrix<S, 2, 1>& x) const {
        Eigen::Matrix<S, 1, 1> z;
        z(0) = sin(x(0));
        return z;
    }
};

TEST_CASE("EKF: nonlinear pendulum estimator tracks true state via Jets") {
    constexpr double dt         = 0.01;
    constexpr int    N_steps    = 2000;     // 20 s
    constexpr int    meas_every = 5;        // 20 Hz measurement
    constexpr double meas_sigma = 0.02;     // sin(theta) noise

    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, meas_sigma);

    EKFKernel<2, 1> ekf;
    Eigen::Vector2d x0(0.1, 0.0);    // small initial guess
    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0,
          0.0, 1.0;
    ekf.set_state(x0);
    ekf.set_covariance(P0);

    Eigen::Matrix2d Q;
    Q << 1e-6, 0.0,
         0.0,  1e-4;
    Eigen::Matrix<double, 1, 1> R;
    R(0, 0) = meas_sigma * meas_sigma;

    // Truth simulation using the same dynamics, started from pi/3 with no
    // initial rate.
    double th  = M_PI / 3.0;
    double thd = 0.0;

    for (int i = 0; i < N_steps; ++i) {
        // Truth advance (semi-implicit, matching the EKF's process model).
        double tdd = -(9.81 / 0.5) * std::sin(th) - 0.2 * thd;
        thd += tdd * dt;
        th  += thd * dt;
        ekf.predict(PendulumProcess{}, dt, Q);
        if (i % meas_every == 0) {
            Eigen::Matrix<double, 1, 1> z;
            z(0) = std::sin(th) + noise(rng);
            ekf.update(SinAngleMeas{}, z, R);
        }
    }

    auto x = ekf.state();
    INFO("estimate th=", x(0), " thd=", x(1),
         "  truth th=", th, " thd=", thd);
    // The pendulum is well-observable through sin(theta), but only
    // up to a sign ambiguity early on. After 20 s of tracking the
    // estimator should have locked on.
    CHECK(std::abs(std::sin(x(0)) - std::sin(th)) < 0.10);
    CHECK(std::abs(x(1) - thd) < 0.5);
}
