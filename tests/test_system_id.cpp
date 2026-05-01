// Demonstrates parameter identification using CraftT<Jet> as the forward
// model and Ceres::Solver to recover a hidden parameter from observed
// trajectory data. This is the proof-of-concept that the existing autodiff
// machinery (the same Jets the EKF uses for Jacobians) can be reused for
// system identification — fit drag coefficients, mass, motor constants,
// thruster max-thrust, etc. — without any extra infrastructure.
//
// The example here recovers the mass of a thruster-driven point body from
// a recorded position trajectory. Closed-form: given constant force F,
// p(t) = (F / 2m) · t². Ceres reads the recorded p_k, builds residuals
// against the CraftT<Jet>-predicted trajectory, and converges on m.

#include <cmath>
#include <cstdio>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/parts/actuator/thruster.hpp"
#include "../include/manta/parts/structure/point_mass.hpp"

using namespace manta;
using namespace manta::parts;

namespace {

// Run the point-mass-plus-thruster craft forward for `n_steps` of `dt`
// seconds with the thruster commanded at full throttle, returning a series
// of x-position samples. Templated on Scalar so the same code path produces
// the truth trajectory (Scalar=double) and the Ceres-autodiff prediction
// (Scalar=ceres::Jet).
template <class Scalar>
std::vector<Scalar> simulate_trajectory(Scalar mass,
                                        Scalar max_thrust,
                                        double dt,
                                        int    n_steps) {
    using DCraft = CraftT<Scalar>;
    using DPM    = PointMassT<Scalar>;
    using DThr   = ThrusterT<Scalar>;

    DCraft c("sysid");
    c.root().template add<DPM>("body", mass);
    auto& thr = c.root().template add<DThr>(
        "thrust", max_thrust,
        geom::Vec3<PartFrame, Scalar>{Scalar(1), Scalar(0), Scalar(0)});
    c.root().compute_params();
    thr.set_throttle(Scalar(1.0));

    typename DCraft::RigidState x;
    x.setZero();
    x(3) = Scalar(1);   // identity quaternion w

    std::vector<Scalar> px;
    px.reserve(n_steps);
    for (int k = 0; k < n_steps; ++k) {
        x = c.evaluate(x, Scalar(dt));
        px.push_back(x(0));
    }
    return px;
}

// Cost functor: input is a single parameter (mass), output is a residual
// per observed sample. Ceres autodiffs through simulate_trajectory<Jet>.
struct MassFitCost {
    MassFitCost(const std::vector<double>& observed_px,
                double max_thrust, double dt)
        : observed_(observed_px), max_thrust_(max_thrust), dt_(dt) {}

    template <class T>
    bool operator()(const T* mass, T* residual) const {
        auto pred = simulate_trajectory(mass[0], T(max_thrust_), dt_,
                                        static_cast<int>(observed_.size()));
        for (size_t k = 0; k < observed_.size(); ++k) {
            residual[k] = pred[k] - T(observed_[k]);
        }
        return true;
    }

    const std::vector<double>& observed_;
    double                     max_thrust_;
    double                     dt_;
};

}  // namespace

TEST_CASE("System ID: fit mass from a recorded thruster-driven trajectory") {
    constexpr double TRUE_MASS  = 2.0;
    constexpr double MAX_THRUST = 10.0;
    constexpr double DT         = 0.01;
    constexpr int    N_STEPS    = 100;       // 1 s

    // Generate truth trajectory.
    auto observed = simulate_trajectory<double>(TRUE_MASS, MAX_THRUST, DT, N_STEPS);

    // Sanity: closed-form check. p(1s) = 0.5 * (F/m) * t² = 0.5 * 5 * 1 = 2.5 m.
    CHECK(observed.back() == doctest::Approx(2.5).epsilon(5e-2));

    // Fit. Start far from the answer.
    double mass = 5.0;

    ceres::Problem problem;
    auto* cost = new ceres::AutoDiffCostFunction<MassFitCost,
                                                 ceres::DYNAMIC,
                                                 1>(
        new MassFitCost(observed, MAX_THRUST, DT), N_STEPS);
    problem.AddResidualBlock(cost, /*loss=*/nullptr, &mass);
    problem.SetParameterLowerBound(&mass, 0, 0.1);   // mass is positive

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    INFO("converged mass = ", mass, " (true ", TRUE_MASS, ")");
    INFO("ceres summary: ", summary.BriefReport());
    CHECK(summary.IsSolutionUsable());
    CHECK(mass == doctest::Approx(TRUE_MASS).epsilon(1e-3));
}
