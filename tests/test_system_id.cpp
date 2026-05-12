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
#include "../include/manta/estimation/parameter_fit.hpp"
#include "../include/manta/fields/fluid_field.hpp"
#include "../include/manta/parts/actuator/thruster.hpp"
#include "../include/manta/parts/structure/mass.hpp"
#include "../include/manta/parts/aero/surface.hpp"

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
    using DPM    = MassT<Scalar>;
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

    // Fit via the new ParameterFit wrapper. Start far from the answer; the
    // wrapper hides the Ceres Problem / Solver / Summary boilerplate while
    // still letting the user write the templated cost functor.
    manta::estimation::ParameterFit<1> fit;
    fit.set_lower_bound(0, 0.1);   // mass > 0.1 kg

    auto result = fit.solve_dynamic(
        /*initial_guess=*/{5.0},
        new MassFitCost(observed, MAX_THRUST, DT),
        /*n_residuals=*/N_STEPS);

    INFO("converged mass = ", result.params[0], " (true ", TRUE_MASS, ")");
    INFO("ceres summary: ", result.brief_report);
    CHECK(result.success);
    CHECK(result.params[0] == doctest::Approx(TRUE_MASS).epsilon(1e-3));
}


// Second case: fit a linear-drag coefficient on a Surface1 part. A point mass
// with diagonal drag −k in a steady wind +x at 5 m/s relaxes exponentially
// toward the wind speed with time constant τ = m/k. Recovering k from the
// observed velocity trajectory exercises:
//   * The Surface1T<Scalar> instantiation under Jet<double, 1>.
//   * The value-bridge inside Surface's update() that queries the (non-
//     templated) FluidField. This is the integration test for
//     "autodiff-through-non-templated-field-queries" — proves we don't
//     accidentally lose the parameter's derivative when the field's value
//     gets cast to MFloat in the bridge.

namespace {
template <class Scalar>
std::vector<Scalar> simulate_drag_trajectory(Scalar drag_k,
                                             double mass,
                                             double wind_x,
                                             double dt,
                                             int    n_steps) {
    using DCraft = CraftT<Scalar>;
    using DPM    = MassT<Scalar>;
    using DSurf  = Surface1T<Scalar>;
    using TensorT = geom::Mat3<PartFrame, PartFrame, Scalar>;

    DCraft c("drag_fit");
    c.root().template add<DPM>("body", Scalar(mass));

    // Build a diagonal +k drag tensor. Surface's F = A · v_rel where
    // v_rel = v_fluid − v_self, so a positive diagonal makes F point along
    // (v_fluid − v_self), pulling the body toward the wind.
    auto force_tensor  = TensorT::identity();
    force_tensor.raw()(0, 0) = drag_k;
    force_tensor.raw()(1, 1) = drag_k;
    force_tensor.raw()(2, 2) = drag_k;
    auto torque_tensor = TensorT::identity();
    torque_tensor.raw()(0, 0) = Scalar(0);
    torque_tensor.raw()(1, 1) = Scalar(0);
    torque_tensor.raw()(2, 2) = Scalar(0);
    std::array<TensorT, 1> A{force_tensor};
    std::array<TensorT, 1> B{torque_tensor};
    c.root().template add<DSurf>("drag", A, B);

    // Register a uniform incompressible fluid moving in +x as a single
    // PERSISTENT disturbance. Field is value-only; the part bridges through it.
    fields::FluidField wind;
    wind.add(fields::FluidField::Disturbance::uniform_incompressible(
                 MFloat(1.0),
                 geom::Vec3<SceneFrame>{MFloat(wind_x), MFloat(0), MFloat(0)}),
             fields::PERSISTENT);
    c.template register_field<fields::FluidField>(wind);

    c.root().compute_params();

    typename DCraft::RigidState x;
    x.setZero();
    x(3) = Scalar(1);

    std::vector<Scalar> vx;
    vx.reserve(n_steps);
    for (int k = 0; k < n_steps; ++k) {
        x = c.evaluate(x, Scalar(dt));
        vx.push_back(x(7));   // v_x of rigid state
    }
    return vx;
}

struct DragFitCost {
    DragFitCost(const std::vector<double>& observed_vx,
                double mass, double wind_x, double dt)
        : observed_(observed_vx), mass_(mass), wind_x_(wind_x), dt_(dt) {}

    template <class T>
    bool operator()(const T* k, T* residual) const {
        auto pred = simulate_drag_trajectory(k[0], mass_, wind_x_, dt_,
                                             static_cast<int>(observed_.size()));
        for (size_t i = 0; i < observed_.size(); ++i) {
            residual[i] = pred[i] - T(observed_[i]);
        }
        return true;
    }

    const std::vector<double>& observed_;
    double mass_, wind_x_, dt_;
};
}  // namespace

TEST_CASE("System ID: fit drag coefficient on a Surface1 in flowing wind") {
    constexpr double TRUE_K = 2.0;
    constexpr double MASS   = 1.0;
    constexpr double WIND   = 5.0;
    constexpr double DT     = 0.01;
    constexpr int    NS     = 100;

    auto observed = simulate_drag_trajectory<double>(TRUE_K, MASS, WIND, DT, NS);
    // Sanity: τ = m/k = 0.5 s. After 1 s, v_x ≈ WIND·(1 − e^{-2}) ≈ 4.32.
    CHECK(observed.back() == doctest::Approx(4.32).epsilon(0.05));

    manta::estimation::ParameterFit<1> fit;
    fit.set_lower_bound(0, 0.01);

    auto result = fit.solve_dynamic(
        /*initial_guess=*/{0.5},
        new DragFitCost(observed, MASS, WIND, DT),
        /*n_residuals=*/NS);

    INFO("converged k = ", result.params[0], " (true ", TRUE_K, ")");
    INFO("ceres summary: ", result.brief_report);
    CHECK(result.success);
    CHECK(result.params[0] == doctest::Approx(TRUE_K).epsilon(1e-3));
}
