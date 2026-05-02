#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>

#include <ceres/ceres.h>

namespace manta::estimation {

// Thin wrapper around a Ceres autodiff parameter-fit problem.
//
// The user still writes the Scalar-templated cost functor — that's where
// the simulation lives, and the user knows their dynamics best. This API
// hides the surrounding boilerplate: building the Problem, wrapping the
// functor in `AutoDiffCostFunction`, applying parameter bounds, choosing
// solver options, and unpacking the summary.
//
// Pattern:
//
//     struct MyMassFit {
//         std::vector<double> observed_x;
//         double max_thrust, dt;
//         template <class T>
//         bool operator()(const T* mass, T* residual) const {
//             auto pred = simulate<T>(mass[0], T(max_thrust), dt, observed_x.size());
//             for (size_t k = 0; k < observed_x.size(); ++k)
//                 residual[k] = pred[k] - T(observed_x[k]);
//             return true;
//         }
//     };
//
//     manta::estimation::ParameterFit<1> fit;
//     fit.set_lower_bound(0, 0.1);                // mass > 0.1 kg
//     auto result = fit.solve_dynamic(
//         /*initial_guess=*/{5.0},
//         new MyMassFit{observed, max_thrust, dt},
//         /*n_residuals=*/observed.size());
//     CHECK(result.success);
//     std::cout << result.brief_report << "\n";
//     double mass_fit = result.params[0];
//
// The CostFunctor pointer is heap-allocated by the caller and owned by
// the AutoDiffCostFunction (Ceres' default ownership) — `solve` does not
// delete it again on this side.
template <int NParams>
class ParameterFit {
public:
    static_assert(NParams > 0, "ParameterFit<NParams>: need at least one parameter");

    struct Result {
        std::array<double, NParams> params{};
        bool                        success = false;
        std::string                 brief_report;
    };

    // Constrain individual parameters. Use only when needed; unconstrained
    // problems converge faster and tend to find better optima.
    void set_lower_bound(int i, double v) noexcept { lower_[i] = v; }
    void set_upper_bound(int i, double v) noexcept { upper_[i] = v; }

    // Static-count residuals (compile-time NResiduals). Preferred when the
    // count is known ahead of time — Ceres dispatches autodiff Jets more
    // efficiently and the cost function is plain stack-typed.
    template <int NResiduals, class CostFunctor>
    Result solve(const std::array<double, NParams>& initial_guess,
                 CostFunctor*                       functor,           // Ceres takes ownership
                 ceres::Solver::Options             opts = default_options()) {
        Result r;
        r.params = initial_guess;
        ceres::Problem problem;
        auto* cf = new ceres::AutoDiffCostFunction<CostFunctor, NResiduals, NParams>(functor);
        problem.AddResidualBlock(cf, /*loss=*/nullptr, r.params.data());
        apply_bounds(problem, r.params.data());
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        r.success      = summary.IsSolutionUsable();
        r.brief_report = summary.BriefReport();
        return r;
    }

    // Dynamic residual count — uses `ceres::DYNAMIC` and a runtime size.
    // Use this when the trajectory length isn't a compile-time constant.
    template <class CostFunctor>
    Result solve_dynamic(const std::array<double, NParams>& initial_guess,
                         CostFunctor*                       functor,           // Ceres takes ownership
                         int                                n_residuals,
                         ceres::Solver::Options             opts = default_options()) {
        Result r;
        r.params = initial_guess;
        ceres::Problem problem;
        auto* cf = new ceres::AutoDiffCostFunction<CostFunctor, ceres::DYNAMIC, NParams>(
            functor, n_residuals);
        problem.AddResidualBlock(cf, /*loss=*/nullptr, r.params.data());
        apply_bounds(problem, r.params.data());
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        r.success      = summary.IsSolutionUsable();
        r.brief_report = summary.BriefReport();
        return r;
    }

    // Default solver options — DENSE_QR with 50 max iterations. Plenty for
    // small parameter counts; tune via the `opts` argument when scaling up.
    static ceres::Solver::Options default_options() {
        ceres::Solver::Options o;
        o.linear_solver_type           = ceres::DENSE_QR;
        o.minimizer_progress_to_stdout = false;
        o.max_num_iterations           = 50;
        return o;
    }

private:
    void apply_bounds(ceres::Problem& problem, double* params) {
        for (int i = 0; i < NParams; ++i) {
            if (lower_[i]) problem.SetParameterLowerBound(params, i, *lower_[i]);
            if (upper_[i]) problem.SetParameterUpperBound(params, i, *upper_[i]);
        }
    }

    std::array<std::optional<double>, NParams> lower_{};
    std::array<std::optional<double>, NParams> upper_{};
};

} // namespace manta::estimation
