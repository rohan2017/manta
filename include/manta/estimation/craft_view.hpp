#pragma once

// CraftView<Filter, SliceIdx> — per-craft accessor adapter.
// =========================================================
//
// A `CraftView` is a thin wrapper around a generic EKF/UKF that exposes
// the rigid-body position / orientation / velocity / angular-velocity
// state for ONE tracked craft, plus the corresponding tangent stddevs.
// It is the user's primary interaction point with per-craft state in
// the StateSpec-based API.
//
// Construction: pass a Filter and the index of the RigidBody slice
// inside its StateSpec. The compile-time slot offsets pop straight out
// of the StateSpec.
//
//   auto state = make_state().track(drone0).track(drone1).build();
//   EKFGeneric<decltype(state), MeasDim, NoiseSlots> ekf{state};
//   CraftView drone0_view{ekf, /*SliceIdx=*/0};
//   CraftView drone1_view{ekf, /*SliceIdx=*/1};
//
//   auto p = drone0_view.position();
//   auto p_stddev = drone0_view.position_stddev();
//
// The accessors mirror the legacy `EKF::position(idx)` etc., so user
// code that takes references to the legacy filter ports straight over
// once it's reading from the CraftView instead.

#include <Eigen/Core>
#include <cmath>

#include "../core/types.hpp"

namespace manta::estimation {

template <class FilterT, int SliceIdx>
class CraftView {
public:
    using Spec = typename FilterT::StateSpec;
    static constexpr int kAmbOff = Spec::template ambient_offset<SliceIdx>;
    static constexpr int kTanOff = Spec::template tangent_offset<SliceIdx>;

    explicit CraftView(const FilterT& filter) noexcept
        : filter_(filter) {}

    // ---- Reference-state slices ----
    Eigen::Matrix<double, 3, 1> position() const noexcept {
        return filter_.state().template segment<3>(kAmbOff + 0);
    }
    // (w, x, y, z) order, matching the rest of the library.
    Eigen::Matrix<double, 4, 1> orientation() const noexcept {
        return filter_.state().template segment<4>(kAmbOff + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear() const noexcept {
        return filter_.state().template segment<3>(kAmbOff + 7);
    }
    Eigen::Matrix<double, 3, 1> vel_angular() const noexcept {
        return filter_.state().template segment<3>(kAmbOff + 10);
    }

    // ---- Tangent-covariance stddevs ----
    Eigen::Matrix<double, 3, 1> position_stddev() const noexcept {
        return diag_stddev<3>(kTanOff + 0);
    }
    // 3-DOF axis-angle tangent stddev (NOT the 4-DOF quaternion).
    Eigen::Matrix<double, 3, 1> orientation_stddev() const noexcept {
        return diag_stddev<3>(kTanOff + 3);
    }
    Eigen::Matrix<double, 3, 1> vel_linear_stddev() const noexcept {
        return diag_stddev<3>(kTanOff + 6);
    }
    Eigen::Matrix<double, 3, 1> vel_angular_stddev() const noexcept {
        return diag_stddev<3>(kTanOff + 9);
    }

private:
    template <int N>
    Eigen::Matrix<double, N, 1> diag_stddev(int start) const noexcept {
        const auto& P = filter_.covariance();
        Eigen::Matrix<double, N, 1> out;
        for (int i = 0; i < N; ++i) {
            const double v = P(start + i, start + i);
            out(i) = std::sqrt(v > 0.0 ? v : 0.0);
        }
        return out;
    }

    const FilterT& filter_;
};

} // namespace manta::estimation
