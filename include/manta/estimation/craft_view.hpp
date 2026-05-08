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
//   CraftView<decltype(ekf), /*SliceIdx=*/0> drone0_view{ekf};
//   CraftView<decltype(ekf), /*SliceIdx=*/1> drone1_view{ekf};
//
//   // Initial state setup — no manual segment<>() math.
//   drone0_view.set_position({0.0, 0.0, 5.0});
//   drone0_view.set_position_var(1e-4);
//
//   // Reads — same shape as the legacy `EKF::position(idx)` etc.
//   auto p = drone0_view.position();
//   auto p_stddev = drone0_view.position_stddev();
//
// The view holds a non-const reference to the filter; const-correctness
// in user code is enforced by passing a `const auto&` reference to a
// CraftView constructed over a const filter (the setters won't compile
// because `set_state` / `set_covariance` aren't const).

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "../core/types.hpp"
#include "../geom/ori.hpp"
#include "../geom/vec3.hpp"

namespace manta::estimation {

template <class FilterT, int SliceIdx>
class CraftView {
public:
    using Spec = typename FilterT::StateSpec;
    static constexpr int kAmbOff = Spec::template ambient_offset<SliceIdx>;
    static constexpr int kTanOff = Spec::template tangent_offset<SliceIdx>;

    explicit CraftView(FilterT& filter) noexcept : filter_(filter) {}

    // ---- Reference-state reads ----
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

    // ---- Tangent-covariance stddev reads ----
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

    // ---- State setters ----
    //
    // Each setter copies the current state vector, mutates the relevant
    // segment, and pushes via `filter_.set_state(...)`. For an initial
    // setup that touches several fields in a row, prefer the bundled
    // `set_state(p, q, v, w)` overload below to avoid four copies.

    // Set the craft's scene-frame position.
    void set_position(const Eigen::Matrix<double, 3, 1>& p) {
        auto x = filter_.state();
        x.template segment<3>(kAmbOff + 0) = p;
        filter_.set_state(x);
    }
    void set_position(double x, double y, double z) {
        set_position(Eigen::Vector3d{x, y, z});
    }
    template <class Frame, class Scalar>
    void set_position(const geom::Vec3<Frame, Scalar>& p) {
        set_position(Eigen::Vector3d{
            double(p.x()), double(p.y()), double(p.z())});
    }

    // Set orientation as a (w, x, y, z) quaternion.
    void set_orientation(const Eigen::Matrix<double, 4, 1>& q) {
        // Normalize defensively — the EKF assumes a unit quat.
        Eigen::Quaterniond qn(q(0), q(1), q(2), q(3));
        qn.normalize();
        auto x = filter_.state();
        x(kAmbOff + 3) = qn.w();
        x(kAmbOff + 4) = qn.x();
        x(kAmbOff + 5) = qn.y();
        x(kAmbOff + 6) = qn.z();
        filter_.set_state(x);
    }
    template <class Frame, class Scalar>
    void set_orientation(const geom::Ori<Frame, Scalar>& o) {
        const auto& q = o.raw();
        set_orientation(Eigen::Vector4d{
            double(q.w()), double(q.x()), double(q.y()), double(q.z())});
    }
    void set_orientation_identity() {
        set_orientation(Eigen::Vector4d{1.0, 0.0, 0.0, 0.0});
    }

    // Set the craft's scene-frame linear velocity.
    void set_vel_linear(const Eigen::Matrix<double, 3, 1>& v) {
        auto x = filter_.state();
        x.template segment<3>(kAmbOff + 7) = v;
        filter_.set_state(x);
    }
    void set_vel_linear(double x, double y, double z) {
        set_vel_linear(Eigen::Vector3d{x, y, z});
    }
    template <class Frame, class Scalar>
    void set_vel_linear(const geom::Vec3<Frame, Scalar>& v) {
        set_vel_linear(Eigen::Vector3d{
            double(v.x()), double(v.y()), double(v.z())});
    }

    // Set the craft's body-frame angular velocity.
    void set_vel_angular(const Eigen::Matrix<double, 3, 1>& w) {
        auto x = filter_.state();
        x.template segment<3>(kAmbOff + 10) = w;
        filter_.set_state(x);
    }
    void set_vel_angular(double x, double y, double z) {
        set_vel_angular(Eigen::Vector3d{x, y, z});
    }
    template <class Frame, class Scalar>
    void set_vel_angular(const geom::Vec3<Frame, Scalar>& w) {
        set_vel_angular(Eigen::Vector3d{
            double(w.x()), double(w.y()), double(w.z())});
    }

    // Bundled setter — mutates all four rigid attributes in one
    // set_state() call. Use this for initial setup to avoid four
    // separate state-copy round-trips.
    void set_state(const Eigen::Matrix<double, 3, 1>& p,
                   const Eigen::Matrix<double, 4, 1>& q,
                   const Eigen::Matrix<double, 3, 1>& v,
                   const Eigen::Matrix<double, 3, 1>& w) {
        Eigen::Quaterniond qn(q(0), q(1), q(2), q(3));
        qn.normalize();
        auto x = filter_.state();
        x.template segment<3>(kAmbOff + 0) = p;
        x(kAmbOff + 3) = qn.w();
        x(kAmbOff + 4) = qn.x();
        x(kAmbOff + 5) = qn.y();
        x(kAmbOff + 6) = qn.z();
        x.template segment<3>(kAmbOff + 7)  = v;
        x.template segment<3>(kAmbOff + 10) = w;
        filter_.set_state(x);
    }

    // Reset to a "rest at origin" pose (zero everything, identity quat).
    void reset_to_rest() {
        set_state(Eigen::Vector3d::Zero(),
                  Eigen::Vector4d{1.0, 0.0, 0.0, 0.0},
                  Eigen::Vector3d::Zero(),
                  Eigen::Vector3d::Zero());
    }

    // ---- Covariance setters ----
    //
    // Each `set_*_var(v)` writes the 3x3 block at the relevant tangent
    // offset to `v · I_3`. Off-diagonal entries within the block are
    // zeroed; cross-craft / cross-block entries are left alone.

    void set_position_var(double var) {
        set_block_var<3>(kTanOff + 0, var);
    }
    void set_position_stddev(double sd) { set_position_var(sd * sd); }

    void set_attitude_var(double var) {
        set_block_var<3>(kTanOff + 3, var);
    }
    void set_attitude_stddev(double sd) { set_attitude_var(sd * sd); }

    void set_vel_linear_var(double var) {
        set_block_var<3>(kTanOff + 6, var);
    }
    void set_vel_linear_stddev(double sd) { set_vel_linear_var(sd * sd); }

    void set_vel_angular_var(double var) {
        set_block_var<3>(kTanOff + 9, var);
    }
    void set_vel_angular_stddev(double sd) { set_vel_angular_var(sd * sd); }

    // Bundled covariance setter — one set_covariance() call.
    void set_state_covariance(double pos_var,
                              double attitude_var,
                              double vel_var,
                              double angvel_var) {
        auto P = filter_.covariance();
        zero_block<3>(P, kTanOff + 0);
        zero_block<3>(P, kTanOff + 3);
        zero_block<3>(P, kTanOff + 6);
        zero_block<3>(P, kTanOff + 9);
        for (int i = 0; i < 3; ++i) {
            P(kTanOff + 0 + i, kTanOff + 0 + i) = pos_var;
            P(kTanOff + 3 + i, kTanOff + 3 + i) = attitude_var;
            P(kTanOff + 6 + i, kTanOff + 6 + i) = vel_var;
            P(kTanOff + 9 + i, kTanOff + 9 + i) = angvel_var;
        }
        filter_.set_covariance(P);
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

    template <int N>
    void set_block_var(int start, double var) {
        auto P = filter_.covariance();
        zero_block<N>(P, start);
        for (int i = 0; i < N; ++i) P(start + i, start + i) = var;
        filter_.set_covariance(P);
    }

    template <int N, class Mat>
    static void zero_block(Mat& P, int start) {
        // Zero rows + cols intersecting [start, start+N) — both off-
        // diagonal cross-couplings of the block AND its diagonal.
        const int total = static_cast<int>(P.rows());
        for (int i = 0; i < N; ++i) {
            P.row(start + i).setZero();
            P.col(start + i).setZero();
            (void)total;
        }
    }

    FilterT& filter_;
};

} // namespace manta::estimation
