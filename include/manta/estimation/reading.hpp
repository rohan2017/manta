#pragma once

// Reading<Dim> — type-erased z (measurement) source.
// ===================================================
//
// A `Reading<Dim>` provides the EKF with:
//   * the latest measurement value `z` (Vec of size Dim, in double),
//   * a freshness bit indicating whether `z` is new since the last pull.
//
// Where `z` actually comes from is the user's choice. Built-in factories:
//
//   * `reading_from(measurement)` — pulls from another part's
//     Measurement (sim → est wiring). Reads the value-side h(x) cache
//     and the part's freshness flag.
//
//   * `reading_from_buffer<Dim>(z_ptr, fresh_ptr)` — pulls from a
//     user-owned Vec buffer + atomic-flag pair. Used by Zenoh-fed
//     subscribers and other external sources.
//
//   * `reading_from_lambda<Dim>(callable)` — escape hatch. The callable
//     returns `Reading<Dim>::Sample{z, fresh}` directly.

#include <Eigen/Core>
#include <atomic>
#include <functional>
#include <utility>

#include "measurement.hpp"

namespace manta {

template <int Dim>
class Reading {
public:
    using Vec    = Eigen::Matrix<double, Dim, 1>;
    struct Sample { Vec z; bool fresh; };

    // Type-erased construction.
    template <class F>
    /*implicit*/ Reading(F f) : pull_(std::move(f)) {}

    Sample pull() const { return pull_(); }

    int dim() const noexcept { return Dim; }

private:
    std::function<Sample()> pull_;
};

// ---------------------------------------------------------------------
// Factory: pull from another part's Measurement. The source Measurement
// must be on a value-typed (double) part (sim-craft IMU, etc.) — the
// freshness comes from the source part's rate gate.
// ---------------------------------------------------------------------
template <int Dim>
inline Reading<Dim> reading_from(const Measurement& src) {
    return Reading<Dim>{[&src]() -> typename Reading<Dim>::Sample {
        typename Reading<Dim>::Vec z;
        src.read_value(z.data());
        return {z, src.fresh()};
    }};
}

// ---------------------------------------------------------------------
// Factory: pull from a user-owned buffer + freshness flag. The flag is
// an atomic<bool> so producer and consumer can be on different threads
// (Zenoh subscribers typically run on a separate thread). The flag is
// cleared by `pull()` once the value has been consumed.
// ---------------------------------------------------------------------
template <int Dim>
inline Reading<Dim> reading_from_buffer(
        const Eigen::Matrix<double, Dim, 1>* z_ptr,
        std::atomic<bool>* fresh_ptr) {
    return Reading<Dim>{[z_ptr, fresh_ptr]() -> typename Reading<Dim>::Sample {
        bool was_fresh = fresh_ptr->exchange(false);
        return {*z_ptr, was_fresh};
    }};
}

// ---------------------------------------------------------------------
// Factory: arbitrary callable. The callable must return a Sample.
// ---------------------------------------------------------------------
template <int Dim, class F>
inline Reading<Dim> reading_from_lambda(F&& f) {
    return Reading<Dim>{std::forward<F>(f)};
}

} // namespace manta
