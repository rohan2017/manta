#pragma once

#include "../core/types.hpp"

namespace manta::sim {

// RateGate — converts a fast sim tick into a slower fixed-rate fire.
//
//   gate.tick(dt) returns true on:
//     * the very first call (so consumers see a "fresh" reading on tick 0).
//     * any subsequent call where the accumulated `dt` since the previous
//       fire has reached the gate's period (= 1 / rate_hz).
//
// Setting rate_hz <= 0 disables the gate — every call returns true.
//
// The gate accumulates real elapsed sim time, not tick count, so a sim
// running with variable dt still hits the right rate. Period subtraction
// (rather than reset-to-zero) preserves average rate when `dt` doesn't
// divide evenly into the period.
class RateGate {
public:
    constexpr RateGate() noexcept = default;
    explicit constexpr RateGate(MFloat rate_hz) noexcept
        : period_(rate_hz > MFloat(0) ? MFloat(1) / rate_hz : MFloat(0)) {}

    bool tick(MFloat dt) noexcept {
        if (period_ <= MFloat(0)) return true;
        if (!started_) {
            started_ = true;
            accum_   = MFloat(0);
            return true;
        }
        accum_ += dt;
        if (accum_ + MFloat(1e-9) >= period_) {
            accum_ -= period_;
            return true;
        }
        return false;
    }

    // Reset to the "first call fires" state. Useful for unit tests.
    void reset() noexcept { started_ = false; accum_ = MFloat(0); }

    MFloat period() const noexcept { return period_; }

private:
    MFloat period_  = MFloat(0);   // 0 = ungated
    MFloat accum_   = MFloat(0);
    bool started_ = false;
};

} // namespace manta::sim
