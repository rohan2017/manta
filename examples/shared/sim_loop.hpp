#pragma once

#include <chrono>
#include <thread>

namespace manta::examples {

// Real-time sim loop pacer. Call wait_for_next_tick() at the bottom of each
// loop iteration; it sleeps so iterations occur at `period_seconds` intervals.
// Drift accumulates to next tick (no catch-up bursts).
class RealTimePacer {
public:
    using Clock = std::chrono::steady_clock;

    explicit RealTimePacer(double period_seconds)
        : period_(std::chrono::duration_cast<Clock::duration>(
                      std::chrono::duration<double>(period_seconds)))
        , next_(Clock::now() + period_) {}

    void wait_for_next_tick() {
        auto now = Clock::now();
        if (now < next_) std::this_thread::sleep_until(next_);
        next_ += period_;
        // If we fell badly behind, resync rather than catching up.
        if (Clock::now() > next_ + period_ * 5) next_ = Clock::now() + period_;
    }

private:
    Clock::duration   period_;
    Clock::time_point next_;
};

} // namespace manta::examples
