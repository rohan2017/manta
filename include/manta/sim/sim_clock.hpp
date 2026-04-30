#pragma once

namespace manta {

// Owns simulation time and the time-step. In single-process runs it is
// advanced by World::update(). In distributed runs a Zenoh-backed
// implementation will replace or wrap this.
class SimClock {
public:
    explicit SimClock(float dt = 0.02f) noexcept : dt_(dt) {}

    float time() const noexcept { return time_; }
    float dt()   const noexcept { return dt_; }

    void set_dt(float dt) noexcept { dt_ = dt; }
    void advance()        noexcept { time_ += dt_; }
    void reset()          noexcept { time_ = 0.0f; }

private:
    float time_ = 0.0f;
    float dt_;
};

} // namespace manta
