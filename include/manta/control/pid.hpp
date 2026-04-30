#pragma once

#include <algorithm>

namespace manta::control {

template<typename T = float>
class PID {
public:
    PID(T kp, T ki, T kd, T ilim = T(1e9f))
        : kp_(kp), ki_(ki), kd_(kd), ilim_(ilim) {}

    T update(T error, float dt) {
        integral_ += error * T(dt);
        integral_ = std::clamp(integral_, -ilim_, ilim_);
        T deriv = (error - prev_error_) / T(dt);
        prev_error_ = error;
        return kp_ * error + ki_ * integral_ + kd_ * deriv;
    }

    void reset() {
        integral_   = T(0);
        prev_error_ = T(0);
    }

    T kp() const noexcept { return kp_; }
    T ki() const noexcept { return ki_; }
    T kd() const noexcept { return kd_; }

private:
    T kp_, ki_, kd_, ilim_;
    T integral_   = T(0);
    T prev_error_ = T(0);
};

} // namespace manta::control
