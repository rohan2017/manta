#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <type_traits>
#include "../geom/vec3.hpp"
#include "types.hpp"

namespace manta {

// --- Noise policies (value types — carry only the scalar parameters) ---

struct WhiteGaussian {
    float sigma = 0.0f;
    explicit WhiteGaussian(float s = 0.0f) : sigma(s) {}
};

struct RandomWalk {
    float sigma = 0.0f;   // per-second diffusion coefficient
    explicit RandomWalk(float s = 0.0f) : sigma(s) {}
};

// --- RNG seed helper ---
// Sets the thread-local RNG seed. Call at test/scenario start for determinism.
void noise_seed(std::uint32_t seed) noexcept;

// Internal: draw one N(0,1) sample from the thread-local RNG.
float noise_rng_next() noexcept;

// --- Noise<WhiteGaussian> ---

template<typename Policy>
class Noise;

template<>
class Noise<WhiteGaussian> {
public:
    Noise() noexcept = default;
    explicit Noise(float sigma) noexcept : policy_(sigma) {}
    explicit Noise(WhiteGaussian p) noexcept : policy_(p) {}

    float sigma() const noexcept { return policy_.sigma; }

    // Draw 3 IID samples from N(0, sigma^2).
    Eigen::Matrix<float, 3, 1> sample3() const noexcept {
        float s = policy_.sigma;
        return {noise_rng_next() * s,
                noise_rng_next() * s,
                noise_rng_next() * s};
    }

    float sample1() const noexcept {
        return noise_rng_next() * policy_.sigma;
    }

private:
    WhiteGaussian policy_;
};

// --- Noise<RandomWalk> ---

template<>
class Noise<RandomWalk> {
public:
    Noise() noexcept = default;
    explicit Noise(float sigma) noexcept : policy_(sigma) {}
    explicit Noise(RandomWalk p) noexcept : policy_(p) {}

    float sigma() const noexcept { return policy_.sigma; }

    // Current 3-vector bias state.
    const Eigen::Matrix<float, 3, 1>& state3() const noexcept { return state3_; }
    float state1() const noexcept { return state1_; }

    // Advance the random walk by dt seconds: state += N(0, sigma^2 * dt).
    void advance(float dt) noexcept {
        float s = policy_.sigma * std::sqrt(dt);
        state3_ += s * Eigen::Matrix<float, 3, 1>{
            noise_rng_next(), noise_rng_next(), noise_rng_next()};
        state1_ += s * noise_rng_next();
    }

private:
    RandomWalk policy_;
    Eigen::Matrix<float, 3, 1> state3_ = Eigen::Matrix<float, 3, 1>::Zero();
    float state1_ = 0.0f;
};

// --- operator+(Vec3<F, Scalar>, Noise) and operator+(Scalar, Noise) ---
//
// Behavior depends on Scalar:
//   - Floating-point Scalar (float, double): sample noise and add. Sim path.
//   - Non-floating-point Scalar (e.g. ceres::Jet<...>): return unchanged.
//     Estimation path — noise is registered into covariance separately, not
//     baked into the value, because that would break Jet's autodiff
//     semantics.
//
// Dispatch is `if constexpr (std::is_floating_point_v<Scalar>)`. ceres::Jet
// has no `is_floating_point` specialization, so it naturally takes the
// else branch.

template<typename F, typename Scalar>
geom::Vec3<F, Scalar> operator+(const geom::Vec3<F, Scalar>& v,
                                const Noise<WhiteGaussian>& n) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return geom::Vec3<F, Scalar>::from_raw(
            v.raw() + n.sample3().template cast<Scalar>());
    } else {
        (void)n;
        return v;
    }
}

template<typename F, typename Scalar>
geom::Vec3<F, Scalar> operator+(const geom::Vec3<F, Scalar>& v,
                                const Noise<RandomWalk>& n) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return geom::Vec3<F, Scalar>::from_raw(
            v.raw() + n.state3().template cast<Scalar>());
    } else {
        (void)n;
        return v;
    }
}

// Scalar overloads — same dispatch.
template<typename Scalar>
Scalar operator+(Scalar v, const Noise<WhiteGaussian>& n) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return v + Scalar(n.sample1());
    } else {
        (void)n;
        return v;
    }
}

template<typename Scalar>
Scalar operator+(Scalar v, const Noise<RandomWalk>& n) noexcept {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return v + Scalar(n.state1());
    } else {
        (void)n;
        return v;
    }
}

} // namespace manta
