#include "manta/core/noise.hpp"

#include <cmath>
#include <random>

namespace manta {
namespace {

struct ThreadRng {
    std::mt19937                     engine{std::random_device{}()};
    std::normal_distribution<float>  dist{0.0f, 1.0f};

    float next() { return dist(engine); }

    void seed(std::uint32_t s) {
        engine.seed(s);
        dist.reset();
    }
};

thread_local ThreadRng tls_rng;

} // anonymous

void noise_seed(std::uint32_t seed) noexcept {
    tls_rng.seed(seed);
}

float noise_rng_next() noexcept {
    return tls_rng.next();
}

} // namespace manta
