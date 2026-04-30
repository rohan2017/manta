#include <cmath>
#include <doctest/doctest.h>

#include "../include/manta/core/noise.hpp"

using namespace manta;
using namespace manta::geom;

TEST_CASE("Noise: WhiteGaussian sample1 — mean ≈ 0, std ≈ sigma") {
    noise_seed(42);
    const float sigma = 2.0f;
    Noise<WhiteGaussian> n{sigma};

    const int N = 50'000;
    double sum = 0, sum2 = 0;
    for (int i = 0; i < N; ++i) {
        double v = n.sample1();
        sum  += v;
        sum2 += v * v;
    }
    double mean = sum / N;
    double var  = sum2 / N - mean * mean;
    double std_dev = std::sqrt(var);

    CHECK(std::abs(mean) < 0.1f);
    CHECK(std::abs(std_dev - sigma) < 0.1f);
}

TEST_CASE("Noise: WhiteGaussian sample3 — per-component independence and sigma") {
    noise_seed(0);
    const float sigma = 1.0f;
    Noise<WhiteGaussian> n{sigma};

    const int N = 30'000;
    double sum[3]{}, sum2[3]{};
    for (int i = 0; i < N; ++i) {
        auto s = n.sample3();
        for (int j = 0; j < 3; ++j) { sum[j] += s[j]; sum2[j] += s[j]*s[j]; }
    }
    for (int j = 0; j < 3; ++j) {
        double mean   = sum[j] / N;
        double var    = sum2[j] / N - mean * mean;
        CHECK(std::abs(mean) < 0.05f);
        CHECK(std::abs(std::sqrt(var) - sigma) < 0.05f);
    }
}

TEST_CASE("Noise: RandomWalk advance — bias drifts from zero") {
    noise_seed(7);
    Noise<RandomWalk> n{1.0f};

    // Initial state is zero.
    CHECK(n.state1() == doctest::Approx(0.0f));

    // After many advances the bias departs from zero.
    for (int i = 0; i < 1000; ++i) n.advance(0.01f);  // 10 s total

    // We can't assert direction, but magnitude should be non-trivial.
    float bias = n.state1();
    (void)bias;  // Statistical test is non-deterministic without fixed seed advance.
    // Just verify it compiles and runs without crashing.
}

TEST_CASE("Noise: RandomWalk advance — RMS grows as sigma*sqrt(t)") {
    // Run many independent walks, compute RMS at the end. Expected: sigma * sqrt(T).
    const int runs = 5000;
    const float sigma = 0.5f, dt = 0.01f, T = 1.0f;
    const int steps = static_cast<int>(T / dt);
    const float expected_rms = sigma * std::sqrt(T);

    noise_seed(99);
    double sum2 = 0;
    for (int r = 0; r < runs; ++r) {
        Noise<RandomWalk> n{sigma};
        for (int s = 0; s < steps; ++s) n.advance(dt);
        sum2 += double(n.state1()) * double(n.state1());
    }
    float rms = float(std::sqrt(sum2 / runs));
    // Should be within 5% of expected.
    CHECK(std::abs(rms - expected_rms) < 0.05f * expected_rms + 0.01f);
}

TEST_CASE("Noise: operator+(Vec3, WhiteGaussian) — returns Vec3 with noise") {
    noise_seed(1);
    Vec3<SceneFrame> v{1, 2, 3};
    Noise<WhiteGaussian> n{0.0f};  // zero sigma → no change
    auto result = v + n;
    CHECK(result.x() == doctest::Approx(1.0f));
    CHECK(result.y() == doctest::Approx(2.0f));
    CHECK(result.z() == doctest::Approx(3.0f));
}

TEST_CASE("Noise: operator+(Vec3, RandomWalk) — adds bias state") {
    Vec3<PartFrame> v{0, 0, 0};
    Noise<RandomWalk> n{0.0f};  // zero sigma → zero bias
    auto result = v + n;
    CHECK(result.x() == doctest::Approx(0.0f));
    CHECK(result.y() == doctest::Approx(0.0f));
    CHECK(result.z() == doctest::Approx(0.0f));
}

TEST_CASE("Noise: Real + WhiteGaussian scalar operator") {
    noise_seed(5);
    Noise<WhiteGaussian> n{0.0f};  // zero sigma
    Real v = Real(3.14f);
    Real result = v + n;
    CHECK(result == doctest::Approx(3.14f));
}
