#pragma once

#include <Eigen/Core>
#include <ceres/jet.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "../geom/vec3.hpp"
#include "types.hpp"

namespace manta {

// ---- Type trait: is this Scalar a Ceres Jet? ----

namespace detail {
template <class T> struct is_ceres_jet : std::false_type {};
template <class T, int N> struct is_ceres_jet<ceres::Jet<T, N>> : std::true_type {};
} // namespace detail

template <class T>
inline constexpr bool is_ceres_jet_v = detail::is_ceres_jet<T>::value;

// ---- Noise policies (value types — carry only the scalar parameters) ---

struct WhiteGaussian {
    float sigma = 0.0f;
    explicit WhiteGaussian(float s = 0.0f) : sigma(s) {}
};

// RandomWalk is templated on its dimension. `Dim=3` is the typical case
// (3-axis IMU bias); `Dim=1` covers scalar biases (single-DOF joint
// bias). Other dimensions are valid but the operator+ only ships
// overloads for Vec3 (Dim=3) and Scalar (Dim=1) — Dim=2/4/N would need
// new operator+ overloads.
template <int Dim>
struct RandomWalk {
    static_assert(Dim >= 1, "RandomWalk dimension must be >= 1");
    float sigma = 0.0f;   // per-second diffusion coefficient
    explicit RandomWalk(float s = 0.0f) : sigma(s) {}
};

// ---- RNG seed helper ---
// Sets the thread-local RNG seed. Call at test/scenario start for determinism.
void noise_seed(std::uint32_t seed) noexcept;

// Internal: draw one N(0,1) sample from the thread-local RNG.
float noise_rng_next() noexcept;

// Sentinel: this noise source has not been registered with an EKF.
inline constexpr int kNoNoiseSlot = -1;

// ---- Noise<WhiteGaussian> ---
//
// A 3-axis IID white-Gaussian noise source. On the value path (Scalar =
// MFloat / double) the operator+ samples 3 N(0,σ²) draws and adds them to
// a Vec3. On the autodiff path (Scalar = ceres::Jet) the same operator+
// either:
//
//   * injects a Jet input at the registered slot (when bound to an EKF
//     that called register_noise()); each component i of the Vec3 picks
//     up a Jet derivative of magnitude σ in slot [base+i]. The EKF then
//     reads back the noise-input gain L from those Jet columns and
//     assembles Q = L · diag(σ²·dt) · Lᵀ;
//   * returns the input unchanged when no slot is registered (the typical
//     case for a manta-aware filter that hasn't opted into auto-Q, or for
//     part code running in the value path of a non-EKF context).
//
// State-dependent sigma: the σ field is mutable. Parts can compute σ each
// tick from internal state (e.g. an IMU's temperature) and the EKF picks
// up the latest σ when assembling Q. Just call `set_sigma()` from your
// part's update() before the noise is added.

template <typename Policy>
class Noise;

template <>
class Noise<WhiteGaussian> {
public:
    Noise() noexcept = default;
    explicit Noise(float sigma) noexcept : policy_(sigma) {}
    explicit Noise(WhiteGaussian p) noexcept : policy_(p) {}

    float sigma() const noexcept { return policy_.sigma; }
    void  set_sigma(float s) noexcept { policy_.sigma = s; }
    // Pointer to the σ field — used by `Measurement` to read R live
    // (state-dependent σ flows through automatically).
    const float* sigma_ptr() const noexcept { return &policy_.sigma; }

    // EKF-side: assign the global noise-input slot. The first slot covers
    // axis 0; axes 1 and 2 use slot+1 and slot+2 respectively.
    int  slot() const noexcept { return slot_; }
    bool slot_assigned() const noexcept { return slot_ >= 0; }
    void set_slot(int s) noexcept { slot_ = s; }
    void clear_slot() noexcept { slot_ = kNoNoiseSlot; }

    // Draw 3 IID samples from N(0, sigma²).
    Eigen::Matrix<float, 3, 1> sample3() const noexcept {
        const float s = policy_.sigma;
        return {noise_rng_next() * s,
                noise_rng_next() * s,
                noise_rng_next() * s};
    }

    float sample1() const noexcept {
        return noise_rng_next() * policy_.sigma;
    }

private:
    WhiteGaussian policy_;
    int           slot_ = kNoNoiseSlot;
};

// ---- Noise<RandomWalk<Dim>> ---
//
// A bias that drifts under a white-noise driver of strength σ_rw (units:
// per-second PSD). On the value/sim path `advance(dt)` integrates the
// bias forward via σ·√dt scaled draws. The current bias value is added
// to the input by operator+ (the "+ noise" sums in the current bias).
//
// On the autodiff/EKF path, when registered with an EKF that has
// BiasDim > 0, the bias becomes an estimated filter state:
//
//   * `state_slot()` is the bias's slot in the EKF's augmented tangent
//     state (rows [12·N + state_slot] in F and P).
//   * `driver_slot()` is the bias's white-noise driver slot in the Jet
//     noise-input range, used for the bias evolution
//     `bias_post = bias_pre + driver · σ_rw · √dt`.
//   * `state()` holds the EKF's *current bias estimate* on the Jet-side.
//     The operator+ Jet path injects this value with an identity Jet in
//     the bias state slot.
//   * Each Kalman update applies its bias delta into state() via the
//     EKF's inject_delta path.
//
// On the value-side, state() holds the *simulated* true bias (which the
// user's sim drifts via `advance`). The two sides have independent
// `Noise<RandomWalk<Dim>>` instances; the EKF estimates the truth from
// measurements.

// Type-erased base used by NoiseRegistry to store heterogeneous-Dim
// random walks in a single container, and to expose a uniform read/write
// interface to EKF/UKF code that operates on δ-slices generically.
class NoiseRandomWalkBase {
public:
    virtual ~NoiseRandomWalkBase() = default;

    virtual int   dim()        const noexcept = 0;
    virtual float sigma()      const noexcept = 0;
    virtual void  set_sigma(float s) noexcept = 0;

    virtual int   state_slot()  const noexcept = 0;
    virtual int   driver_slot() const noexcept = 0;
    virtual void  set_state_slot(int)  noexcept = 0;
    virtual void  set_driver_slot(int) noexcept = 0;
    virtual void  clear_slots() noexcept = 0;

    virtual void  advance(float dt) noexcept = 0;

    // Direct pointer to the state buffer (Dim contiguous floats). Used
    // by the EKF to inject Kalman δ-corrections per-axis without
    // virtual-dispatching on every element.
    virtual       float* state_data()       noexcept = 0;
    virtual const float* state_data() const noexcept = 0;
};

template <int Dim>
class Noise<RandomWalk<Dim>> : public NoiseRandomWalkBase {
public:
    using StateVec = Eigen::Matrix<float, Dim, 1>;

    Noise() noexcept = default;
    explicit Noise(float sigma) noexcept : policy_(sigma) {}
    explicit Noise(RandomWalk<Dim> p) noexcept : policy_(p) {}

    int   dim()        const noexcept override { return Dim; }
    float sigma()      const noexcept override { return policy_.sigma; }
    void  set_sigma(float s) noexcept override { policy_.sigma = s; }

    int  state_slot()  const noexcept override { return state_slot_;  }
    int  driver_slot() const noexcept override { return driver_slot_; }
    bool state_slot_assigned()  const noexcept { return state_slot_  >= 0; }
    bool driver_slot_assigned() const noexcept { return driver_slot_ >= 0; }
    void set_state_slot(int s)  noexcept override { state_slot_  = s; }
    void set_driver_slot(int s) noexcept override { driver_slot_ = s; }
    void clear_slots() noexcept override {
        state_slot_  = kNoNoiseSlot;
        driver_slot_ = kNoNoiseSlot;
    }

    const StateVec& state() const noexcept { return state_; }
    StateVec&       state()       noexcept { return state_; }
    void set_state(const StateVec& s) noexcept { state_ = s; }

          float* state_data()       noexcept override { return state_.data(); }
    const float* state_data() const noexcept override { return state_.data(); }

    // Advance the random walk by dt seconds: state += N(0, σ² · dt).
    void advance(float dt) noexcept override {
        const float s = policy_.sigma * std::sqrt(dt);
        for (int i = 0; i < Dim; ++i) {
            state_(i) += s * noise_rng_next();
        }
    }

private:
    RandomWalk<Dim> policy_;
    StateVec        state_       = StateVec::Zero();
    int             state_slot_  = kNoNoiseSlot;
    int             driver_slot_ = kNoNoiseSlot;
};

// ---- operator+(Vec3<F, Scalar>, Noise) and operator+(Scalar, Noise) ---
//
// Behavior depends on Scalar:
//
//   * Floating-point Scalar (MFloat, double): sample noise and add. Sim
//     path. State-dependent σ picks up the latest sigma() value.
//
//   * Ceres Jet Scalar with a registered noise slot: inject a Jet input
//     of magnitude σ at the registered slot. The EKF reads the noise-
//     input gain L back from these Jet columns and assembles Q.
//
//   * Ceres Jet Scalar without a registered slot: pass-through. The
//     value path's noise sample wouldn't be reproducible under autodiff
//     anyway; for filters that haven't opted into auto-Q this is the
//     same behavior as before.

template <typename F, typename Scalar>
geom::Vec3<F, Scalar> operator+(const geom::Vec3<F, Scalar>& v,
                                const Noise<WhiteGaussian>& n) noexcept {
    // σ < 0 ⇒ noise channel is disabled (matches the register_noise()
    // skip convention). The value-side path returns the input unchanged
    // so a "default-disabled" noise source doesn't perturb measurements.
    if (n.sigma() < 0.0f) return v;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return geom::Vec3<F, Scalar>::from_raw(
            v.raw() + n.sample3().template cast<Scalar>());
    } else if constexpr (is_ceres_jet_v<Scalar>) {
        if (!n.slot_assigned() || n.sigma() == 0.0f) return v;
        const int s = n.slot();
        const double sigma = static_cast<double>(n.sigma());
        Eigen::Matrix<Scalar, 3, 1> r = v.raw();
        Scalar n0(0.0); n0.v[s + 0] = sigma; r(0) += n0;
        Scalar n1(0.0); n1.v[s + 1] = sigma; r(1) += n1;
        Scalar n2(0.0); n2.v[s + 2] = sigma; r(2) += n2;
        return geom::Vec3<F, Scalar>::from_raw(r);
    } else {
        (void)n;
        return v;
    }
}

// Vec3 + 3-axis random walk. Templated only for Dim=3 — adding a Vec3
// to a Dim=1 RW would mix dimensions.
template <typename F, typename Scalar>
geom::Vec3<F, Scalar> operator+(const geom::Vec3<F, Scalar>& v,
                                const Noise<RandomWalk<3>>& n) noexcept {
    // σ < 0 ⇒ disabled (see WhiteGaussian operator+).
    if (n.sigma() < 0.0f) return v;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return geom::Vec3<F, Scalar>::from_raw(
            v.raw() + n.state().template cast<Scalar>());
    } else if constexpr (is_ceres_jet_v<Scalar>) {
        // Inject the EKF's current bias estimate as a Jet:
        //   value channel: + state[i]
        //   tangent channel: identity in the bias state slot
        // so any measurement that reads this picks up dependence on the
        // estimated bias state, and the EKF's H matrix's bias column
        // entries are populated automatically.
        if (!n.state_slot_assigned()) return v;
        const int s = n.state_slot();
        const Eigen::Matrix<float, 3, 1>& b = n.state();
        Eigen::Matrix<Scalar, 3, 1> r = v.raw();
        Scalar j0(static_cast<double>(b(0))); j0.v[s + 0] = 1.0; r(0) += j0;
        Scalar j1(static_cast<double>(b(1))); j1.v[s + 1] = 1.0; r(1) += j1;
        Scalar j2(static_cast<double>(b(2))); j2.v[s + 2] = 1.0; r(2) += j2;
        return geom::Vec3<F, Scalar>::from_raw(r);
    } else {
        (void)n;
        return v;
    }
}

// Scalar overloads — same dispatch.
template <typename Scalar>
Scalar operator+(Scalar v, const Noise<WhiteGaussian>& n) noexcept {
    if (n.sigma() < 0.0f) return v;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return v + Scalar(n.sample1());
    } else if constexpr (is_ceres_jet_v<Scalar>) {
        if (!n.slot_assigned() || n.sigma() == 0.0f) return v;
        Scalar n0(0.0);
        n0.v[n.slot()] = static_cast<double>(n.sigma());
        return v + n0;
    } else {
        (void)n;
        return v;
    }
}

// Scalar + 1-axis random walk.
template <typename Scalar>
Scalar operator+(Scalar v, const Noise<RandomWalk<1>>& n) noexcept {
    if (n.sigma() < 0.0f) return v;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return v + Scalar(n.state()(0));
    } else if constexpr (is_ceres_jet_v<Scalar>) {
        if (!n.state_slot_assigned()) return v;
        Scalar j(static_cast<double>(n.state()(0)));
        j.v[n.state_slot()] = 1.0;
        return v + j;
    } else {
        (void)n;
        return v;
    }
}

} // namespace manta
