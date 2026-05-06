#pragma once

// NoiseRegistry — collects per-part white-noise sources at EKF bind time
// and assembles the diagonal of the noise-input covariance Σ each tick.
//
// Used by `manta::estimation::EKF` to implement automatic Q assembly.
// At bind, the EKF walks the Jet-side part tree calling
// `PartT::register_noise(reg)` on each part. Parts hand their
// `Noise<WhiteGaussian>` instances to the registry, which assigns each
// one a contiguous slot block (3 slots for a 3-axis Vec noise, 1 slot
// for a scalar noise).
//
// Each tick during predict, the EKF queries `sigma_diag(dt, ...)` to
// build the per-input covariance diagonal `diag(σᵢ²·dt)`. The Jet
// world's noise inputs propagate through the dynamics as additional
// Jet columns; the EKF reads back the noise-input gain matrix L from
// those Jet columns and assembles
//
//     Q = L · diag(σᵢ²·dt) · Lᵀ
//
// State-dependent σ is supported: the registry calls `source->sigma()`
// fresh each tick, so a part that mutated `set_sigma()` from its
// `update()` produces the correct Q on the next predict.

#include <vector>

#include "noise.hpp"

namespace manta {

class NoiseRegistry {
public:
    // Register a 3-axis white-noise source. Allocates 3 contiguous slots
    // and stores the slot start on the source. Returns the slot start
    // (also written into the source via set_slot).
    int register_white_3d(Noise<WhiteGaussian>& source) {
        const int start = next_slot_;
        source.set_slot(start);
        entries_.push_back(Entry{&source, /*dim=*/3, start});
        next_slot_ += 3;
        return start;
    }

    // Register a scalar white-noise source. 1 slot.
    int register_white_1d(Noise<WhiteGaussian>& source) {
        const int start = next_slot_;
        source.set_slot(start);
        entries_.push_back(Entry{&source, /*dim=*/1, start});
        next_slot_ += 1;
        return start;
    }

    // Register a 3-axis random-walk bias source. Allocates 3 contiguous
    // bias-state slots AND 3 contiguous driver slots in the white-noise
    // input range. Updates the source's state_slot/driver_slot.
    int register_random_walk_3d(Noise<RandomWalk>& source) {
        const int driver_start = next_slot_;
        const int bias_start   = next_bias_slot_;
        source.set_driver_slot(driver_start);
        source.set_state_slot (bias_start);
        rw_entries_.push_back(RWEntry{&source, /*dim=*/3, driver_start, bias_start});
        next_slot_      += 3;
        next_bias_slot_ += 3;
        return driver_start;
    }

    int num_slots()      const noexcept { return next_slot_; }
    int num_bias_slots() const noexcept { return next_bias_slot_; }
    std::size_t source_count()    const noexcept { return entries_.size(); }
    std::size_t rw_source_count() const noexcept { return rw_entries_.size(); }

    // RW noise inspection — used by the EKF for bias evolution
    // (F's identity bias rows + L's σ_rw·√dt driver gain) and δ injection.
    struct RWAccess {
        Noise<RandomWalk>* source;
        int                dim;
        int                driver_slot;
        int                bias_slot;
    };
    std::vector<RWAccess> rw_sources() const {
        std::vector<RWAccess> out;
        out.reserve(rw_entries_.size());
        for (const auto& e : rw_entries_) {
            out.push_back({e.source, e.dim, e.driver_slot, e.bias_slot});
        }
        return out;
    }

    // Shift every registered source's external slots by the EKF's
    // global offsets. Called once at bind time:
    //
    //   noise_input_offset: where the noise-input range starts in the
    //                       Jet width (= 12·N + BiasDim).
    //   bias_state_offset:  where the bias-state range starts in the
    //                       augmented tangent (= 12·N).
    //
    // The registry's internal indexing (driver_start, bias_start in
    // entries) stays registry-local so sigma_squared_diag and friends
    // remain indexable.
    void apply_slot_offsets(int noise_input_offset, int bias_state_offset) {
        for (auto& e : entries_) {
            e.source->set_slot(e.source->slot() + noise_input_offset);
        }
        for (auto& e : rw_entries_) {
            e.source->set_driver_slot(e.source->driver_slot() + noise_input_offset);
            e.source->set_state_slot (e.source->state_slot()  + bias_state_offset);
        }
    }

    // Backward-compat shim used during early migration. Prefer
    // `apply_slot_offsets(noise_offset, bias_offset)` going forward.
    void apply_slot_offset(int offset) {
        apply_slot_offsets(offset, /*bias_state_offset=*/0);
    }

    // Per-tick variance of the unit noise inputs at each slot, into
    // `out` (resized to num_slots()).
    //
    // For Noise<WhiteGaussian>: the σ scaling is already inside the
    // noise-input gain L (the operator+ injects `Jet(0)·σ`), so the
    // input itself is unit-variance and this returns 1.0 per slot.
    // The resulting auto-Q is L·diag(1)·Lᵀ = L·Lᵀ — which works out to
    // L_eff·diag(σ²)·L_effᵀ where L_eff = L/σ is the unit-input gain,
    // i.e. the standard Kalman discrete Q.
    //
    // RandomWalk biases (Phase E) will return σ²·dt per slot instead;
    // this method is the per-noise-policy hook for that.
    //
    // dt is unused for WhiteGaussian (per-tick discrete noise has σ as
    // its per-tick stddev directly) but accepted for consistency with
    // future RandomWalk slots.
    void input_variance_diag(double /*dt*/, std::vector<double>& out) const {
        out.assign(static_cast<std::size_t>(next_slot_), 1.0);
    }

    // Clear all slots — flips every registered source back to "no slot."
    // Lets a filter rebind cleanly.
    void clear() {
        for (auto& e : entries_)    e.source->clear_slot();
        for (auto& e : rw_entries_) e.source->clear_slots();
        entries_.clear();
        rw_entries_.clear();
        next_slot_      = 0;
        next_bias_slot_ = 0;
    }

private:
    struct Entry {
        Noise<WhiteGaussian>* source;
        int                   dim;
        int                   slot_start;
    };
    struct RWEntry {
        Noise<RandomWalk>* source;
        int                dim;
        int                driver_slot;   // local index in noise-input range
        int                bias_slot;     // local index in bias-state range
    };

    std::vector<Entry>   entries_;
    std::vector<RWEntry> rw_entries_;
    int                  next_slot_      = 0;
    int                  next_bias_slot_ = 0;
};

} // namespace manta
